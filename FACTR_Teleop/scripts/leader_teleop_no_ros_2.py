#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader Teleop (No ROS2, high-fidelity control):
- 和原本的无ROS版本一样，使用硬编码路径、ZeroMQ通信、Dynamixel驱动、while循环定频。
- 但控制算法尽量还原 ROS 版本的完整逻辑：
    * Null space 投影控制 (J, pinv, N = I - J⁺J)
    * 关节限位屏障力矩
    * 重力补偿 (Pinocchio RNEA) + 静摩擦补偿
    * 力反馈带阻尼项
    * 夹爪反馈通道接口（占位，暂不强推力矩）
- 依旧：
    * 读取领导端 Dynamixel 关节位置/速度
    * ZeroMQ 发布7维关节角到6001，夹爪到6004
    * ZeroMQ 非阻塞订阅6003(外力矩)，没数据不阻塞
- 不依赖 ROS2
"""

import os
import sys
sys.path.append('/home/galbot/zyf/FACTR_Teleop/src/factr_teleop')
sys.path.append('/home/galbot/zyf/FACTR_Teleop/src/factr_teleop/factr_teleop/urdf')
sys.path.append('/home/galbot/zyf/FACTR_Teleop/src/factr_teleop/factr_teleop/dynamixel/python/src')

import time
import subprocess
import numpy as np
import zmq
import pinocchio as pin
from typing import Optional

from factr_teleop.dynamixel.driver import DynamixelDriver  # noqa: E402


# ====== 配置（沿用你原来的无ROS版本常量） ======

DYNAMIXEL_PORT_ID = "usb-FTDI_USB__-__Serial_Converter_FTAA0992-if00-port0"
LEADER_URDF       = "/home/galbot/zyf/FACTR_Teleop/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf"

CTRL_FREQ = 500.0
DT = 1.0 / CTRL_FREQ

NUM_ARM_JOINTS = 7

SERVO_TYPES = [
    "XC330_T288_T", "XM430_W210_T", "XC330_T288_T", "XM430_W210_T",
    "XC330_T288_T", "XC330_T288_T", "XC330_T288_T", "XC330_T288_T",
]
# 7个手臂关节 + 1个夹爪
JOINT_SIGNS = np.array([1, 1, 1, -1, 1, -1, 1, -1], dtype=float)

# 关节限位（和你无ROS版本一致）+ 安全裕度 0.1
ARM_LIMIT_MAX = np.array([2.8973, 1.7628, 2.8973, -0.8698, 2.8973, 3.7525, 2.8973]) - 0.1
ARM_LIMIT_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) + 0.1

GRIPPER_MIN, GRIPPER_MAX = -0., 0.4  # 夹爪范围

# 标定姿态（校零用）和开机配平位姿（要求手动对上）
CALIB_POS = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0])
MATCH_POS = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0])

# --- 控制增益（保留你无ROS版的数值） ---
# 重力补偿
GRAVITY_COMP_ENABLE = True
GRAVITY_COMP_GAIN   = 0.8  # 在ROS版里是 gravity_comp_modifier

# Null-space 目标姿态（长度给到8，最后元素可为夹爪占位）
NULL_TARGET = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0, 0.0])
NULL_KP, NULL_KD = 0.1, 0.01  # 用于null-space调节

# 静摩擦补偿
STICTION_ENABLE_SPEED = 0.9
STICTION_GAIN         = 0.6

# 关节限位屏障增益
LIMIT_KP, LIMIT_KD = 1.03561658022, 0.04315069084

# 力反馈（from follower）
TORQUE_FB_ENABLE       = True
TORQUE_FB_GAIN         = 3.0
TORQUE_FB_DAMPING      = 0.0
TORQUE_FB_MOTOR_SCALAR = 94.5652173913

# 夹爪反馈（目前保留接口，不强推）
GRIPPER_FB_ENABLE = True  # 和ROS版思路一致：开关由配置决定，这里先设True占位

# ZeroMQ 地址（保持你原有的端口）
ADDR_QCMD_PUB = "tcp://*:6001"          # 主臂 -> 从臂：7维关节角
ADDR_GCMD_PUB = "tcp://*:6004"          # 主臂 -> 从臂：夹爪开度
ADDR_TAU_SUB  = "tcp://127.0.0.1:6003"  # 从臂 -> 主臂：外关节力矩 (for haptics)
# 可选：夹爪反馈通道（如果以后需要，可以加，比如 tcp://127.0.0.1:6005）
ADDR_GRIP_FB_SUB = None  # 暂无


# ====== 系统工具函数（跟你无ROS版一致的做法） ======
def resolve_ttyusb(port_id: str) -> str:
    """
    /dev/serial/by-id/<port_id> -> ttyUSBx
    """
    p = "/dev/serial/by-id/" + port_id
    if not os.path.exists(p):
        raise RuntimeError(f"找不到 {p}")
    target = os.readlink(p)         # ../../ttyUSB0
    return os.path.basename(target) # ttyUSB0

def check_latency(ttyusb: str):
    """
    检查 latency_timer 是否为 1，避免高延迟导致控制循环崩
    """
    p = f"/sys/bus/usb-serial/devices/{ttyusb}/latency_timer"
    try:
        val = int(subprocess.check_output(["bash","-lc", f"cat {p}"]).decode().strip())
        if val != 1:
            print(f"[WARN] {p}={val}，建议设为1： echo 1 | sudo tee {p}")
    except Exception as e:
        print(f"[WARN] 读取 {p} 失败：{e}")

def build_zmq():
    """
    建立ZeroMQ:
    - PUB (6001): 7维关节角 (float64 bytes)
    - PUB (6004): 夹爪开度 (float64 bytes)
    - SUB (6003): 从臂外力矩 (float32/float64都容忍)
    - SUB (夹爪反馈): 可选
    """
    ctx = zmq.Context.instance()

    pub_q = ctx.socket(zmq.PUB); pub_q.setsockopt(zmq.SNDHWM, 5); pub_q.bind(ADDR_QCMD_PUB)
    print(f"[ZMQ] PUB 6001 bound to {ADDR_QCMD_PUB}")

    pub_g = ctx.socket(zmq.PUB); pub_g.setsockopt(zmq.SNDHWM, 5); pub_g.bind(ADDR_GCMD_PUB)
    print(f"[ZMQ] PUB 6004 bound to {ADDR_GCMD_PUB}")

    sub_tau = ctx.socket(zmq.SUB)
    sub_tau.setsockopt(zmq.SUBSCRIBE, b"")
    sub_tau.setsockopt(zmq.RCVHWM, 5)
    sub_tau.setsockopt(zmq.CONFLATE, 1)
    sub_tau.setsockopt(zmq.RCVTIMEO, 0)  # 非阻塞
    sub_tau.connect(ADDR_TAU_SUB)
    print(f"[ZMQ] SUB torque connected to {ADDR_TAU_SUB}")

    sub_grip = None
    if ADDR_GRIP_FB_SUB is not None:
        sub_grip = ctx.socket(zmq.SUB)
        sub_grip.setsockopt(zmq.SUBSCRIBE, b"")
        sub_grip.setsockopt(zmq.RCVHWM, 5)
        sub_grip.setsockopt(zmq.CONFLATE, 1)
        sub_grip.setsockopt(zmq.RCVTIMEO, 0)
        sub_grip.connect(ADDR_GRIP_FB_SUB)
        print(f"[ZMQ] SUB gripper fb connected to {ADDR_GRIP_FB_SUB}")

    return pub_q, pub_g, sub_tau, sub_grip

def send_array(pub, arr: np.ndarray):
    pub.send(np.asarray(arr, dtype=np.float64).tobytes())

def recv_array_any_nonblock(sub, expected_len: Optional[int] = None) -> Optional[np.ndarray]:
    """
    非阻塞读ZMQ一帧，并尝试按float32/64解析。
    如果 expected_len 给了，就优先挑长度匹配的。
    """
    try:
        buf = sub.recv(flags=zmq.NOBLOCK)
    except zmq.Again:
        return None

    # 先试 float32
    a32 = np.frombuffer(buf, dtype=np.float32)
    if expected_len is None:
        if a32.size in (1,7,8):
            return a32
    else:
        if a32.size == expected_len:
            return a32

    # 再试 float64
    a64 = np.frombuffer(buf, dtype=np.float64)
    if expected_len is not None and a64.size == expected_len:
        return a64.astype(np.float32)

    # 回退策略
    return a32 if a32.size > 0 else a64.astype(np.float32)


def pin_build(urdf_path: str):
    """
    pinocchio.buildModelsFromUrdf 可能依赖cwd定位mesh。
    为了稳定，暂时chdir到URDF目录再切回来。
    """
    urdf_dir = os.path.dirname(urdf_path)
    cwd = os.getcwd()
    os.chdir(urdf_dir)
    try:
        model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=[urdf_dir])
        data = model.createData()
    finally:
        os.chdir(cwd)
    return model, data


# ====== 控制子模块（这里改成ROS版同款策略） ======

def joint_limit_barrier(q: np.ndarray, dq: np.ndarray,
                        grip: float, dgrip: float):
    """
    和ROS版 joint_limit_barrier 等价：
    - 对超过上下限的关节，加弹簧-阻尼力矩。
    - 对夹爪也做上下限保护。
    """
    tau = np.zeros_like(q)

    exceed_max = q > ARM_LIMIT_MAX
    tau += (-LIMIT_KP * (q - ARM_LIMIT_MAX) - LIMIT_KD * dq) * exceed_max

    exceed_min = q < ARM_LIMIT_MIN
    tau += (-LIMIT_KP * (q - ARM_LIMIT_MIN) - LIMIT_KD * dq) * exceed_min

    # 夹爪部分
    if grip > GRIPPER_MAX:
        tau_grip = -LIMIT_KP * (grip - GRIPPER_MAX) - LIMIT_KD * dgrip*0.2
    elif grip < GRIPPER_MIN:
        tau_grip = -LIMIT_KP * (grip - GRIPPER_MIN) - LIMIT_KD * dgrip*0.2
    else:
        tau_grip = 0.0
    tau_grip = tau_grip*0.2
    return tau, tau_grip


def gravity_compensation(model, data, q: np.ndarray, dq: np.ndarray):
    """
    和ROS版 gravity_compensation一致：
    用 RNEA 算 tau_g，然后乘 GRAVITY_COMP_GAIN
    """
    tau_g = pin.rnea(model, data, q, dq, np.zeros_like(dq))
    tau_g = tau_g * GRAVITY_COMP_GAIN
    return tau_g


def friction_compensation(dq: np.ndarray, tau_g: np.ndarray,
                          stiction_flag: np.ndarray):
    """
    和ROS版 friction_compensation一致：
    低速关节交替 +/-, 幅值与 |tau_g| 成正比，帮助克服静摩擦卡滞。
    stiction_flag[i] 会在每次调用后翻转。
    """
    tau_ss = np.zeros_like(dq)
    for i in range(dq.size):
        if abs(dq[i]) < STICTION_ENABLE_SPEED:
            if stiction_flag[i]:
                tau_ss[i] += STICTION_GAIN * abs(tau_g[i])
            else:
                tau_ss[i] -= STICTION_GAIN * abs(tau_g[i])
            stiction_flag[i] = ~stiction_flag[i]
    return tau_ss


def nullspace_regulation(model, data,
                         q: np.ndarray, dq: np.ndarray):
    """
    这是ROS版的 null_space_regulation：
    - 计算末端雅可比 J
    - Moore-Penrose 伪逆 J_dagger
    - Null投影 N = I - J⁺J
    - PD往 NULL_TARGET 靠，但只在 null 空间里施力
    """
    # Pinocchio的 computeJointJacobian(model, data, q, joint_id)
    # joint_id 用 NUM_ARM_JOINTS 假定是末端关节索引
    J = pin.computeJointJacobian(model, data, q, NUM_ARM_JOINTS)
    J_dagger = np.linalg.pinv(J)
    N = np.eye(NUM_ARM_JOINTS) - J_dagger @ J

    q_des = NULL_TARGET[:NUM_ARM_JOINTS]
    q_err = q - q_des
    tau_n = N @ (-NULL_KP * q_err - NULL_KD * dq)
    return tau_n


def torque_feedback_term(ext_tau: np.ndarray, dq: np.ndarray):
    """
    ROS版 torque_feedback：
    - 把从臂外力矩映射成领导端的反作用力
    - 再加阻尼
    """
    tau_fb = -(TORQUE_FB_GAIN / TORQUE_FB_MOTOR_SCALAR) * ext_tau
    tau_fb += -(TORQUE_FB_DAMPING) * dq
    return tau_fb


def gripper_feedback_term(grip_pos: float, grip_vel: float,
                          follower_grip_feedback: float):
    """
    夹爪反馈接口。
    ROS版里它是抽象的，由子类实现。
    在本版本中，我们保留接口，但默认返回0（不对夹爪加力矩）。
    你可以改成比例控制，比如 kp*force - kd*vel.
    """
    if not GRIPPER_FB_ENABLE:
        return 0.0
    # 先占位，不施加夹爪力矩
    return 0.0


# ====== 主流程 ======
def main():
    # 1) 连接 Dynamixel
    tty = resolve_ttyusb(DYNAMIXEL_PORT_ID)
    check_latency(tty)

    joint_ids = np.arange(8) + 1  # 7关节+1夹爪
    driver = DynamixelDriver(joint_ids, SERVO_TYPES, "/dev/serial/by-id/" + DYNAMIXEL_PORT_ID)
    driver.set_torque_mode(False)
    driver.set_operating_mode(0)  # current mode
    driver.set_torque_mode(True)

    # 2) Pinocchio 模型
    if not os.path.exists(LEADER_URDF):
        raise FileNotFoundError(f"URDF 不存在：{LEADER_URDF}")
    model, data = pin_build(LEADER_URDF)

    # 3) ZeroMQ
    pub_q, pub_g, sub_tau, sub_grip = build_zmq()
    time.sleep(0.5)  # 处理ZMQ订阅慢加入

    # 4) 标定offset：和ROS版 _get_dynamixel_offsets()一致
    for _ in range(10):
        driver.get_positions_and_velocities()
    curr_pos, _ = driver.get_positions_and_velocities()

    def best_offset_for_joint(j_idx):
        best_off, best_err = 0.0, 1e9
        # 用 pi/2 步进在 [-20π, 20π] 搜索，使 (raw-off)*sign 接近 CALIB_POS
        for off in np.linspace(-20*np.pi, 20*np.pi, 81):
            joint_i = JOINT_SIGNS[j_idx] * (curr_pos[j_idx] - off)
            err = abs(joint_i - CALIB_POS[j_idx])
            if err < best_err:
                best_err, best_off = err, off
        return best_off

    joint_offsets = np.zeros(8)
    for i in range(NUM_ARM_JOINTS):
        joint_offsets[i] = best_offset_for_joint(i)
    joint_offsets[-1] = curr_pos[-1]  # 夹爪零点 = 当前值

    print("[INFO] offsets(deg):", np.round(joint_offsets * 180/np.pi, 2))

    # 5) 一些运行时状态
    stiction_flag = np.ones(NUM_ARM_JOINTS, dtype=bool)

    # 用来记录夹爪前一次位置，推导夹爪速度
    prev_grip_pos = 0.0
    last_tau_ext = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
    last_grip_feedback = 0.0

    # 6) 读当前关节状态的函数（和ROS版 get_leader_joint_states()等价）
    def get_states():
        nonlocal prev_grip_pos
        q_raw, dq_raw = driver.get_positions_and_velocities()

        # 领导臂 -> 从臂坐标: (raw - offset) * sign
        q_arm = (np.array(q_raw[:NUM_ARM_JOINTS]) - joint_offsets[:NUM_ARM_JOINTS]) * JOINT_SIGNS[:NUM_ARM_JOINTS]
        dq_arm = np.array(dq_raw[:NUM_ARM_JOINTS]) * JOINT_SIGNS[:NUM_ARM_JOINTS]

        grip_pos = (q_raw[-1] - joint_offsets[-1]) * JOINT_SIGNS[-1]
        grip_vel = (grip_pos - prev_grip_pos) / DT
        prev_grip_pos = grip_pos

        return q_arm, dq_arm, grip_pos, grip_vel

    # 7) 要求开机配平：和ROS版 _match_start_pos()一致
    q, dq, qg, dqg = get_states()
    while np.linalg.norm(q - MATCH_POS) > 0.6:
        print(f"[INFO] 请手动把领导端关节配平到初始位 (误差 {np.linalg.norm(q - MATCH_POS):.3f})")
        time.sleep(0.5)
        q, dq, qg, dqg = get_states()
    print("[INFO] 初始关节位置匹配完成。")

    print("[INFO] 进入控制循环(高保真算法，非ROS)...")

    # 控制主循环
    t_prev = time.perf_counter()
    send_count = 0
    last_log = time.time()

    try:
        while True:
            # 1. 读领导端当前状态
            q, dq, qg, dqg = get_states()

            # 2. 关节限位屏障力矩
            tau_lim, tau_lim_grip = joint_limit_barrier(q, dq, qg, dqg)

            # 3. Null space 调节 (投影版)
            tau_ns = nullspace_regulation(model, data, q, dq)

            # 4. 重力补偿 + 静摩擦补偿
            tau_arm = np.zeros(NUM_ARM_JOINTS, dtype=float)
            tau_arm += tau_lim
            tau_arm += tau_ns

            if GRAVITY_COMP_ENABLE:
                tau_g = gravity_compensation(model, data, q, dq)
                tau_arm += tau_g
                tau_arm += friction_compensation(dq, tau_g, stiction_flag)

            # 5. 力反馈（外部关节力矩 + 阻尼）
            if TORQUE_FB_ENABLE:
                tau_ext_arr = recv_array_any_nonblock(sub_tau, expected_len=NUM_ARM_JOINTS)
                if tau_ext_arr is not None and tau_ext_arr.size >= NUM_ARM_JOINTS:
                    last_tau_ext = tau_ext_arr[:NUM_ARM_JOINTS].astype(np.float32)
                tau_fb = torque_feedback_term(last_tau_ext, dq)
                tau_arm += tau_fb

            # 6. 夹爪反馈
            tau_grip = tau_lim_grip
            if GRIPPER_FB_ENABLE:
                if ADDR_GRIP_FB_SUB is not None:
                    grip_fb_arr = recv_array_any_nonblock(sub_grip, expected_len=1)
                    if grip_fb_arr is not None and grip_fb_arr.size >= 1:
                        last_grip_feedback = float(grip_fb_arr[0])
                tau_grip += gripper_feedback_term(qg, dqg, last_grip_feedback)

            # 7. 把合成的力矩下发到 Dynamixel
            #    注意：扩展到8个通道(7关节 + 1夹爪)
            full_tau = np.append(tau_arm, tau_grip) * JOINT_SIGNS
            driver.set_torque(full_tau)

            # 8. 把领导端状态发给从臂
            send_array(pub_q, q)              # 7x float64
            send_array(pub_g, np.array([qg])) # 1x float64

            # 打印一下循环频率
            send_count += 1
            now = time.time()
            if now - last_log > 1.0:
                print(f"[PUB] ->6001 rate≈{send_count}/s  q[0:3]={q[:3]}")
                send_count = 0
                last_log = now

            # 9. 定频睡眠（类似你无ROS版本）
            t_prev += DT
            sleep_time = t_prev - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 掉帧了，重置时钟，避免累计漂移
                t_prev = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C 收到，安全停机...")
    finally:
        # 和ROS版 shut_down() 一致的安全停机逻辑
        try:
            driver.set_torque(np.zeros(8))
            driver.set_torque_mode(False)
        except Exception as e:
            print(f"[WARN] 停机过程中 set_torque 失败：{e}")

        # 再尝试一次保险复位（和你无ROS版本最后的“reconnect尝试”类似）
        try:
            driver = DynamixelDriver(joint_ids, SERVO_TYPES, "/dev/serial/by-id/" + DYNAMIXEL_PORT_ID)
            driver.set_torque_mode(False)
            driver.set_operating_mode(0)
            driver.set_torque_mode(True)
            if getattr(driver, "torque_enabled", False):
                driver.set_torque([0]*8)
                driver.set_torque_mode(False)
            print("Dynamixel reconnected and torque set to zero")
        except Exception as e:
            print(f"Failed to reconnect Dynamixel: {e}")


if __name__ == "__main__":
    main()

