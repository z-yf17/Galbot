#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader Teleop (NO ROS2):
- 读取领导端 Dynamixel 关节位置/速度
- Pinocchio 做重力补偿 + （可选）静摩擦/限位/零空间
- ZMQ 发布 7 关节目标到 6001、夹爪命令到 6004（按 float64 发送）
- ZMQ 非阻塞订阅 6003（跟随端外力矩），没数据也不阻塞控制循环
- 不依赖 ROS2
"""
import os
import time
import subprocess
import numpy as np
import zmq
import pinocchio as pin

# ====== 配置 ======
DYNAMIXEL_PORT_ID = "usb-FTDI_USB__-__Serial_Converter_FTAA0992-if00-port0"
LEADER_URDF = "/workspace/factr_ws/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf"

CTRL_FREQ = 500.0
DT = 1.0 / CTRL_FREQ

NUM_ARM_JOINTS = 7
SERVO_TYPES = [
    "XC330_T288_T", "XM430_W210_T", "XC330_T288_T", "XM430_W210_T",
    "XC330_T288_T", "XC330_T288_T", "XC330_T288_T", "XC330_T288_T",
]
JOINT_SIGNS = np.array([1, 1, 1, -1, 1, -1, 1, -1], dtype=float)  # 7臂+1夹爪

ARM_LIMIT_MAX = np.array([2.8973, 1.7628, 2.8973, -0.8698, 2.8973, 3.7525, 2.8973]) - 0.1
ARM_LIMIT_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) + 0.1
GRIPPER_MIN, GRIPPER_MAX = 0.0, 0.8

CALIB_POS = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0])
MATCH_POS  = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0])

GRAVITY_COMP_ENABLE = True
GRAVITY_COMP_GAIN   = 0.8

NULL_KP, NULL_KD = 0.1, 0.01
NULL_TARGET = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0, 0.0])

STICTION_ENABLE_SPEED = 0.9
STICTION_GAIN         = 0.6

LIMIT_KP, LIMIT_KD = 1.03561658022, 0.04315069084

TORQUE_FB_ENABLE       = True
TORQUE_FB_GAIN         = 3.0
TORQUE_FB_DAMPING      = 0.0
TORQUE_FB_MOTOR_SCALAR = 94.5652173913

ADDR_QCMD_PUB = "tcp://*:6001"          # 领导→跟随：7维关节目标
ADDR_GCMD_PUB = "tcp://*:6004"          # 领导→跟随：夹爪命令/宽度
ADDR_TAU_SUB  = "tcp://127.0.0.1:6003"  # 跟随→领导：外力矩（力反馈）

from factr_teleop.dynamixel.driver import DynamixelDriver  # noqa: E402

# ------------------ 工具 ------------------
def resolve_ttyusb(port_id: str) -> str:
    p = "/dev/serial/by-id/" + port_id
    if not os.path.exists(p):
        raise RuntimeError(f"找不到 {p}")
    target = os.readlink(p)         # ../../ttyUSB0
    return os.path.basename(target) # ttyUSB0

def check_latency(ttyusb: str):
    p = f"/sys/bus/usb-serial/devices/{ttyusb}/latency_timer"
    try:
        val = int(subprocess.check_output(["bash","-lc", f"cat {p}"]).decode().strip())
        if val != 1:
            print(f"[WARN] {p}={val}，建议设 1： echo 1 | sudo tee {p}")
    except Exception as e:
        print(f"[WARN] 读取 {p} 失败：{e}")

def build_zmq():
    ctx = zmq.Context.instance()

    pub_q = ctx.socket(zmq.PUB); pub_q.setsockopt(zmq.SNDHWM, 5); pub_q.bind(ADDR_QCMD_PUB)
    print(f"[ZMQ] PUB 6001 bound to {ADDR_QCMD_PUB}")

    pub_g = ctx.socket(zmq.PUB); pub_g.setsockopt(zmq.SNDHWM, 5); pub_g.bind(ADDR_GCMD_PUB)
    print(f"[ZMQ] PUB 6004 bound to {ADDR_GCMD_PUB}")

    sub_tau = ctx.socket(zmq.SUB)
    sub_tau.setsockopt(zmq.SUBSCRIBE, b"")
    sub_tau.setsockopt(zmq.RCVHWM, 5)
    sub_tau.setsockopt(zmq.CONFLATE, 1)
    # 关键：设置非阻塞接收（通过 NOBLOCK 标志），这里给个很小的超时也可以
    sub_tau.setsockopt(zmq.RCVTIMEO, 0)  # 0 = 立即返回 EAGAIN
    sub_tau.connect(ADDR_TAU_SUB)
    print(f"[ZMQ] SUB torque connected to {ADDR_TAU_SUB}")

    return pub_q, pub_g, sub_tau

def send_array(pub, arr: np.ndarray):
    pub.send(np.asarray(arr, dtype=np.float64).tobytes())  # 与 torque_control.py 一致

def recv_array_any_nonblock(sub) -> np.ndarray | None:
    try:
        buf = sub.recv(flags=zmq.NOBLOCK)
    except zmq.Again:
        return None
    a32 = np.frombuffer(buf, dtype=np.float32)
    if a32.size in (1, 7, 8):
        return a32
    return np.frombuffer(buf, dtype=np.float64).astype(np.float32)

def pin_build(urdf_path: str):
    model, _, _ = pin.buildModelsFromUrdf(
        filename=urdf_path, package_dirs=os.path.dirname(urdf_path)
    )
    data = model.createData()
    return model, data

# ------------------ 控制子模块 ------------------
def joint_limit_barrier(q: np.ndarray, dq: np.ndarray):
    tau = np.zeros_like(q)
    exceed_max = q > ARM_LIMIT_MAX
    tau += (-LIMIT_KP * (q - ARM_LIMIT_MAX) - LIMIT_KD * dq) * exceed_max
    exceed_min = q < ARM_LIMIT_MIN
    tau += (-LIMIT_KP * (q - ARM_LIMIT_MIN) - LIMIT_KD * dq) * exceed_min
    return tau

def nullspace_pd(q: np.ndarray, dq: np.ndarray):
    target = NULL_TARGET[:NUM_ARM_JOINTS]
    return -NULL_KP * (q - target) - NULL_KD * dq

class StictionComp:
    def __init__(self, n): self.flag = np.ones(n, dtype=bool)
    def compute(self, dq: np.ndarray, tau_g: np.ndarray):
        tau = np.zeros_like(dq)
        for i in range(dq.size):
            if abs(dq[i]) < STICTION_ENABLE_SPEED:
                tau[i] += (STICTION_GAIN * abs(tau_g[i])) * (1.0 if self.flag[i] else -1.0)
                self.flag[i] = ~self.flag[i]
        return tau

# ------------------ 主流程 ------------------
def main():
    # 1) Dynamixel
    tty = resolve_ttyusb(DYNAMIXEL_PORT_ID); check_latency(tty)
    joint_ids = np.arange(8) + 1
    driver = DynamixelDriver(joint_ids, SERVO_TYPES, "/dev/serial/by-id/" + DYNAMIXEL_PORT_ID)
    driver.set_torque_mode(False); driver.set_operating_mode(0); driver.set_torque_mode(True)

    # 2) Pinocchio
    if not os.path.exists(LEADER_URDF):
        raise FileNotFoundError(f"URDF 不存在：{LEADER_URDF}")
    model, data = pin_build(LEADER_URDF)

    # 3) ZMQ
    pub_q, pub_g, sub_tau = build_zmq()
    time.sleep(0.5)  # 处理“慢加入”

    # 4) 标定 offset
    for _ in range(10): driver.get_positions_and_velocities()
    curr_pos, _ = driver.get_positions_and_velocities()

    def best_offset_for_joint(j_idx):
        best_off, best_err = 0.0, 1e9
        for off in np.linspace(-20*np.pi, 20*np.pi, 81):  # pi/2 步进搜索
            joint_i = JOINT_SIGNS[j_idx] * (curr_pos[j_idx] - off)
            err = abs(joint_i - CALIB_POS[j_idx])
            if err < best_err: best_err, best_off = err, off
        return best_off

    joint_offsets = np.zeros(8)
    for i in range(NUM_ARM_JOINTS): joint_offsets[i] = best_offset_for_joint(i)
    joint_offsets[-1] = curr_pos[-1]  # 夹爪零点为当前读数
    print("[INFO] offsets:", np.round(joint_offsets, 4))

    # 5) 初始配平
    def get_states():
        q_raw, dq_raw = driver.get_positions_and_velocities()
        q_arm = (np.array(q_raw[:NUM_ARM_JOINTS]) - joint_offsets[:NUM_ARM_JOINTS]) * JOINT_SIGNS[:NUM_ARM_JOINTS]
        dq_arm = np.array(dq_raw[:NUM_ARM_JOINTS]) * JOINT_SIGNS[:NUM_ARM_JOINTS]
        q_grip = (q_raw[-1] - joint_offsets[-1]) * JOINT_SIGNS[-1]
        return q_arm, dq_arm, q_grip

    q, dq, qg = get_states()
    while np.linalg.norm(q - MATCH_POS) > 0.6:
        print(f"[INFO] 请手动把领导端关节配平到初始位（误差 {np.linalg.norm(q - MATCH_POS):.3f}）")
        time.sleep(0.5)
        q, dq, qg = get_states()
    print("[INFO] 初始关节位置匹配完成。")

    # 6) 控制循环（非阻塞力反馈 + 心跳）
    stiction = StictionComp(NUM_ARM_JOINTS)
    t_prev = time.perf_counter()
    send_count, last_log = 0, time.time()
    print("[INFO] 进入控制循环（无 ROS2）...")
    try:
        while True:
            q, dq, qg = get_states()

            tau_g = pin.rnea(model, data, q, dq, np.zeros_like(dq)) * GRAVITY_COMP_GAIN if GRAVITY_COMP_ENABLE else np.zeros(NUM_ARM_JOINTS)
            tau_ss = stiction.compute(dq, tau_g) if GRAVITY_COMP_ENABLE else np.zeros_like(q)
            tau_lim = joint_limit_barrier(q, dq)
            tau_ns  = nullspace_pd(q, dq)

            tau_fb = np.zeros(NUM_ARM_JOINTS)
            if TORQUE_FB_ENABLE:
                tau_ext_arr = recv_array_any_nonblock(sub_tau)  # 非阻塞
                if tau_ext_arr is not None:
                    tau_ext = tau_ext_arr[:NUM_ARM_JOINTS]
                    tau_fb  = - (TORQUE_FB_GAIN / TORQUE_FB_MOTOR_SCALAR) * tau_ext - TORQUE_FB_DAMPING * dq

            tau_cmd_arm = tau_g + tau_ss + tau_lim + tau_ns + tau_fb
            driver.set_torque(np.append(tau_cmd_arm, 0.0) * JOINT_SIGNS)

            send_array(pub_q, q)              # 7 x float64
            send_array(pub_g, np.array([qg])) # 1 x float64

            send_count += 1
            now = time.time()
            if now - last_log > 1.0:
                print(f"[PUB] ->6001 rate≈{send_count}/s  q[0:3]={q[:3]}")
                send_count, last_log = 0, now

            t_prev += DT
            sleep = t_prev - time.perf_counter()
            if sleep > 0: time.sleep(sleep)
            else: t_prev = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C 收到，安全停机...")
    finally:
        try:
            driver.set_torque(np.zeros(8))
            driver.set_torque_mode(False)
        except Exception as e:
            print(f"[WARN] 停机过程中 set_torque 失败：{e}")

if __name__ == "__main__":
    main()

