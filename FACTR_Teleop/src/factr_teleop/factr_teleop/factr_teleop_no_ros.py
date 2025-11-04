#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader Teleop (NO ROS2, single-file):
- 纯 Python：Dynamixel 串口驱动 + Pinocchio 重力补偿 + ZMQ 通信
- 不依赖 ROS2/rclpy
- 发布：6001(7 关节 q, float64) / 6004(夹爪宽度, float64)
- 订阅：6003(跟随端外力矩 float32，非阻塞)

依赖 pip/conda:
  conda install -c conda-forge pinocchio pyzmq numpy dynamixel_sdk urdfdom
或
  pip install numpy pyzmq dynamixel-sdk
  pip install pinocchio  # 可能需要系统库：sudo apt-get install -y liburdfdom-tools liburdfdom-dev
"""

import os
import time
import struct
import subprocess
from typing import List, Tuple

import numpy as np
import zmq
import pinocchio as pin

# ====================== 用户区：基础配置 ======================
DYNAMIXEL_PORT_ID = "usb-FTDI_USB__-__Serial_Converter_FTAA0992-if00-port0"
SERIAL_BAUD = 1_000_000                      # 常见 57600 / 1_000_000
PROTO_VERSION = 2.0

LEADER_URDF = "/workspace/factr_ws/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf"

CTRL_FREQ = 500.0
DT = 1.0 / CTRL_FREQ

NUM_ARM_JOINTS = 7
SERVO_TYPES = [
    "XC330_T288_T", "XM430_W210_T", "XC330_T288_T", "XM430_W210_T",
    "XC330_T288_T", "XC330_T288_T", "XC330_T288_T", "XC330_T288_T",  # 最后一位是夹爪
]
JOINT_SIGNS = np.array([1, 1, 1, -1, 1, -1, 1, -1], dtype=float)  # 7臂 + 1夹爪

# 关节限位（臂 7 维，单位 rad）
ARM_LIMIT_MAX = np.array([2.8973, 1.7628, 2.8973, -0.8698, 2.8973, 3.7525, 2.8973]) - 0.1
ARM_LIMIT_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) + 0.1
GRIPPER_MIN, GRIPPER_MAX = 0.0, 0.8  # 若用宽度值，可在跟随端映射

# 初始配平
CALIB_POS = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0])
MATCH_POS  = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0])

# 重力补偿/零空间/限位/静摩擦
GRAVITY_COMP_ENABLE = True
GRAVITY_COMP_GAIN   = 0.8

NULL_KP, NULL_KD = 0.1, 0.01
NULL_TARGET = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0, 0.0])

STICTION_ENABLE_SPEED = 0.9
STICTION_GAIN         = 0.6

LIMIT_KP, LIMIT_KD = 1.03561658022, 0.04315069084

# 力反馈
TORQUE_FB_ENABLE       = True
TORQUE_FB_GAIN         = 3.0
TORQUE_FB_DAMPING      = 0.0
TORQUE_FB_MOTOR_SCALAR = 94.5652173913  # 跟随端→领导端的力矩到电机尺度的经验系数（仍保留你原数值）

# ZMQ 地址
ADDR_QCMD_PUB = "tcp://*:6001"          # 领导→跟随：7维关节目标
ADDR_GCMD_PUB = "tcp://*:6004"          # 领导→跟随：夹爪命令/宽度
ADDR_TAU_SUB  = "tcp://127.0.0.1:6003"  # 跟随→领导：外力矩（力反馈）

# ====================== Dynamixel 驱动（纯串口，无 ROS） ======================
# 寄存器地址（X 系列，Protocol 2.0）
ADDR_TORQUE_ENABLE     = 64     # 1 byte
ADDR_OPERATING_MODE    = 11     # 1 byte: 0=Current, 1=Velocity, 3=Position, 5=CurPos, 16=PWM
ADDR_GOAL_CURRENT      = 102    # 2 bytes (signed)
ADDR_PRESENT_CURRENT   = 126    # 2 bytes (signed)
ADDR_PRESENT_VELOCITY  = 128    # 4 bytes (signed)
ADDR_PRESENT_POSITION  = 132    # 4 bytes (signed)

# 单位换算（X 系列通用）
TICKS_PER_REV   = 4096.0                       # 0.088°/tick
RAD_PER_TICK    = 2*np.pi / TICKS_PER_REV
VEL_RPM_PER_TICK= 0.229                        # Present_Velocity: 0.229 rpm / tick
VEL_RADPS_PER_TICK = (VEL_RPM_PER_TICK * 2*np.pi / 60.0)
CURR_A_PER_TICK = 0.00269                      # Goal/Present Current: 2.69 mA / tick（约值）

# 型号映射（近似数，建议实测标定；用于 Nm -> A 映射：I = tau / Kt）
MODEL_KT_NM_PER_A = {
    "XM430_W210_T": 0.65,      # 近似；请按你电源电压/实际效率微调
    "XC330_T288_T": 0.30,      # 近似；小型伺服减速比不同
}
# 电流限幅（输出侧电流换算的上限），避免伤机（单位 A）
MODEL_I_LIMIT_A = {
    "XM430_W210_T": 1.8,
    "XC330_T288_T": 1.2,
}

def _to_signed(val: int, bits: int) -> int:
    mask = 1 << (bits - 1)
    return (val & (mask - 1)) - (val & mask)

class DynamixelDriver:
    """
    纯串口 Dynamixel 驱动（X 系列，Protocol 2.0）
    - 支持批量读位置/速度
    - 批量写电流（以 Nm 输入，按型号 Kt 换算到 A）
    - 可切换工作模式/开关扭矩
    """
    def __init__(self, ids: List[int], types: List[str], device: str, baud: int = SERIAL_BAUD):
        from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite

        self.ids = list(ids)
        self.types = list(types)
        assert len(self.ids) == len(self.types), "IDs 与型号长度不一致"

        # 端口
        dev_path = device if device.startswith("/dev/") else ("/dev/serial/by-id/" + device)
        if not os.path.exists(dev_path):
            raise FileNotFoundError(f"串口不存在：{dev_path}")

        self.port = PortHandler(dev_path)
        if not self.port.openPort():
            raise RuntimeError(f"打开串口失败：{dev_path}")
        if not self.port.setBaudRate(baud):
            raise RuntimeError(f"设置波特率失败：{baud}")

        self.ph = PacketHandler(PROTO_VERSION)

        # 批量读 present_position / present_velocity
        self.sync_pos = GroupSyncRead(self.port, self.ph, ADDR_PRESENT_POSITION, 4)
        self.sync_vel = GroupSyncRead(self.port, self.ph, ADDR_PRESENT_VELOCITY, 4)
        for dxl_id in self.ids:
            self.sync_pos.addParam(dxl_id)
            self.sync_vel.addParam(dxl_id)

        # 批量写 goal_current
        self.sync_curr = GroupSyncWrite(self.port, self.ph, ADDR_GOAL_CURRENT, 2)

        # 每轴 Kt/I 限
        self.kt_nm_per_a = np.array([MODEL_KT_NM_PER_A.get(t, 0.5) for t in self.types], dtype=float)
        self.i_limit_a   = np.array([MODEL_I_LIMIT_A.get(t, 1.2)      for t in self.types], dtype=float)

    # ---------- 低层写寄存器 ----------
    def _write1(self, dxl_id: int, addr: int, val: int):
        dxl_comm_result, dxl_error = self.ph.write1ByteTxRx(self.port, dxl_id, addr, val)
        if dxl_comm_result != 0 or dxl_error != 0:
            raise RuntimeError(f"DXL[{dxl_id}] write1 addr {addr} err: comm={dxl_comm_result}, stat={dxl_error}")

    def _write2(self, dxl_id: int, addr: int, val: int):
        dxl_comm_result, dxl_error = self.ph.write2ByteTxRx(self.port, dxl_id, addr, val & 0xFFFF)
        if dxl_comm_result != 0 or dxl_error != 0:
            raise RuntimeError(f"DXL[{dxl_id}] write2 addr {addr} err: comm={dxl_comm_result}, stat={dxl_error}")

    # ---------- 模式/扭矩 ----------
    def set_operating_mode(self, mode: int):
        """0=current(力矩), 1=速度, 3=位置, 5=电流叠加位置, 16=PWM"""
        # 切模式前先关扭矩
        self.set_torque_mode(False)
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_OPERATING_MODE, mode)
        # 切完可再开扭矩（由上层决定）

    def set_torque_mode(self, enable: bool):
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)

    # ---------- 读位置/速度 ----------
    def get_positions_and_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        # 触发读取
        self.sync_pos.txRxPacket()
        self.sync_vel.txRxPacket()

        q = np.zeros(len(self.ids), dtype=float)
        dq = np.zeros(len(self.ids), dtype=float)

        for i, dxl_id in enumerate(self.ids):
            if not self.sync_pos.isAvailable(dxl_id, ADDR_PRESENT_POSITION, 4):
                raise RuntimeError(f"DXL[{dxl_id}] pos not available")
            if not self.sync_vel.isAvailable(dxl_id, ADDR_PRESENT_VELOCITY, 4):
                raise RuntimeError(f"DXL[{dxl_id}] vel not available")

            raw_pos = self.sync_pos.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
            raw_vel = self.sync_vel.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4)

            # 两者都是 signed 32-bit
            pos_tick = _to_signed(raw_pos, 32)
            vel_tick = _to_signed(raw_vel, 32)

            q[i]  = pos_tick * RAD_PER_TICK
            dq[i] = vel_tick * VEL_RADPS_PER_TICK

        return q, dq

    # ---------- 写“力矩”命令（内部以电流写入） ----------
    def set_torque(self, tau_nm: np.ndarray):
        """
        接收每轴输出侧期望“力矩” tau (Nm)，按 Kt 映射到电流，
        写入 Goal_Current（单位：tick, 2.69mA/tick）。
        """
        tau_nm = np.asarray(tau_nm, dtype=float)
        assert tau_nm.shape[0] == len(self.ids), "tau 维度不符"

        i_cmd_a   = np.clip(tau_nm / self.kt_nm_per_a, -self.i_limit_a, self.i_limit_a)
        i_ticks   = np.clip(np.round(i_cmd_a / CURR_A_PER_TICK), -32768, 32767).astype(np.int16)

        # 组包（小端）
        self.sync_curr.clearParam()
        for dxl_id, it in zip(self.ids, i_ticks):
            param = struct.pack("<h", int(it))  # int16 little-endian
            if not self.sync_curr.addParam(dxl_id, param):
                raise RuntimeError(f"DXL[{dxl_id}] addParam failed")
        dxl_comm_result = self.sync_curr.txPacket()
        if dxl_comm_result != 0:
            raise RuntimeError(f"sync write current failed: comm={dxl_comm_result}")

    def close(self):
        try:
            self.set_torque_mode(False)
        except Exception:
            pass
        try:
            self.port.closePort()
        except Exception:
            pass

# ====================== 工具函数 ======================
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
    sub_tau.setsockopt(zmq.RCVTIMEO, 0)  # 非阻塞
    sub_tau.connect(ADDR_TAU_SUB)
    print(f"[ZMQ] SUB torque connected to {ADDR_TAU_SUB}")

    return pub_q, pub_g, sub_tau

def send_array(pub, arr: np.ndarray):
    pub.send(np.asarray(arr, dtype=np.float64).tobytes())  # 与跟随端约定：float64

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

# ====================== 控制子模块 ======================
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

# ====================== 主流程 ======================
def main():
    # 1) Dynamixel
    tty = resolve_ttyusb(DYNAMIXEL_PORT_ID); check_latency(tty)
    joint_ids = np.arange(8) + 1
    driver = DynamixelDriver(joint_ids.tolist(), SERVO_TYPES, "/dev/serial/by-id/" + DYNAMIXEL_PORT_ID, SERIAL_BAUD)
    # 切到电流模式（力矩），再开扭矩
    driver.set_operating_mode(0)
    driver.set_torque_mode(True)

    # 2) Pinocchio
    if not os.path.exists(LEADER_URDF):
        raise FileNotFoundError(f"URDF 不存在：{LEADER_URDF}")
    model, data = pin_build(LEADER_URDF)
    # 如需自定义重力方向可设：model.gravity.linear = np.array([0, 0, -9.81])

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

    # 6) 控制循环
    stiction = StictionComp(NUM_ARM_JOINTS)
    t_prev = time.perf_counter()
    send_count, last_log = 0, time.time()
    print("[INFO] 进入控制循环（NO ROS2）...")
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
            # 发送到电机（内部按 Kt → A → Goal_Current）
            driver.set_torque(np.append(tau_cmd_arm, 0.0) * JOINT_SIGNS)

            # ZMQ 广播
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
            driver.close()
        except Exception as e:
            print(f"[WARN] 停机过程中处理失败：{e}")

if __name__ == "__main__":
    main()

