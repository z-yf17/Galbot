#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import signal
import math
import numpy as np
import torch

from polymetis import RobotInterface, GripperInterface

# ===================== 配置区 =====================

MODEL_PATH = "/path/to/your_policy.pt"   # TODO: 替换成你的模型路径
USE_TORCHSCRIPT = False                  # 如果是 torch.jit.trace/script 保存的, 设 True
DEVICE = "cpu"                           # 如果有 GPU 且模型支持，可设 "cuda"

ARM_IP = "127.0.0.1"
ARM_PORT = 50051
GRIP_IP = "127.0.0.1"
GRIP_PORT = 50052

CONTROL_HZ = 200                         # 控制频率（建议 100~500Hz；200Hz 对推理更稳）
DT = 1.0 / CONTROL_HZ

# Franka Panda 关节限幅（弧度）
# 参考常用范围；如你们有更严格的软限位，请替换为自身参数
JOINT_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# 速度限幅（目标增量的限幅）
MAX_JOINT_STEP = np.deg2rad(5.0)         # 每周期最大关节增量 5°，按需调
ACTION_SMOOTH_ALPHA = 0.2                # 指令一阶低通，0.0(不平滑)~1.0(强平滑)

# 外力/力矩保护
TAU_EXT_THRESH = 8.0                     # 任一关节外力矩超过则触发软停，按需调
TAU_FILTER_ALPHA = 0.3

# 夹爪开闭映射
GRIP_OPEN_WIDTH = 0.078                  # Franka 手爪最大开度 ~78mm
GRIP_CLOSE_CMD_SPEED = 0.2
GRIP_CLOSE_CMD_FORCE = 0.1
GRIP_OPEN_SPEED = 0.2
GRIP_OPEN_FORCE = 0.1

# 归一化（示例）
STATE_MEAN = None   # np.array([...])  # TODO: 如果训练时做了归一化，就填这里
STATE_STD  = None   # np.array([...])
# =================================================


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def exp_smooth(prev, new, alpha):
    if prev is None: 
        return new
    return (1 - alpha) * prev + alpha * new

def load_policy(path, use_ts=False, device="cpu"):
    if use_ts:
        policy = torch.jit.load(path, map_location=device)
    else:
        obj = torch.load(path, map_location=device)
        # 兼容两种保存方式：直接保存的 nn.Module 或 {'model': module}
        policy = obj.get("model", obj) if isinstance(obj, dict) else obj
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy

def preprocess(obs_np):
    """将 numpy obs 转成 torch，并做可选归一化"""
    x = obs_np.copy()
    if STATE_MEAN is not None and STATE_STD is not None:
        x = (x - STATE_MEAN) / (STATE_STD + 1e-8)
    x = torch.from_numpy(x).float().unsqueeze(0)  # [1, D]
    return x

def postprocess_action(act_np):
    """
    假定模型输出维度=8：前7个是关节目标/增量，第8个是抓手开合标量 [0,1]
    你可以改成 “输出即绝对关节角”，或 “输出关节速度/增量”等
    这里用 '相对当前角度的增量' 示例，并做限幅与平滑
    """
    assert act_np.shape[-1] in (7, 8), "模型输出维度应为 7 或 8"
    if act_np.shape[-1] == 7:
        grip_s = None
    else:
        grip_s = float(np.clip(act_np[7], 0.0, 1.0))
    joint_step = np.clip(act_np[:7], -1.0, 1.0) * MAX_JOINT_STEP
    return joint_step, grip_s

def grip_command_from_scalar(s):
    """把模型的抓手标量 [0,1] 映射到开/合命令"""
    if s is None:
        return None
    return "open" if s >= 0.5 else "close"


def main():
    # ========== 连接 Polymetis ==========
    robot = RobotInterface(ip_address=ARM_IP, port=ARM_PORT)
    gripper = GripperInterface(ip_address=GRIP_IP, port=GRIP_PORT)

    # 设置目标关节角度
    q_target = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0], dtype=np.float64)
    
    q_target = [
        0.0,
        -0.7854,
        -0.0,
        -2.356,
        -0.0,
        1.57,
        0.0,
    ]
    
    q_target =  [0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0]
    
    # 回到目标位置（可以在程序启动时先移动到目标）
    gripper.goto(GRIP_OPEN_WIDTH, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE, blocking=False)
    gripper.goto(GRIP_OPEN_WIDTH, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE, blocking=False)
    robot.move_to_joint_positions(q_target)

    gripper.goto(GRIP_OPEN_WIDTH, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE, blocking=False)

    

if __name__ == "__main__":
    main()

