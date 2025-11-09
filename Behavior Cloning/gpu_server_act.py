#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ACT inference service (ZeroMQ REP)

- Robot: Franka (example: 7 joints + 1 gripper = 8D state/action)
- Camera: V4L2, use the right half and downsample to 1/4, feed a tensor
          with shape (B, cams=1, C=3, H, W) and value range [0, 1], BGR order
- Bucket aggregation: linearly weighted average of multiple predictions
                      for the same time step to reduce jitter
- Input  (ZMQ REP, JSON):  {"i": int, "joint": [7], "grip": float}
- Output (ZMQ REP, JSON):  {"ok": True, "did_infer": bool, "bucket_size": int,
                            "latency_ms": float, "i": int,
                            "joints_cmd": [7] or None, "grip_cmd": float or None}
Notes:
- This refactor keeps the original logic intact:
  * action de-normalization uses mean/std (not min/max)
  * previous-timestep state for conditioning
  * linear weights 1..K for aggregation (newer predictions weigh more)
  * 200 Hz control pacing with sleep
  * robust JSON validation and graceful error responses
"""

from __future__ import annotations

import time
import threading
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import cv2
import zmq
import sys
import os


# ======= Import your ACT policy (same as original) =======
from policy import ACTPolicy


# ======================= Runtime configuration =======================
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CONTROL_HZ: int = 200
DT: float = 1.0 / CONTROL_HZ
DEBUG: bool = False

# ZeroMQ bind address (client connects to this endpoint)
BIND_ADDR: str = "tcp://0.0.0.0:5555"

# Model/statistics (keep identical to your original layout)
CKPT_PTH: str = "ckpt/cup_32_act/policy_step_10000_seed_0.ckpt"
STATS_PTH: str = "ckpt/cup_32_act/dataset_stats.pkl"


# ============================ Camera reader ============================
class CameraReader:
    """
    Asynchronous V4L2 camera reader.
    - Opens the device with given properties.
    - Keeps the latest frame and timestamp thread-safely.
    - Provides a simple start/stop lifecycle.
    """

    def __init__(
        self,
        dev: str = "/dev/video0",
        api: int = cv2.CAP_V4L2,
        fourcc: str = "MJPG",
        size: Tuple[int, int] = (2560, 720),
        fps: int = 30,
    ) -> None:
        self.dev, self.api = dev, api
        self.fourcc, self.size, self.fps = fourcc, size, fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._stop = False
        self._t: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_ts: float = 0.0

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.dev, self.api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device: {self.dev}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self.latest_frame = frame
                self.latest_ts = time.time()

    def get_latest(self, copy_frame: bool = True) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            f = self.latest_frame
            ts = self.latest_ts
        if f is None:
            return None, 0.0
        return (f.copy() if copy_frame else f), ts

    def stop(self) -> None:
        self._stop = True
        if self._t is not None:
            self._t.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# =========================== Model utilities ===========================
def build_and_load_policy(ckpt_path: str, device: str) -> ACTPolicy:
    """
    Build ACT policy with the same hyperparameters you used for training,
    then load weights via deserialize (preferred) or state_dict as fallback.
    """
    policy_config = {
        "lr": 2e-5,               # --lr 2e-5
        "num_queries": 32,        # --chunk_size 32
        "kl_weight": 10,          # --kl_weight 10
        "hidden_dim": 512,        # --hidden_dim 512
        "dim_feedforward": 3200,  # --dim_feedforward 3200
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        # Keep the camera convention identical to training/evaluation:
        "camera_names": ["image_diagonal_view"],
        "action_dim": 8,  # 7 joints + 1 gripper
    }

    model = ACTPolicy(policy_config)

    ckpt_obj = torch.load(ckpt_path, map_location=torch.device(device))
    if hasattr(model, "deserialize"):
        model.deserialize(ckpt_obj)  # training-style serialized checkpoint
    else:
        model.load_state_dict(ckpt_obj)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model


def _linear_weighted_average(arr: np.ndarray) -> np.ndarray:
    """
    Linearly weighted average along axis=0.
    Weights are 1..K (newer predictions get higher weights).
    """
    # arr: [K, D]
    k = arr.shape[0]
    if k == 1:
        return arr[0]
    w = np.arange(1, k + 1, dtype=np.float32)
    return np.average(arr, axis=0, weights=w)


# ============================== Main loop ==============================
def main() -> None:
    # ----- Load model and normalization stats -----
    model = build_and_load_policy(CKPT_PTH, device=DEVICE)

    with open(STATS_PTH, "rb") as f:
        stats = pickle.load(f)

    # qpos normalization (Z-score)
    qpos_mean: np.ndarray = stats["qpos_mean"].astype(np.float32)  # [8]
    qpos_std: np.ndarray = stats["qpos_std"].astype(np.float32)    # [8]

    # action de-normalization (Gaussian: mean/std)
    act_mean: np.ndarray = stats["action_mean"].astype(np.float32)  # [8]
    act_std: np.ndarray = stats["action_std"].astype(np.float32)    # [8]

    # ----- Camera -----
    cam = CameraReader()
    cam.start()

    # ----- ZeroMQ REP server -----
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)  # no lingering on close
    sock.bind(BIND_ADDR)
    print(f"[server] ACT inference server on {BIND_ADDR}, device={DEVICE}")

    # ----- Bucket aggregation -----
    MAX_STEPS = 2500
    action_buckets: List[List[np.ndarray]] = [[] for _ in range(MAX_STEPS + 100)]
    i_step: int = -1
    prev_joint_state: Optional[np.ndarray] = None
    prev_gripper_state: Optional[float] = None

    try:
        while True:
            # ---- Receive and validate request ----
            t0 = time.perf_counter()
            try:
                req = sock.recv_json()
            except Exception as e:
                sock.send_json({"ok": False, "error": f"bad json: {e}"})
                continue

            if "joint" not in req or "grip" not in req:
                sock.send_json({"ok": False, "error": "missing keys: joint/grip"})
                continue
            if not isinstance(req["joint"], (list, tuple)) or len(req["joint"]) != 7:
                sock.send_json({"ok": False, "error": "joint must be a list of length 7"})
                continue

            i_step = int(req.get("i", i_step + 1))
            joint_state = np.asarray(req["joint"], dtype=np.float32)  # [7]
            grip_width = float(req.get("grip", 0.0))

            # ---- Acquire latest frame ----
            frame, _ts = cam.get_latest(copy_frame=True)

            did_infer = False
            bucket_size = 0
            joints_cmd: Optional[List[float]] = None
            grip_cmd: Optional[float] = None

            if frame is not None:
                # Preprocess image: use right half and downsample by 4 (keep BGR)
                H, W, _ = frame.shape
                right_half = frame[:, W // 2 :, :]
                H2, W2, _ = right_half.shape
                proc = cv2.resize(
                    right_half, (W2 // 4, H2 // 4), interpolation=cv2.INTER_AREA
                )  # e.g., (180, 320) for original 720p

                # Require a previous state for conditioning (same logic)
                if prev_joint_state is not None and prev_gripper_state is not None:
                    # Build previous qpos[8] = 7 joints + 1 gripper
                    qpos_prev = np.zeros(8, dtype=np.float32)
                    qpos_prev[:7] = prev_joint_state
                    qpos_prev[7] = float(prev_gripper_state)

                    # Z-score normalization for qpos
                    qpos_norm = (qpos_prev - qpos_mean) / (qpos_std + 1e-8)
                    qpos_t = (
                        torch.from_numpy(qpos_norm)
                        .to(torch.float32)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )  # (1, 8)

                    # Image tensor: (B, cams=1, C=3, H, W), value in [0, 1], BGR order
                    img_t = (
                        torch.from_numpy(proc)
                        .to(torch.float32)
                        .permute(2, 0, 1)
                        / 255.0
                    )  # (3, H, W)
                    img_t = img_t.unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,3,H,W)

                    # Forward pass (keep return-shape compatibility)
                    with torch.no_grad():
                        out = model(qpos_t, img_t)
                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        out_np = out.float().detach().cpu().numpy()
                        action_list = out_np[0]  # strip batch dim

                        # Fallback "first" action in normalized space
                        first = action_list[0] if action_list.ndim == 2 else action_list

                    did_infer = True

                    # Write predictions into future buckets (normalized space)
                    if action_list.ndim == 2:
                        Tpred = action_list.shape[0]
                        for n in range(Tpred):
                            idx = i_step + n
                            if 0 <= idx < len(action_buckets):
                                action_buckets[idx].append(action_list[n])
                    else:
                        if 0 <= i_step < len(action_buckets):
                            action_buckets[i_step].append(first)

                    # Aggregate or fallback (normalized space)
                    bucket_size = len(action_buckets[i_step]) if 0 <= i_step < len(action_buckets) else 0
                    if 0 <= i_step < len(action_buckets) and bucket_size > 0:
                        arr = np.asarray(action_buckets[i_step], dtype=np.float32)  # [K, 8]
                        target_qpos_norm = _linear_weighted_average(arr)
                    else:
                        target_qpos_norm = first

                    # De-normalize to physical action space (Gaussian)
                    target_qpos = target_qpos_norm * act_std + act_mean
                    joints_cmd = target_qpos[:7].astype(np.float32).tolist()
                    grip_cmd = float(target_qpos[7])

                    if DEBUG:
                        print(
                            f"[dbg] i={i_step} bucket={bucket_size} grip_cmd={grip_cmd:.4f}"
                        )

                # Update previous state after using it (single-frame delay)
                prev_joint_state = joint_state.copy()
                prev_gripper_state = grip_width

            # ---- Respond to client ----
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resp: dict[str, Any] = {
                "ok": True,
                "did_infer": did_infer,
                "bucket_size": int(bucket_size),
                "latency_ms": float(latency_ms),
                "i": int(i_step),
                "joints_cmd": joints_cmd if joints_cmd is not None else None,
                "grip_cmd": grip_cmd if grip_cmd is not None else None,
            }
            sock.send_json(resp)

            # ---- Pace the loop to ~CONTROL_HZ ----
            loop_elapsed = time.perf_counter() - t0
            if loop_elapsed < DT:
                time.sleep(DT - loop_elapsed)

    except KeyboardInterrupt:
        print("\n[server] Ctrl-C received. Stopping...")
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            sock.close(0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
