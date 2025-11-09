#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DiffusionPolicy realtime inference service (ZeroMQ REP)

Protocol
--------
- Endpoint:  tcp://0.0.0.0:5555
- Request:   {"i": int, "joint": [7], "grip": float}
- Response:  {"ok": True, "did_infer": bool, "bucket_size": int, "latency_ms": float,
              "i": int, "joints_cmd": [7] or None, "grip_cmd": float or None}

I/O & Pre/Post-processing
-------------------------
- Camera: /dev/video0, take right-half frame, then downscale by 1/4
- Image tensor: (B, cams=1, C=3, H, W), BGR order, value range [0, 1]
- qpos normalization: Z-score using qpos_mean/std
- Action de-normalization: min-max mapping:
      ((x + 1) / 2) * (action_max - action_min) + action_min
- Model: DiffusionPolicy(cfg), weights loaded with model.deserialize(...) from .ckpt
- Aggregation: linearly weighted average of all candidates for the same time step
               (newer writes get larger weights)
- Visualization: non-blocking cv2.imshow with overlay; press 'q' or ESC to exit

Notes
-----
This refactor keeps the original behavior and algorithmic logic intact.
"""

from __future__ import annotations

import time
import threading
import pickle
from typing import Any, List, Optional, Tuple

import cv2
import zmq
import torch
import numpy as np

# ========================= Paths & Imports =========================
from policy import DiffusionPolicy  # provided by your project

# ====================== Checkpoint & Stats (unchanged) ======================
CKPT_PTH: str  = "/home/galbot/zyf/diffusion_policy_zyf/act-plus-plus/ckpt/cup_32_diffusion/policy_step_100000_seed_0.ckpt"
STATS_PTH: str = "/home/galbot/zyf/diffusion_policy_zyf/act-plus-plus/ckpt/cup_32_diffusion/dataset_stats.pkl"

# ====================== Policy configuration (as-is) =======================
POLICY_CONFIG: dict[str, Any] = {
    "lr": 2e-5,
    "camera_names": ["image_diagonal_view"],
    "action_dim": 8,                     # 7 joints + 1 gripper
    "observation_horizon": 1,
    "action_horizon": 8,
    "prediction_horizon": 32,            # keep aligned with your sim
    "num_queries": 32,
    "num_inference_timesteps": 20,
    "ema_power": 0.99,
    "vq": False,
}

# ============================= ZMQ server ==============================
BIND_ADDR: str = "tcp://0.0.0.0:5555"

# ============================= Device & Threads ==============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
try:
    cv2.setNumThreads(1)  # may fail on some builds; safe to ignore
except Exception:
    pass
torch.backends.cudnn.benchmark = True  # keep original setting

# ============================= Visualization ===============================
SHOW: bool = True
WIN_NAME: str = "DP Inference Input (Right half resized)"
_last_show_failed: bool = False


def _maybe_show(img_bgr: Optional[np.ndarray], overlay_lines: List[str]) -> None:
    """
    Non-blocking visualizer. On first imshow failure, auto-disables SHOW to avoid
    blocking the service (keeps behavior consistent with original code).
    """
    global SHOW, _last_show_failed
    if not SHOW or img_bgr is None:
        return
    try:
        disp = img_bgr.copy()
        y = 18
        for line in overlay_lines:
            cv2.putText(disp, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            y += 18
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WIN_NAME, disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:  # 'q' or ESC
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        raise
    except Exception as e:
        if not _last_show_failed:
            print(f"[viz] imshow failed: {e}. Disable SHOW to keep service running.")
            _last_show_failed = True
        SHOW = False


# ============================== Camera Reader ===============================
class CameraReader:
    """
    Asynchronous camera reader (V4L2).
    - Keeps the most recent frame in a thread-safe manner.
    - Same behavior as original (no extra logic).
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

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.dev, self.api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera device: {self.dev}")
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
                time.sleep(0.003)
                continue
            with self._lock:
                self.latest_frame = frame

    def get_latest(self, copy_frame: bool = True) -> Optional[np.ndarray]:
        with self._lock:
            f = self.latest_frame
        if f is None:
            return None
        return f.copy() if copy_frame else f

    def stop(self) -> None:
        self._stop = True
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        # (Keep original behavior: no join is required for correctness.)


# =========================== Model & Stats Loading ==========================
def build_and_load_policy(ckpt_path: str, cfg: dict, device: torch.device) -> DiffusionPolicy:
    """
    Build DiffusionPolicy, print parameter count, and load serialized weights
    (including EMA) via model.deserialize(...). Keep original behavior.
    """
    model = DiffusionPolicy(cfg)
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[server] DiffusionPolicy params: {n_params_m:.2f}M")
    state = torch.load(ckpt_path, map_location="cpu")
    model.deserialize(state)  # restore nets + EMA (original logic)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_stats(pkl_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset statistics for normalization/de-normalization:
      - qpos_mean/std (Z-score)
      - action_min/max (min-max mapping for actions)
    """
    with open(pkl_path, "rb") as f:
        stats = pickle.load(f)
    qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
    qpos_std  = np.asarray(stats["qpos_std"], dtype=np.float32)
    act_min   = np.asarray(stats["action_min"], dtype=np.float32)
    act_max   = np.asarray(stats["action_max"], dtype=np.float32)
    return qpos_mean, qpos_std, act_min, act_max


def _to_t(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)


# ============================== Pre/Post Ops ===============================
def preprocess(
    qpos8_prev: np.ndarray,
    frame_bgr: np.ndarray,
    qpos_mean: np.ndarray,
    qpos_std: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Build inputs for the model:
      - qpos: previous-step [7 joints + 1 gripper], Z-score -> tensor (1, 8)
      - image: take right half then 1/4 resize, produce tensor (1,1,3,H,W) in [0,1]
      - returns a copy (BGR) for visualization overlay
    """
    # Right half + downscale by 1/4 (keep BGR)
    H, W, _ = frame_bgr.shape
    right = frame_bgr[:, W // 2 :, :]
    H2, W2, _ = right.shape
    proc = cv2.resize(right, (W2 // 4, H2 // 4), interpolation=cv2.INTER_AREA)

    # qpos normalization (Z-score)
    qnorm = (qpos8_prev - qpos_mean) / (qpos_std + 1e-8)
    qpos_t = _to_t(qnorm[None, ...], device=device)  # (1, 8)

    # image tensor: (1,1,3,H,W), value range [0,1], BGR order
    img_t = _to_t(np.ascontiguousarray(proc), device=device)  # HWC
    img_t = (img_t.permute(2, 0, 1) / 255.0).unsqueeze(0).unsqueeze(0)
    return qpos_t, img_t, proc


def bucket_weighted_average(arr: np.ndarray, mode: str = "linear", gamma: float = 0.8) -> np.ndarray:
    """
    Weighted average for candidates in the same time step.
    - arr: [K, D], K ordered from oldest (0) to newest (K-1)
    - mode:
        "linear": weights = 1..K (newer gets higher weight)  <-- used by default
        "exp":    weights = gamma^(K-1..0), 0<gamma<1
    """
    K = arr.shape[0]
    if K == 1:
        return arr[0]
    if mode == "linear":
        w = np.arange(1, K + 1, dtype=np.float32)
    elif mode == "exp":
        w = np.power(gamma, np.arange(K - 1, -1, -1, dtype=np.int32)).astype(np.float32)
    else:
        raise ValueError(f"Unknown weighting mode: {mode}")
    w /= (w.sum() + 1e-8)
    return (arr * w[:, None]).sum(axis=0)


@torch.no_grad()
def infer_actions(model: DiffusionPolicy, qpos_t: torch.Tensor, img_t: torch.Tensor) -> np.ndarray:
    """
    Forward pass. Returns np.ndarray with shape [Tp, 8] in model output space
    (i.e., before min-max de-normalization). Keeps original shape handling.
    """
    out = model(qpos_t, img_t).float()  # (B, Tp, 8) or (B, 8)
    if out.ndim == 3:
        seq = out[0]                    # (Tp, 8)
    else:
        seq = out[0].unsqueeze(0)       # (1, 8)
    return seq.detach().cpu().numpy().astype(np.float32)


def unnormalize_minmax(x_norm: np.ndarray, act_min: np.ndarray, act_max: np.ndarray) -> np.ndarray:
    """
    Min-max de-normalization:
      ((x + 1) / 2) * (a_max - a_min) + a_min
    """
    return ((x_norm + 1.0) / 2.0) * (act_max - act_min) + act_min


# ================================ Main =================================
def main() -> None:
    # 1) Model & stats
    model = build_and_load_policy(CKPT_PTH, POLICY_CONFIG, device=device)
    qpos_mean, qpos_std, act_min, act_max = load_stats(STATS_PTH)

    # 2) Camera
    cam = CameraReader()
    cam.start()

    # 3) ZeroMQ REP
    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.setsockopt(zmq.RCVHWM, 10)
    rep.setsockopt(zmq.SNDHWM, 10)
    rep.bind(BIND_ADDR)
    print(f"[server] JSON REP on {BIND_ADDR}, device={device.type}")

    # 4) Buckets
    MAX_STEPS = 10000
    action_agg: List[List[np.ndarray]] = [[] for _ in range(MAX_STEPS + 512)]
    i_step: int = -1
    prev_joint_state: Optional[np.ndarray] = None
    prev_gripper_state: Optional[float] = None

    # Visualization state
    last_latency_ms: float = 0.0
    last_did_infer: bool = False
    last_bucket: int = 0
    last_proc: Optional[np.ndarray] = None

    try:
        while True:
            # ---- Receive request ----
            req = rep.recv_json()  # {"i": int, "joint": [7], "grip": float}
            t0 = time.perf_counter()

            i_step = int(req.get("i", i_step + 1))
            joint_state = np.asarray(req["joint"], dtype=np.float32)   # [7]
            grip_width  = float(req.get("grip", 0.0))

            frame = cam.get_latest(copy_frame=True)

            did_infer   = False
            bucket_size = 0
            joints_cmd: Optional[List[float]] = None
            grip_cmd: Optional[float] = None
            proc_disp: Optional[np.ndarray] = None

            if frame is not None:
                # Require previous state for conditioning
                if prev_joint_state is not None and prev_gripper_state is not None:
                    # Compose qpos[8] = prev joints(7) + prev gripper(1)
                    qprev = np.zeros(8, dtype=np.float32)
                    qprev[:7] = prev_joint_state
                    qprev[7]  = float(prev_gripper_state)

                    # Preprocess + inference (per-frame)
                    qpos_t, img_t, proc_disp = preprocess(qprev, frame, qpos_mean, qpos_std, device)
                    seq = infer_actions(model, qpos_t, img_t)  # (Tp, 8) or (1, 8)
                    did_infer = (seq is not None) and (seq.size > 0)

                    if did_infer:
                        # Write into time buckets (still in model output space)
                        if seq.ndim == 2 and seq.shape[0] > 1:
                            Tp = seq.shape[0]
                            for n in range(Tp):
                                idx = i_step + n
                                if 0 <= idx < len(action_agg):
                                    action_agg[idx].append(seq[n])
                        else:
                            if 0 <= i_step < len(action_agg):
                                action_agg[i_step].append(seq[0] if seq.ndim == 2 else seq)

                        bucket_size = len(action_agg[i_step]) if 0 <= i_step < len(action_agg) else 0

                        # Aggregate for current step (linear-weighted by default)
                        if 0 <= i_step < len(action_agg) and bucket_size > 0:
                            arr = np.asarray(action_agg[i_step], dtype=np.float32)  # [K, 8]
                            target_norm = bucket_weighted_average(arr, mode="linear")
                        else:
                            # Fallback: use the first frame of current seq
                            target_norm = seq[0] if seq.ndim == 2 else seq

                        # De-normalize to physical domain (min-max)
                        target = unnormalize_minmax(target_norm, act_min, act_max)  # [8]

                        if target is not None and target.shape[0] >= 8 and np.all(np.isfinite(target)):
                            joints_cmd = target[:7].astype(np.float32).tolist()
                            grip_cmd   = float(target[7])

                # Update previous state AFTER using it (one-frame delay)
                prev_joint_state   = joint_state.copy()
                prev_gripper_state = grip_width

            # Timing
            dt_ms = (time.perf_counter() - t0) * 1000.0

            # Update viz state
            last_latency_ms = float(dt_ms)
            last_did_infer  = bool(did_infer)
            last_bucket     = int(bucket_size)
            if proc_disp is not None:
                last_proc = proc_disp

            # ---- Reply ----
            resp: dict[str, Any] = {
                "ok": True,
                "did_infer": bool(did_infer),
                "bucket_size": int(bucket_size),
                "latency_ms": float(dt_ms),
                "i": int(i_step),
                "joints_cmd": joints_cmd,
                "grip_cmd": grip_cmd,
            }
            rep.send_json(resp)

            # ---- Visualization (non-blocking) ----
            if last_proc is not None:
                overlay = [
                    f"i={i_step}  did_infer={last_did_infer}  bucket={last_bucket}",
                    f"latency={last_latency_ms:.1f} ms  size={last_proc.shape[1]}x{last_proc.shape[0]}",
                    "press 'q' or Esc to quit",
                ]
                _maybe_show(last_proc, overlay)

    except KeyboardInterrupt:
        print("\n[server] Ctrl-C received. Stopping...")
    finally:
        # Cleanup
        try:
            cam.stop()
        except Exception:
            pass
        try:
            rep.close(0)
            ctx.term()
        except Exception:
            pass
        try:
            if SHOW:
                cv2.destroyAllWindows()
        except Exception:
            pass
        print("[server] shutdown.")


if __name__ == "__main__":
    main()
