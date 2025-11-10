#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, threading, re, argparse, glob, contextlib
from typing import List, Optional, Union
import numpy as np
import zmq, torch, cv2, h5py

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

with open(os.devnull, "w") as _devnull_out, open(os.devnull, "w") as _devnull_err, \
     contextlib.redirect_stdout(_devnull_out), contextlib.redirect_stderr(_devnull_err):
    from polymetis import RobotInterface, GripperInterface

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("Franka follower (gello / factr)")
    ap.add_argument("--mode", choices=["gello","factr"], required=True)

    ap.add_argument("--save_dir",   type=str, default="dataset/cup")
    ap.add_argument("--save_freq",  type=float, default=20.0)
    ap.add_argument("--save_frames",type=int, default=350)

    ap.add_argument("--no_camera", action="store_true")
    ap.add_argument("--cam_dev",    type=str, default="/dev/video1")
    ap.add_argument("--cam_fourcc", type=str, default="MJPG")
    ap.add_argument("--cam_w",      type=int, default=2560)
    ap.add_argument("--cam_h",      type=int, default=720)
    ap.add_argument("--cam_fps",    type=int, default=20)
    ap.add_argument("--crop_right_half", action="store_true", default=True)
    ap.add_argument("--downsample", type=int, default=4)

    ap.add_argument("--start_hold_s", type=float, default=0.5)
    ap.add_argument("--align_tol",    type=float, default=0.03)
    ap.add_argument("--force_start_pose", action="store_true", default=False)

    ap.add_argument("--no_stream_warn_s", type=float, default=0.5)

    ap.add_argument("--quiet", action="store_true", default=True)
    ap.add_argument("--verbose", action="store_true", default=False)

    return ap.parse_args()

ARGS = parse_args()
if ARGS.verbose:
    ARGS.quiet = False

# ---------------- Constants ----------------
FRANKA_JOINT_LIMITS = {
    "q_max": [ 2.8,  1.66,  2.8 , -0.17,  2.8 ,  3.65,  2.8 ],
    "q_min": [-2.8, -1.66, -2.8 , -2.97, -2.8 ,  0.08, -2.8 ],
}
FRANKA_JOINT_VEL = [2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51]
VEL_SAFETY_RATIO = 0.7
RATE_HZ   = 100.0
PERIOD    = 1.0 / RATE_HZ
EMA_ALPHA = 0.8
MAX_STEP_JOINT = [v * PERIOD * VEL_SAFETY_RATIO for v in FRANKA_JOINT_VEL]

GRIPPER_MIN = 0.0
GRIPPER_MAX = 0.08
GRIPPER_SPEED = 1.5
GRIPPER_FORCE = 30
GRIPPER_MIN_STEP = 1e-6
GRIPPER_KEEPALIVE = 0.25
GRIPPER_CMD_HZ = 60.0
GRIPPER_CMD_PERIOD = 1.0 / GRIPPER_CMD_HZ
W_GAIN = 1.0
SNAP_EPS = 0.001

START_GELLO = [0, 0, 0, -1.57, 0, 1.57, 0]
START_FACTR = [0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0]

# ---------------- Utils ----------------
def safe_list(x):
    if x is None: return None
    if hasattr(x, "tolist"): return x.tolist()
    return list(x)

def clip_q(q: List[float]) -> List[float]:
    return [max(min(v, qmax), qmin) for v, qmax, qmin in zip(q, FRANKA_JOINT_LIMITS["q_max"], FRANKA_JOINT_LIMITS["q_min"]) ]

def step_toward(curr: List[float], tgt: List[float], max_steps: List[float]) -> List[float]:
    return [c + np.clip(t - c, -lim, lim) for c, t, lim in zip(curr, tgt, max_steps)]

def is_close(q1, q2, tol=ARGS.align_tol) -> bool:
    return all(abs(a-b) <= tol for a, b in zip(q1, q2))

# ---------------- Camera ----------------
def _int_from_video_path(path: str) -> Optional[int]:
    if not isinstance(path, str): return None
    if path.startswith("/dev/video"):
        try:
            return int(path.replace("/dev/video",""))
        except Exception:
            return None
    return None

def _candidate_devices() -> List[Union[str,int]]:
    cands = []
    if ARGS.cam_dev is not None:
        cands.append(ARGS.cam_dev)
    cands += ["/dev/video0","/dev/video1","/dev/video2","/dev/video3"]
    cands += [0,1,2,3]
    for p in sorted(glob.glob("/dev/video*")):
        cands.append(p)
    seen=set(); out=[]
    for c in cands:
        if c in seen: continue
        seen.add(c); out.append(c)
    return out

class CameraReader:
    """V4L2 camera thread with robust open & graceful fallback."""
    def __init__(self,
                 dev: Union[str,int]=ARGS.cam_dev,
                 fourcc: str=ARGS.cam_fourcc,
                 size=(ARGS.cam_w, ARGS.cam_h),
                 fps: int=ARGS.cam_fps):
        self.dev, self.fourcc, self.size, self.fps = dev, fourcc, size, fps
        self.cap = None
        self._lock = threading.Lock()
        self._stop = False
        self.latest_frame = None
        self._t = None
        self.disabled = False

    def _try_open(self, dev, api):
        cap = cv2.VideoCapture(dev, api)
        if cap is None or not cap.isOpened():
            return None
        try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.size[0])
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        return cap

    def _open_robust(self):
        devs = [self.dev]
        idx = _int_from_video_path(self.dev) if isinstance(self.dev, str) else None
        if idx is not None:
            devs.append(idx)
        devs += _candidate_devices()

        tried = []
        for dv in devs:
            cap = self._try_open(dv, cv2.CAP_V4L2)
            tried.append((dv,"CAP_V4L2", cap is not None and cap.isOpened()))
            if cap is not None and cap.isOpened(): return cap
            cap = self._try_open(dv, cv2.CAP_ANY)
            tried.append((dv,"CAP_ANY", cap is not None and cap.isOpened()))
            if cap is not None and cap.isOpened(): return cap

        return None

    def start(self):
        if ARGS.no_camera:
            self.disabled = True
            return
        cap = self._open_robust()
        if cap is None:
            self.disabled = True
            return
        self.cap = cap
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005); continue
            with self._lock: self.latest_frame = frame

    def get_latest(self):
        if self.disabled: return None
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self._stop = True
        if self._t is not None: self._t.join(timeout=1.0)
        if self.cap is not None: self.cap.release()
        cv2.destroyAllWindows()

# ---------------- Gripper ----------------
class GripperWorker(threading.Thread):
    """Periodic gripper command thread (rate-limited + keepalive)."""
    def __init__(self, gripper: GripperInterface, getter):
        super().__init__(daemon=True)
        self.g = gripper
        self.get_w_in = getter
        self._stop = threading.Event()
        self._last_w = None
        self._last_t = 0.0

    def stop(self): self._stop.set()

    @staticmethod
    def _map_and_snap(w_in: float) -> float:
        if w_in is None or (isinstance(w_in, float) and np.isnan(w_in)):
            w_in = 0.0
        w_cmd = W_GAIN * float(w_in)
        w_cmd = float(min(max(w_cmd, GRIPPER_MIN), GRIPPER_MAX))
        if abs(w_cmd - GRIPPER_MIN) < SNAP_EPS: w_cmd = GRIPPER_MIN
        if abs(w_cmd - GRIPPER_MAX) < SNAP_EPS: w_cmd = GRIPPER_MAX
        return w_cmd

    def run(self):
        while not self._stop.is_set():
            now = time.time()
            if now - self._last_t < GRIPPER_CMD_PERIOD:
                time.sleep(GRIPPER_CMD_PERIOD - (now - self._last_t)); continue
            w_in = self.get_w_in()
            w_cmd = self._map_and_snap(w_in)
            need = (self._last_w is None) or (abs(w_cmd - self._last_w) > GRIPPER_MIN_STEP) or ((now - self._last_t) > GRIPPER_KEEPALIVE)
            if need:
                self.g.goto(w_cmd, GRIPPER_SPEED, GRIPPER_FORCE, False)
                self._last_w, self._last_t = w_cmd, time.time()
            else:
                time.sleep(0.001)

# ---------------- Recorder ----------------
class EpisodeRecorder:
    """Buffer state/action/image at save_freq; dump HDF5 when full()."""
    def __init__(self, save_dir: str, save_freq: float, save_frames: int, gripper: GripperInterface):
        self.dir = save_dir
        os.makedirs(self.dir, exist_ok=True)
        self.dt = 1.0 / float(save_freq)
        self.N  = int(save_frames)
        self.g = gripper
        self.reset_buffers()

    def reset_buffers(self):
        self.qpos_list: List[np.ndarray] = []
        self.action_list: List[np.ndarray] = []
        self.image_list: List[np.ndarray] = []
        self._init_t = -100.0
        self._count = 0
        self._started = False

    def maybe_push(self, robot: RobotInterface, target_q: List[float], w_cmd: float, frame: Optional[np.ndarray]):
        """Called by control loop when recording is enabled."""
        now = time.time()
        if (now - self._init_t) > (self.dt * self._count):
            if not self._started:
                self._init_t = now; self._started = True
            self._count += 1

            q_raw = safe_list(robot.get_joint_positions())
            qpos = np.zeros(8, dtype=np.float32)
            qpos[:7] = np.array(q_raw if q_raw is not None else target_q, dtype=np.float32)

            try:
                w_state = float(self.g.get_state().width)
            except Exception:
                w_state = w_cmd
            qpos[7] = float(min(max(w_state, GRIPPER_MIN), GRIPPER_MAX))
            self.qpos_list.append(qpos)

            act = np.zeros(8, dtype=np.float32)
            act[:7] = np.array(target_q, dtype=np.float32)
            act[7]  = float(w_cmd)
            self.action_list.append(act)

            if frame is not None:
                H, W, C = frame.shape
                if ARGS.crop_right_half:
                    frame = frame[:, W//2:, :]
                    H, W, C = frame.shape
                if ARGS.downsample and ARGS.downsample > 1:
                    frame = cv2.resize(frame, (W//ARGS.downsample, H//ARGS.downsample), interpolation=cv2.INTER_AREA)
                self.image_list.append(frame.copy())

    def full(self) -> bool:
        return (len(self.action_list) >= self.N) and (self.N > 0)

    def save_and_next(self) -> str:
        pat = re.compile(r"^episode_(\d+)\.hdf5$")
        used = {int(m.group(1)) for f in os.listdir(self.dir) for m in [pat.match(f)] if m}
        idx = 0
        while idx in used: idx += 1
        path = os.path.join(self.dir, f"episode_{idx}.hdf5")
        with h5py.File(path, 'w') as f:
            imgs = np.array(self.image_list) if len(self.image_list)>0 else np.zeros((0,), dtype=np.uint8)
            f.create_dataset('image_diagonal_view', data=imgs, compression='gzip', compression_opts=9)
            f.create_dataset('action', data=self.action_list, compression='gzip', compression_opts=9)
            f.create_dataset('qpos',   data=self.qpos_list, compression='gzip', compression_opts=9)
        os.chmod(path, 0o777)
        self.reset_buffers()
        return path

# ---------------- FACTR ZMQ (factr mode only) ----------------
class ZMQSubscriber:
    def __init__(self, ip_address="tcp://127.0.0.1:6001"):
        context = zmq.Context()
        self._sub_socket = context.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.CONFLATE, False)
        self._sub_socket.connect(ip_address)
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
        self._value = None
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    @property
    def message(self):
        return self._value

    def _loop(self):
        while True:
            try:
                msg = self._sub_socket.recv()
                if len(msg) % 8 == 0:
                    self._value = np.frombuffer(msg, dtype=np.float64)
                elif len(msg) % 4 == 0:
                    self._value = np.frombuffer(msg, dtype=np.float32).astype(np.float64)
                else:
                    self._value = None
            except Exception:
                self._value = None

class ZMQPublisher:
    def __init__(self, ip_address="tcp://*:6003"):
        context = zmq.Context()
        self._pub_socket = context.socket(zmq.PUB)
        self._pub_socket.bind(ip_address)

    def send(self, arr: np.ndarray):
        self._pub_socket.send(np.asarray(arr, dtype=np.float64).tobytes())

# ---------------- Controller ----------------
class Control:
    """Robot bringup, ZMQ I/O, smoothing, rate-limited control, and recording."""
    def __init__(self):
        ip = os.environ.get("POLYMETIS_SERVER_IP", "127.0.0.1")
        self.robot = RobotInterface(ip_address=ip, port=int(os.environ.get("POLYMETIS_SERVER_PORT", "50051")))
        self.grip  = GripperInterface(ip_address=ip, port=int(os.environ.get("POLYMETIS_GRIPPER_PORT", "50052")))
        if hasattr(self.grip, "home"): self.grip.home()
        self.grip.goto((GRIPPER_MIN + GRIPPER_MAX)/2.0, 0.1, 15, False)

        self.cam = CameraReader(); self.cam.start()

        # ZMQ
        ctx = zmq.Context.instance()
        if ARGS.mode == "gello":
            self.rep  = ctx.socket(zmq.REP);  self.rep.bind("tcp://*:6000")
            self.pull = ctx.socket(zmq.PULL); self.pull.bind("tcp://*:6001")
            self.pub  = ctx.socket(zmq.PUB);  self.pub.bind("tcp://*:6002")
            self.sub_q = None
            self.sub_g = None
            self.pub_tau = None
            self._connected = False
            print("[comm] waiting for communication ...")
            self._no_stream_announced = True
        else:
            self.rep = None; self.pull = None; self.pub = ctx.socket(zmq.PUB); self.pub.bind("tcp://*:6002")
            self.sub_q = ZMQSubscriber("tcp://127.0.0.1:6001")
            self.sub_g = ZMQSubscriber("tcp://127.0.0.1:6004")
            self.pub_tau = ZMQPublisher("tcp://*:6003")
            self._connected = False
            print("[comm] waiting for master stream ...")
            self._no_stream_announced = True

        self.recording_enabled = False

        self.target_q = (START_GELLO if ARGS.mode=="gello" else START_FACTR)[:]
        self.ema_q    = self.target_q[:]
        self.curr_q   = self.target_q[:]

        self._w_lock = threading.Lock()
        self._w_in = 0.0

        self.rec = EpisodeRecorder(ARGS.save_dir, ARGS.save_freq, ARGS.save_frames, self.grip)

        self._align_to_pose(self.target_q)

        self.robot.start_joint_impedance()
        q_pos = self.robot.get_joint_positions()
        if isinstance(q_pos, torch.Tensor):
            q_now = q_pos.clone().detach().to(torch.float32)
        else:
            q_now = torch.tensor(q_pos, dtype=torch.float32)
        self.robot.update_desired_joint_positions(q_now)
        time.sleep(0.05)

        self.gw = GripperWorker(self.grip, self.get_w_in); self.gw.start()

        if ARGS.mode == "gello":
            self.poller = zmq.Poller(); self.poller.register(self.pull, zmq.POLLIN)
            self._handshake_and_optional_align()
        else:
            pass

        self.last_cmd_t, self.last_pub_t = time.time(), 0.0

    def _align_to_pose(self, pose: List[float]):
        try:
            curr = safe_list(self.robot.get_joint_positions())
        except Exception:
            curr = None
        invalid = (curr is None) or (len(curr) != 7) or all(abs(x) < 1e-6 for x in curr)
        if invalid or (not is_close(curr, pose)):
            print(f"[align] moving to start pose: {pose}")
            tgt = pose if isinstance(pose, torch.Tensor) else torch.tensor(pose, dtype=torch.float32)
            self.robot.move_to_joint_positions(tgt)
            t0 = time.time(); timeout = 15.0
            while time.time() - t0 < timeout:
                try:
                    now = safe_list(self.robot.get_joint_positions())
                    if now and is_close(now, pose):
                        break
                except Exception:
                    pass
                time.sleep(0.02)
        else:
            print("[align] already at start pose")
        time.sleep(ARGS.start_hold_s)

    def _handshake_and_optional_align(self):
        msg = self.rep.recv_json()
        if not isinstance(msg, dict) or msg.get("type") != "init":
            self.rep.send_json({"status":"ERR","reason":"expect init"}); raise RuntimeError("expect init")
        from_msg = msg.get("virtual_init_arm") if "virtual_init_arm" in msg else msg.get("virtual_init")
        if (not ARGS.force_start_pose) and isinstance(from_msg, (list, tuple)) and len(from_msg) == 7:
            pose = clip_q([float(v) for v in from_msg])
            self._align_to_pose(pose)
            self.target_q = pose[:]; self.ema_q = pose[:]; self.curr_q = pose[:]
        else:
            self._align_to_pose(self.target_q)
        self.rep.send_json({"status":"READY"})
        self._connected = True
        self._enable_recording()

    def _enable_recording(self):
        if not self.recording_enabled:
            self.rec.reset_buffers()
            self.recording_enabled = True
            dur = ARGS.save_frames/ARGS.save_freq if ARGS.save_freq>0 else 0
            print("[comm] connected, recording enabled.")
            print(f"[rec] data format: qpos/action shape=(N, 8), image_diagonal_view shape=(<=N, H, W, 3), N={self.rec.N}, episode_duration_s≈{dur:.1f}")
            print("[rec] note: only full episodes (N frames) are saved; filenames increment as episode_0.hdf5, episode_1.hdf5, ...")

    def set_w_in(self, x: float):
        with self._w_lock:
            self._w_in = float(x)

    def get_w_in(self) -> float:
        with self._w_lock:
            return self._w_in

    def _ensure_joint_impedance(self):
        try:
            q_pos = self.robot.get_joint_positions()
            if isinstance(q_pos, torch.Tensor):
                q_now = q_pos.clone().detach().to(torch.float32)
            else:
                q_now = torch.tensor(q_pos, dtype=torch.float32)
            self.robot.update_desired_joint_positions(q_now)
        except Exception:
            self.robot.start_joint_impedance()
            q_pos = self.robot.get_joint_positions()
            if isinstance(q_pos, torch.Tensor):
                q_now = q_pos.clone().detach().to(torch.float32)
            else:
                q_now = torch.tensor(q_pos, dtype=torch.float32)
            self.robot.update_desired_joint_positions(q_now)
            time.sleep(0.02)

    def _maybe_set_connected_factr(self):
        if self._connected:
            return
        qmsg = self.sub_q.message
        gmsg = self.sub_g.message
        if qmsg is None and gmsg is None:
            return
        self._connected = True
        self._enable_recording()

    def loop_once(self) -> bool:
        t0 = time.time()

        # --- receive commands ---
        latest = None
        if ARGS.mode == "gello":
            socks = dict(self.poller.poll(timeout=0))
            if self.pull in socks and socks[self.pull] == zmq.POLLIN:
                while True:
                    try:
                        payload = self.pull.recv_json(flags=zmq.NOBLOCK)
                        latest = payload
                    except zmq.Again:
                        break
            if latest and isinstance(latest, dict) and latest.get("type") == "cmd":
                arm = latest.get("arm") or latest.get("q")
                w   = latest.get("gripper") or latest.get("w")
                if isinstance(arm, (list, tuple)) and len(arm) == 7:
                    self.target_q = clip_q([float(v) for v in arm])
                if w is not None:
                    try: self.set_w_in(float(w))
                    except Exception: pass
                self.last_cmd_t = t0
                if not self._connected:
                    self._connected = True
                    self._enable_recording()
        else:
            self._maybe_set_connected_factr()
            qmsg = self.sub_q.message
            if qmsg is not None and len(qmsg) >= 7:
                self.target_q = clip_q([float(v) for v in qmsg[:7]])
                self.last_cmd_t = t0
            gmsg = self.sub_g.message
            if gmsg is not None and len(gmsg) >= 1:
                try: self.set_w_in(float(gmsg[0]))
                except Exception: pass
                self.last_cmd_t = t0

        # --- smoothing & rate limiting & send to controller ---
        self.ema_q  = [(1-EMA_ALPHA)*e + EMA_ALPHA*t for e, t in zip(self.ema_q, self.target_q)]
        next_q      = step_toward(self.curr_q, self.ema_q, MAX_STEP_JOINT)
        try:
            q_pos = torch.tensor(next_q, dtype=torch.float32)
            self.robot.update_desired_joint_positions(q_pos)
        except Exception as e:
            if "no controller running" in str(e).lower():
                self._ensure_joint_impedance()
                self.robot.update_desired_joint_positions(torch.tensor(next_q, dtype=torch.float32))
            else:
                raise
        self.curr_q = next_q

        # --- camera & recording ---
        frame = self.cam.get_latest()
        w_cmd_for_action = GripperWorker._map_and_snap(self.get_w_in())
        if self.recording_enabled:
            self.rec.maybe_push(self.robot, self.target_q, w_cmd_for_action, frame)
            if self.rec.full():
                path = self.rec.save_and_next()
                print(f"[comm] episode saved: {path}")
                # 采集完成后，直接结束主循环，不再张开夹爪、sleep、或回零
                return False

        # --- publish state (6002) ---
        if ARGS.mode == "gello" and (t0 - self.last_pub_t) > 0.2:
            self.last_pub_t = t0
            state = {
                "type": "state",
                "ts": t0,
                "q":  safe_list(self.robot.get_joint_positions()),
                "dq": safe_list(self.robot.get_joint_velocities()),
                "w_cmd": float(w_cmd_for_action),
            }
            self.pub.send_json(state)

        # --- publish external torques (factr only, 6003) ---
        if ARGS.mode == "factr" and self.pub_tau is not None:
            try:
                tau = np.array(self.robot.get_robot_state().motor_torques_external, dtype=np.float64)
                self.pub_tau.send(tau)
            except Exception:
                pass

        dt = time.time() - t0
        if dt < PERIOD: time.sleep(PERIOD - dt)
        return True

    def run(self):
        try:
            while self.loop_once():
                pass
        finally:
            # 结束时不再自动回到起始位姿，只做资源清理
            try: self.cam.stop()
            except: pass
            try: self.gw.stop()
            except: pass

class App:
    def run(self):
        ctrl = Control()
        ctrl.run()

if __name__ == "__main__":
    App().run()

