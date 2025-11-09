#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==== path setup ====
import sys, os
sys.path.append('src/factr_teleop')
sys.path.append('src/factr_teleop/factr_teleop/urdf')
sys.path.append('src/factr_teleop/factr_teleop/dynamixel/python/src')
"""
Unified master for teleop:
- --mode gello  : Dynamixel -> JSON (PUSH/REQ) stream, no haptics
- --mode factr  : Dynamixel -> high-fidelity control + haptics (PUB/SUB)
"""

import time, argparse, glob, subprocess
import numpy as np
import zmq

# ===================== CLI =====================
def parse_args():
    ap = argparse.ArgumentParser(description="Unified Master (gello / FACTR)")

    ap.add_argument("--mode", choices=["gello","factr"], required=True)

    # per-mode ports
    ap.add_argument("--port_id_gello",
                    default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTALIGKF-if00-port0")
    ap.add_argument("--port_id_factr",
                    default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAA08RV-if00-port0")
    ap.add_argument("--port_id", default=None)

    # common
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--print_every_s", type=float, default=1.0)

    # gello
    ap.add_argument("--client_rate_hz", type=float, default=50.0)
    ap.add_argument("--use_sim", action="store_true")
    ap.add_argument("--grip_deg_min", type=float, default=156.984375)
    ap.add_argument("--grip_deg_max", type=float, default=198.784375)
    ap.add_argument("--sim_urdf", default=None)

    # alignment (used by both modes; FACTR defaults to L2 with larger tol)
    ap.add_argument("--align_tol", type=float, default=0.2)
    ap.add_argument("--align_metric", choices=["max","l2"], default="max")
    ap.add_argument("--align_hold_s", type=float, default=0.20)
    ap.add_argument("--align_debug", action="store_true")

    # FACTR control and model
    ap.add_argument("--ctrl_hz", type=float, default=500.0)
    ap.add_argument("--urdf",
                    default="/home/galbot/zyf/galbot/Tele/src/factr_teleop/factr_teleop/urdf/factr_teleop_franka.urdf")
    ap.add_argument("--gravity_gain", type=float, default=0.8)
    ap.add_argument("--null_kp", type=float, default=0.1)
    ap.add_argument("--null_kd", type=float, default=0.01)
    ap.add_argument("--stiction_speed", type=float, default=0.9)
    ap.add_argument("--stiction_gain",  type=float, default=0.6)
    ap.add_argument("--limit_kp", type=float, default=1.03561658022)
    ap.add_argument("--limit_kd", type=float, default=0.04315069084)
    ap.add_argument("--torque_gain", type=float, default=3.0)
    ap.add_argument("--torque_damping", type=float, default=0.0)
    ap.add_argument("--warmup_s", type=float, default=0.80)

    ap.add_argument("--factr_py_path", default=None)
    return ap.parse_args()

ARGS = parse_args()

# ================= USB helpers =================
def _list_byid():
    try:
        base = "/dev/serial/by-id"
        if not os.path.isdir(base):
            return []
        return [os.path.join(base, x) for x in sorted(os.listdir(base))]
    except Exception:
        return []

def resolve_port(for_mode: str):
    if ARGS.port_id is not None:
        src = ARGS.port_id
    else:
        src = ARGS.port_id_gello if for_mode == "gello" else ARGS.port_id_factr

    if src.startswith("/dev/ttyUSB"):
        if not os.path.exists(src):
            raise RuntimeError(f"tty device not found: {src}")
        return src, src

    if src.startswith("/dev/serial/by-id/"):
        if not os.path.exists(src):
            listing = _list_byid()
            msg = f"not found: {src}\nAvailable by-id:\n  " + "\n  ".join(listing or ["<none>"])
            raise RuntimeError(msg)
        link = os.readlink(src)
        tty = "/dev/" + os.path.basename(link)
        if not os.path.exists(tty): tty = None
        return src, tty

    byid = "/dev/serial/by-id/" + src
    if os.path.exists(byid):
        link = os.readlink(byid)
        tty = "/dev/" + os.path.basename(link)
        if not os.path.exists(tty): tty = None
        return byid, tty

    listing = _list_byid()
    msg = f"invalid port identifier: {src}\nAvailable by-id:\n  " + "\n  ".join(listing or ["<none>"])
    raise RuntimeError(msg)

def check_latency(ttyusb_dev: str):
    if not ttyusb_dev: return
    tty = os.path.basename(ttyusb_dev)
    p = f"/sys/bus/usb-serial/devices/{tty}/latency_timer"
    try:
        val = int(subprocess.check_output(["bash","-lc", f"cat {p}"]).decode().strip())
        if val != 1:
            print(f"[WARN] {p}={val}, suggest: echo 1 | sudo tee {p}")
    except Exception:
        pass

# ================= PATH/PACKAGE HELPERS (FACTR) =================
def force_patch_paths_for_factr():
    candidates = []
    if ARGS.factr_py_path: candidates.append(ARGS.factr_py_path)
    env = os.environ.get("FACTR_PY_PATH")
    if env: candidates.append(env)

    # common locations + project src
    candidates += [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "src", "factr_teleop"),
    ]
    # dynamixel/python/src
    more = []
    for base in list(candidates):
        more.append(os.path.join(base, "factr_teleop", "dynamixel", "python", "src"))
    candidates += more

    # update sys.path
    added = []
    for pth in candidates:
        if os.path.isdir(pth) and (pth not in sys.path):
            sys.path.append(pth); added.append(pth)

    if added:
        print("[FACTR] sys.path patched (forced):")
        for a in added: print("   +", a)
    else:
        print("[FACTR] WARNING: no path added; will still try fallbacks.")

def import_driver(mode: str):
    if mode == "gello":
        try:
            from gello.dynamixel.driver import DynamixelDriver
            return DynamixelDriver
        except Exception as e:
            print("[WARN] gello driver import failed:", e)
            # fallback to FACTR driver
            force_patch_paths_for_factr()
            from factr_teleop.dynamixel.driver import DynamixelDriver
            print("[WARN] fallback to FACTR DynamixelDriver for gello")
            return DynamixelDriver
    else:
        try:
            from factr_teleop.dynamixel.driver import DynamixelDriver
            return DynamixelDriver
        except Exception as e:
            print("[WARN] Preferred 'factr_teleop' import failed:", e)
            force_patch_paths_for_factr()
            from factr_teleop.dynamixel.driver import DynamixelDriver
            return DynamixelDriver

# ================= Common configs =================
NUM_ARM = 7
JOINT_IDS = tuple(range(1,9))  # 7 arm + 1 gripper

START_GELLO = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]
START_FACTR = [0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0]

# gello mapping (fixed)
GELLO_SIGNS   = (1, -1, 1, -1, 1, -1, 1)
GELLO_OFFSETS = (np.pi, np.pi, 2*np.pi, np.pi, np.pi, np.pi, 1.5*np.pi)

FRANKA_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
FRANKA_Q_MIN = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])

FACTR_SIGNS = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
ARM_LIMIT_MAX = np.array([2.8973, 1.7628, 2.8973, -0.8698, 2.8973, 3.7525, 2.8973]) - 0.1
ARM_LIMIT_MIN = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]) + 0.1
GRIP_MIN, GRIP_MAX = 0.0, 0.4

def build_ctx():
    return zmq.Context.instance()

# ========== alignment params (FACTR defaults: L2 + larger tol) ==========
def _resolved_align_params():
    metric = ARGS.align_metric
    tol = ARGS.align_tol
    if ARGS.mode == "factr":
        if "--align_metric" not in sys.argv:
            metric = "l2"
        if "--align_tol" not in sys.argv and metric == "l2":
            tol = 1.2
    return metric, tol

# ================= gello master =================
def _gello_wait_manual_align(drv, tol=0.4, metric="max", hold_s=0.20, debug=False):
    """
    Manual alignment for gello.
    """
    print(f"[gello] alignment waiting... tol={tol:.3f} rad, metric={metric}, hold={hold_s:.2f}s")
    ok_since = None
    last_print = 0.0
    while True:
        raw = np.array(drv.get_joints()[:NUM_ARM], dtype=float)
        q   = np.array([GELLO_SIGNS[i] * (raw[i] - GELLO_OFFSETS[i]) for i in range(NUM_ARM)], dtype=float)
        target = np.array(START_GELLO, dtype=float)
        err_vec = np.abs(q - target)
        if metric == "max":
            err_val = float(np.max(err_vec))
        else:
            err_val = float(np.linalg.norm(q - target))
        now = time.time()

        within = (err_val <= tol)
        if within:
            if ok_since is None:
                ok_since = now
            if (now - ok_since) >= hold_s:
                print(f"[gello] alignment OK (err={err_val:.3f} rad, metric={metric})")
                return
        else:
            ok_since = None

        if (now - last_print) >= 0.5:
            if debug:
                print(f"[gello] aligning: err={err_val:.3f} (tol={tol}) | per-joint={np.round(err_vec,3).tolist()}")
            else:
                print(f"[gello] please move master arm to {START_GELLO} (error {err_val:.3f} rad, tol={tol}) ...")
            last_print = now

        time.sleep(0.05)

def run_gello_master():
    try:
        import pybullet as p, pybullet_data
    except Exception:
        p = None; pybullet_data = None

    PERIOD = 1.0 / ARGS.client_rate_hz

    driver_port, tty_for_latency = resolve_port("gello")
    check_latency(tty_for_latency)
    print(f"[gello] using port: {driver_port} (latency on {tty_for_latency or 'n/a'})")

    DynamixelDriver = import_driver("gello")
    # retry several times for robustness
    for i in range(1,4):
        try:
            print(f"Attempting to initialize Dynamixel driver (attempt {i}/3)")
            drv = DynamixelDriver(list(JOINT_IDS), port=driver_port, baudrate=ARGS.baud)
            for _ in range(5): drv.get_joints()
            print(f"Successfully initialized Dynamixel driver on {driver_port}")
            break
        except Exception:
            if i==3: raise
            time.sleep(0.3)

    # strict manual alignment (gello)
    _gello_wait_manual_align(drv, tol=ARGS.align_tol, metric=ARGS.align_metric,
                             hold_s=ARGS.align_hold_s, debug=ARGS.align_debug)

    # sim (optional)
    sim_pack = None
    if ARGS.use_sim and p is not None and ARGS.sim_urdf:
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0,0,-9.8)
        rid = p.loadURDF(ARGS.sim_urdf, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        name2id = {p.getJointInfo(rid,i)[1].decode():i for i in range(p.getNumJoints(rid))}
        arm_names = [f"fr3_joint{i}" for i in range(1,8)]
        arm_idx = [name2id[n] for n in arm_names if n in name2id]
        fing_idx = [name2id.get("fr3_finger_joint1"), name2id.get("fr3_finger_joint2")]
        sim_pack = (p, rid, arm_idx, fing_idx)

    # ZMQ
    ctx = build_ctx()
    req  = ctx.socket(zmq.REQ);  req.connect("tcp://127.0.0.1:6000")
    push = ctx.socket(zmq.PUSH); push.connect("tcp://127.0.0.1:6001")

    hello = {"type":"init","virtual_init_arm":START_GELLO,"virtual_init_grip":0.02}
    req.send_json(hello); rep = req.recv_json()
    if rep.get("status") != "READY":
        print("[gello] server not ready:", rep); return
    print("[gello] READY. streaming...")

    sent = 0; t_last = time.time()
    try:
        while True:
            ang = drv.get_joints()
            arm = [ GELLO_SIGNS[i] * (ang[i] - GELLO_OFFSETS[i]) for i in range(NUM_ARM) ]
            arm = np.clip(arm, FRANKA_Q_MIN, FRANKA_Q_MAX).tolist()
            g_raw = ang[7]
            g_m = float(np.interp(g_raw,
                                  (np.deg2rad(ARGS.grip_deg_min), np.deg2rad(ARGS.grip_deg_max)),
                                  (0.0, 0.08)))
            push.send_json({"type":"cmd","arm":arm,"gripper":g_m}); sent += 1

            if sim_pack:
                p, rid, arm_idx, fing_idx = sim_pack
                if arm_idx: p.setJointMotorControlArray(rid, arm_idx, p.POSITION_CONTROL, targetPositions=arm)
                for fj in fing_idx:
                    if fj is not None: p.setJointMotorControl2(rid, fj, p.POSITION_CONTROL, targetPosition=g_m)
                p.stepSimulation()

            now = time.time()
            if now - t_last >= ARGS.print_every_s:
                print(f"[gello] tx ~{sent/(now-t_last):.1f} Hz | w={g_m:.3f} | q0={arm[0]:.3f}")
                t_last, sent = now, 0
            time.sleep(PERIOD)
    except KeyboardInterrupt:
        print("[gello] bye.")

# ================= FACTR master =================
def _resolve_urdf_path(p: str) -> str:
    p = os.path.expanduser(os.path.expandvars(p))
    if not os.path.isabs(p):
        base = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.abspath(os.path.join(base, p))
        if os.path.exists(cand):
            return cand
    return p

def find_urdf(initial_path: str) -> str:
    """If the given path does not exist, search nearby for a URDF containing 'franka' in the name."""
    search_roots = list(dict.fromkeys(
        [os.path.dirname(initial_path)] +
        sys.path +
        [
            os.getcwd(),
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "src", "factr_teleop", "factr_teleop", "urdf"),
        ]
    ))
    candidates = []
    for root in search_roots:
        if not root or not os.path.isdir(root): continue
        for pat in [
            os.path.join(root, "**", "*franka*.urdf"),
            os.path.join(root, "**", "factr_teleop_franka.urdf"),
            os.path.join(root, "*.urdf"),
        ]:
            candidates += [p for p in glob.glob(pat, recursive=True) if os.path.isfile(p)]
    for c in candidates:
        name = os.path.basename(c).lower()
        if "franka" in name:
            print("[FACTR] URDF auto-discovered:", c)
            return c
    return initial_path

def recv_tau_nonblock(sub, num, last):
    """Non-blocking receive of a tau frame; choose dtype by buffer size; return last on failure."""
    try:
        buf = sub.recv(flags=zmq.NOBLOCK)
    except zmq.Again:
        return last
    n = len(buf)
    if n <= 0:
        return last

    if (n % 8) == 0:
        try:
            arr64 = np.frombuffer(buf, dtype=np.float64, count=n // 8)
            if arr64.size >= num:
                return arr64[:num].astype(np.float32)
        except Exception:
            pass
    if (n % 4) == 0:
        try:
            arr32 = np.frombuffer(buf, dtype=np.float32, count=n // 4)
            if arr32.size >= num:
                return arr32[:num]
        except Exception:
            pass
    return last

def _factr_wait_manual_align(drv, offsets, tol=1.5, metric="l2", hold_s=0.25, debug=False):
    """
    Manual alignment for FACTR, mapping into Franka joint space.
    """
    print(f"[FACTR] alignment waiting... tol={tol:.3f} ({metric}), hold={hold_s:.2f}s")
    ok_since = None
    last_print = 0.0
    while True:
        try:
            q_raw, _ = drv.get_positions_and_velocities()
        except Exception:
            time.sleep(0.05); continue
        signs = FACTR_SIGNS
        q_map = (np.array(q_raw[:NUM_ARM]) - offsets[:NUM_ARM]) * signs[:NUM_ARM]
        q_map = np.clip(q_map, FRANKA_Q_MIN, FRANKA_Q_MAX)
        target = np.array(START_FACTR, dtype=float)

        if metric == "max":
            err_vec = np.abs(q_map - target)
            err_val = float(np.max(err_vec))
        else:
            err_val = float(np.linalg.norm(q_map - target))
            err_vec = np.abs(q_map - target)

        now = time.time()
        within = (err_val <= tol)
        if within:
            if ok_since is None: ok_since = now
            if (now - ok_since) >= hold_s:
                print(f"[FACTR] alignment OK (err={err_val:.3f}, {metric})")
                return
        else:
            ok_since = None

        if (now - last_print) >= 0.5:
            if debug:
                print(f"[FACTR] aligning: err={err_val:.3f} (tol={tol}) | per-joint={np.round(err_vec,3).tolist()}")
            else:
                print(f"[FACTR] please move master arm to {START_FACTR} (error {err_val:.3f}) ...")
            last_print = now
        time.sleep(0.05)

def run_factr_master():
    import pinocchio as pin

    CTRL_HZ = ARGS.ctrl_hz
    DT = 1.0 / CTRL_HZ

    driver_port, tty_for_latency = resolve_port("factr")
    check_latency(tty_for_latency)
    print(f"[FACTR] using port: {driver_port} (latency on {tty_for_latency or 'n/a'})")

    # ---- Dynamixel driver ----
    DynamixelDriver = import_driver("factr")
    SERVO_TYPES = [
        "XC330_T288_T", "XM430_W210_T", "XC330_T288_T", "XM430_W210_T",
        "XC330_T288_T", "XC330_T288_T", "XC330_T288_T", "XC330_T288_T",
    ]
    try:
        drv = DynamixelDriver(list(JOINT_IDS), SERVO_TYPES, driver_port)
    except TypeError:
        drv = DynamixelDriver(list(JOINT_IDS), port=driver_port, baudrate=ARGS.baud)

    # torque mode
    for fn, args in [("set_torque_mode",(False,)), ("set_operating_mode",(0,)), ("set_torque_mode",(True,))]:
        try: getattr(drv, fn)(*args)
        except Exception: pass

    # ---- URDF & Pinocchio ----
    urdf_path = _resolve_urdf_path(ARGS.urdf)
    if not os.path.exists(urdf_path):
        urdf_path = find_urdf(urdf_path)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    urdf_abs = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_abs)
    print(f"[FACTR] URDF: {urdf_abs}")

    model, _, _ = pin.buildModelsFromUrdf(urdf_abs, package_dirs=[urdf_dir])
    data = model.createData()

    # end-effector frame: prefer ee/tcp/tool/hand, else last one
    ee_fid = None
    preferred = ["ee", "tcp", "tool", "hand"]
    for i, fr in enumerate(model.frames):
        n = fr.name.lower()
        if any(k in n for k in preferred):
            ee_fid = i
    if ee_fid is None:
        ee_fid = model.nframes - 1
    print(f"[FACTR] using EE frame id={ee_fid}, name='{model.frames[ee_fid].name}'")

    # ---- ZMQ ----
    ctx = build_ctx()
    pub_q = ctx.socket(zmq.PUB); pub_q.setsockopt(zmq.SNDHWM,5); pub_q.bind("tcp://*:6001")
    pub_g = ctx.socket(zmq.PUB); pub_g.setsockopt(zmq.SNDHWM,5); pub_g.bind("tcp://*:6004")
    sub_tau = ctx.socket(zmq.SUB)
    sub_tau.setsockopt(zmq.SUBSCRIBE, b"")
    sub_tau.setsockopt(zmq.RCVHWM,5)
    sub_tau.setsockopt(zmq.CONFLATE,1)
    sub_tau.setsockopt(zmq.RCVTIMEO,0)
    sub_tau.connect("tcp://127.0.0.1:6003")
    print("[FACTR] ZMQ ready. PUB 6001/6004, SUB 6003")

    # ---- calibrate offsets (target = FACTR start pose) ----
    CALIB_POS = np.array(START_FACTR)
    for _ in range(10):
        try: drv.get_positions_and_velocities()
        except: pass
    curr, _ = drv.get_positions_and_velocities()

    def best_off(jidx):
        best, err = 0.0, 1e9
        for off in np.linspace(-20*np.pi, 20*np.pi, 81):
            joint_i = FACTR_SIGNS[jidx] * (curr[jidx] - off)
            e = abs(joint_i - CALIB_POS[jidx])
            if e < err: err, best = e, off
        return best

    offsets = np.zeros(8)
    for i in range(NUM_ARM): offsets[i] = best_off(i)
    offsets[-1] = curr[-1]
    print("[FACTR] offsets(deg):", np.round(offsets*180/np.pi, 2))

    # strict manual alignment (FACTR, default L2)
    _metric, _tol = _resolved_align_params()
    _factr_wait_manual_align(drv, offsets, tol=_tol, metric=_metric,
                             hold_s=ARGS.align_hold_s, debug=ARGS.align_debug)

    # ---- state/gains ----
    stiction_flag = np.ones(NUM_ARM, dtype=bool)
    last_tau_ext  = np.zeros(NUM_ARM, dtype=np.float32)
    prev_grip = 0.0
    NULL_KP, NULL_KD = ARGS.null_kp, ARGS.null_kd

    # torque warm start (avoid sudden jump)
    start_t = time.time()
    warmup_T = max(0.0, float(ARGS.warmup_s))

    def warm_scale(now):
        if warmup_T <= 0: return 1.0
        x = (now - start_t) / warmup_T
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        return 3*x*x - 2*x*x*x

    def get_states():
        nonlocal prev_grip
        q_raw, dq_raw = drv.get_positions_and_velocities()
        q_arm = (np.array(q_raw[:NUM_ARM]) - offsets[:NUM_ARM]) * FACTR_SIGNS[:NUM_ARM]
        dq_arm = np.array(dq_raw[:NUM_ARM]) * FACTR_SIGNS[:NUM_ARM]
        g = (q_raw[-1] - offsets[-1]) * FACTR_SIGNS[-1]
        dg = (g - prev_grip) / DT; prev_grip = g
        return q_arm, dq_arm, g, dg

    def limit_barrier(q, dq, g, dg):
        tau = np.zeros_like(q)
        exceed_max = q > ARM_LIMIT_MAX; tau += (-ARGS.limit_kp*(q-ARM_LIMIT_MAX) - ARGS.limit_kd*dq)*exceed_max
        exceed_min = q < ARM_LIMIT_MIN; tau += (-ARGS.limit_kp*(q-ARM_LIMIT_MIN) - ARGS.limit_kd*dq)*exceed_min
        tau_g = 0.0
        if g > GRIP_MAX: tau_g = -ARGS.limit_kp*(g-GRIP_MAX) - ARGS.limit_kd*dg*0.2
        elif g < GRIP_MIN: tau_g = -ARGS.limit_kp*(g-GRIP_MIN) - ARGS.limit_kd*dg*0.2
        return tau, tau_g*0.2

    def grav_comp(q, dq):
        tau_g = np.asarray(pin.rnea(model, data, q, dq, np.zeros_like(dq)), dtype=float)
        return ARGS.gravity_gain * tau_g

    def friction(dq, tau_g):
        tau = np.zeros_like(dq)
        for i in range(dq.size):
            if abs(dq[i]) < ARGS.stiction_speed:
                tau[i] += (ARGS.stiction_gain * abs(tau_g[i])) * (1 if stiction_flag[i] else -1)
                stiction_flag[i] = ~stiction_flag[i]
        return tau

    def nullspace(q, dq):
        pin.forwardKinematics(model, data, q, dq, np.zeros_like(dq))
        pin.updateFramePlacements(model, data)
        J6 = pin.computeFrameJacobian(model, data, q, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = np.asarray(J6[:6, :NUM_ARM], dtype=float)
        J_pinv = np.linalg.pinv(J)
        N = np.eye(NUM_ARM) - J_pinv @ J
        q_des = np.array(START_FACTR, dtype=float)
        q_err = q - q_des
        return N @ (-NULL_KP*q_err - NULL_KD*dq)

    print("[FACTR] control loop start...")
    t_prev = time.perf_counter(); count=0; last_log=time.time()
    try:
        while True:
            q, dq, g, dg = get_states()

            tau_lim, tau_g_l = limit_barrier(q, dq, g, dg)
            tau = np.zeros(NUM_ARM); tau += tau_lim; tau += nullspace(q, dq)
            tau_g = grav_comp(q, dq); tau += tau_g; tau += friction(dq, tau_g)

            last_tau_ext = recv_tau_nonblock(sub_tau, NUM_ARM, last_tau_ext)
            tau_fb = -(ARGS.torque_gain/94.5652173913) * last_tau_ext + (-ARGS.torque_damping)*dq
            tau += tau_fb

            s = warm_scale(time.time())
            tau *= s

            full_tau = np.append(tau, tau_g_l) * FACTR_SIGNS
            try: drv.set_torque(full_tau)
            except Exception: pass

            pub_q.send(np.asarray(q, dtype=np.float64).tobytes())
            pub_g.send(np.asarray([g], dtype=np.float64).tobytes())

            count += 1
            now=time.time()
            if now-last_log>ARGS.print_every_s:
                print(f"[FACTR] ->6001 {count/(now-last_log):.1f} Hz | q0={q[0]:.3f}")
                count=0; last_log=now

            t_prev += (1.0/ARGS.ctrl_hz)
            slp = t_prev - time.perf_counter()
            if slp>0: time.sleep(slp)
            else: t_prev = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[FACTR] stopping...")
    finally:
        for fn, args in [("set_torque_mode",(False,)), ("set_torque",(np.zeros(8),))]:
            try: getattr(drv, fn)(*args)
            except Exception: pass

# ================= main =================
if __name__ == "__main__":
    if ARGS.mode == "gello":
        run_gello_master()
    else:
        run_factr_master()
