````markdown
# Imitation Learning Environment (RT Control + GPU Compute via ZMQ)

Because the real-time (RT) kernel does not support CUDA, this repository uses local **ZMQ** communication to split the system into two parts:
- **RT environment (Polymetis):** real-time control of the Franka arm/gripper.
- **GPU environment (non-RT):** deep learning training and inference (e.g., ACT / Diffusion).

ZMQ passes observations and actions between the two environments.

---

## Step Overview
1. Create the GPU environment (non-RT)
2. Install ZMQ (and OpenCV) in both environments
3. Train IL policies
4. Run inference on a real Franka
5. Start the GPU-side inference server (model selection & parameters)
6. Start the RT-side inference script

---

## 1. Create the GPU environment (non-RT)

```bash
conda create -n env_IL python=3.8
conda activate env_IL

cd "Behavior Cloning"
pip install -r requirements.txt

cd robomimic
pip install -e .
cd ..
````

---

## 2. Install ZMQ & OpenCV in both environments

Install in the **Polymetis (RT) environment**:

```bash
conda activate polymetis-local
conda install pyzmq --freeze-installed
conda install -c conda-forge opencv --freeze-installed
```


---

## 3. Train IL Policies

Run in the **GPU environment** (`env_IL`):

```bash
# ACT
python imitate_episodes.py \
  --task_name sim_cup \
  --ckpt_dir ckpt/your_act_path \
  --policy ACT \
  --chunk_size 32 \
  --dim_feedforward 3200 \
  --hidden_dim 512 \
  --batch_size 8 \
  --num_steps 200000 \
  --lr 2e-5 \
  --seed 0 \
  --kl_weight 10

# Diffusion
python imitate_episodes.py \
  --task_name sim_cup \
  --ckpt_dir ckpt/your_diffusion_path \
  --policy Diffusion \
  --chunk_size 32 \
  --hidden_dim 512 \
  --batch_size 8 \
  --num_steps 200000 \
  --lr 2e-5 \
  --seed 0
```

---

## 4. Inference on Real Franka

Start the robot client and the gripper client in the **RT environment**.

### Terminal 1 — Arm

```bash
# terminal 1
conda activate polymetis-local
taskset -c 0-14 \
  chrt -f 98 \
  launch_robot.py \
  robot_client=franka_hardware \
  robot_client.executable_cfg.robot_ip=192.168.1.10 \
  robot_client.executable_cfg.exec=franka_panda_client \
  robot_client.executable_cfg.use_real_time=true \
  robot_client.executable_cfg.control_port=50051
```

### Terminal 2 — Gripper

```bash
# terminal 2
conda activate polymetis-local
taskset -c 16-23 \
  chrt -f 97 \
  launch_gripper.py \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=192.168.1.10 \
  gripper.executable_cfg.control_port=50052 \
  gripper.executable_cfg.control_ip=127.0.0.1
```

---

## 5. Start the GPU Server (model selection & parameters)

Run in the **GPU environment** (`env_IL`):

```bash
conda activate env_IL
# Start the server for the policy you will use.
python gpu_server_act.py
# or
python gpu_server_diffusion.py
```

**What lives here:** model selection and inference parameters.

Configure these **inside the server script(s)** (or via their CLI flags if provided by your implementation):

* **Checkpoint path** (e.g., ACT/Diffusion weights saved from training).
* **ZMQ endpoint** (bind/connect address and port used by the RT side).
* **Device** (e.g., `cuda:0`), batch size, and any preprocessing settings.
* **Policy-specific options** (e.g., sequence/chunk length, observation keys, normalization).

> In short: choose the server (`ACT` or `Diffusion`) and set all inference-related options (checkpoint, endpoints, device, etc.) here before starting.

---

## 6. Start Inference (RT side)

Run in the **Polymetis** environment:

```bash
conda activate polymetis-local
python inference.py
```

```
```
