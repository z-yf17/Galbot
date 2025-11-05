````markdown
# Installation of Polymetis

> **Note**  
> Polymetis is highly environment-dependent, so you must install the specific library versions exactly as required.  
> Ideally, isolate this environment and have it communicate with other environment that need other libraries via ZMQ.

---

## 📦 Clone repo

```bash
git clone git@github.com:facebookresearch/fairo
cd fairo/polymetis
````

---

## 🧪 Create Polymetis environment

```bash
conda env create -f ./polymetis/environment.yml
conda activate polymetis-local
```

---

## 🐍 Install Python package in editable mode

```bash
pip install -e ./polymetis
```

---

## 🛠️ Build Frankalib from source

```bash
./scripts/build_libfranka.sh <version_tag_or_commit_hash>
```

<small>Franka Emika Research 3 needs version 0.15+. By default, the above command-line instructions prioritize binding libfranka to the ROS path.</small>

```bash
./scripts/build_libfranka_conda.sh <version_tag_or_commit_hash>
```

*To build a ROS-free, conda-only dependency setup of Polymetis, you need to run this command.*

---

## 🧱 Build Polymetis from source

```bash
mkdir -p ./polymetis/build
cd ./polymetis/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=ON -DBUILD_TESTS=ON -DBUILD_DOCS=ON
make -j
```

---

## 🖥️ Run FACTR (requires **4 terminals**)

> **Tip**
> Open four separate terminals. Each terminal activates the same Conda env and runs one process.
> Commands below are **copy-ready** and keep your original settings unchanged.

```bash
cd galbot
```

### Terminal 1 — Robot Client

```bash
# terminal 1
conda activate polymetis-local
taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
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
taskset -c 16,17,18,19,20,21,22,23 \
  chrt -f 97 \
  launch_gripper.py \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=192.168.1.10 \
  gripper.executable_cfg.control_port=50052 \
  gripper.executable_cfg.control_ip=127.0.0.1
```

### Terminal 3 — Torque Control

```bash
# terminal 3
conda activate polymetis-local
cd FACTR_Teleop/src/factr_teleop/factr_teleop
taskset -c 24,25,26,27 \
  chrt -f 98 \
  python3 torque_control.py --dataset_folder your_dataset_path
```
Ctrl + C to stop and save the data.

### Terminal 4 — Leader Teleop (No ROS 2)

```bash
# terminal 4
conda activate polymetis-local
cd FACTR_Teleop
taskset -c 28,29,30,31 \
  chrt -f 98 \
  python3 scripts/leader_teleop_no_ros_2.py
```

---

## 📚 Create Imitation Learning Environment

Due to the fact that the RT real-time kernel does not support CUDA, you need to set up communication via ZMQ locally. The RT environment will handle the control of the Franka robot arm, while GPU computation will be done in the non-RT environment. The setup process is as follows:

### Steps:

1. **Create the GPU environment (non-RT environment):**  
   Set up a dedicated environment for GPU computation, ensuring CUDA is properly installed and usable. This environment will handle deep learning or other GPU-intensive tasks.

2. **Install ZMQ in both environments:**  
   You need to install ZMQ both in the newly created GPU environment and in the Polymetis environment. This allows communication between the RT-controlled robot arm and the GPU computation environment.

### Setup Instructions:

#### 1. Create the GPU environment:

```bash
conda create -n env_IL python=3.8
conda activate env_IL
cd Behavior/Cloning
pip install -r requirements.txt
cd IL/Policy/robomimic
pip install -e .
cd ..
```

#### 2. Install ZMQ and Opencv in both environment:

```bash
conda activate polymetis-local
conda install pyzmq --freeze-installed
conda install -c conda-forge opencv --freeze-installed
```

### Train your IL Policy:

```bash
cd Behavior\ Cloning
python IL\ policy/imitate_episodes.py
```
Remember to chage your config in IL\ Policy/constants.py
