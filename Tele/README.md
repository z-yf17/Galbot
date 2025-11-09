# Gello and FACTR Tele
## Environment Configuration
First, confirm that your computer has a real-time kernel and polymetis installed.

Then configure the environment：
```bash
conda create --name Tele_envs python=3.11
conda activate Tele_envs
pip install -r requirements.txt
conda install -c conda-forge pyqt=6
pip install -e .
pip install -e third_party/DynamixelSDK/python
```

## Quick Start

terminal 1:
```bash
conda activate polymetis-local
cd <Polymetis installation path>/fairo/polymetis/polymetis/python/scripts
python launch_robot.py \
  ip=0.0.0.0 port=50051 \
  robot_client=franka_hardware \
  robot_client.executable_cfg.exec=franka_panda_client \
  robot_client.executable_cfg.robot_ip=192.168.1.10 \
  robot_client.executable_cfg.use_real_time=true
  robot_client.executable_cfg.control_port=50051
```

terminal 2:
```bash
conda activate polymetis-local
cd <Polymetis installation path>/fairo/polymetis/polymetis/python/scripts
python launch_gripper.py \
  ip=0.0.0.0 port=50052 \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=192.168.1.10 \
  gripper.executable_cfg.control_ip=127.0.0.1 \
  gripper.executable_cfg.control_port=50052
```
The gripper will open and close once.

terminal 3:
```bash
conda activate polymetis-local
cd tele/scripts
```

gello:
```bash
python franka_follower.py --mode gello
```
FACTR:
```bash
python franka_follower.py --mode factr
```

terminal 4:
```bash
conda activate tele_envs
cd tele/scripts
python master_unified.py --mode gello
```

gello:
```bash
python master_unified.py --mode gello
```
FACTR:
```bash
python master_unified.py --mode factr
```