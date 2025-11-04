import torch
import time
import numpy as np
from polymetis import RobotInterface, GripperInterface
import threading
import zmq
import copy
import h5py
import cv2
import argparse
import os


# 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Torque control with robot arm and gripper.")
    parser.add_argument('--dataset_folder', type=str, default="dataset/cup", help='Directory to save the dataset.')
    return parser.parse_args()


class ZMQSubscriber:
    """
    Creates a thread that subscribes to a ZMQ publisher
    """
    def __init__(self, ip_address="tcp://192.168.1.3:2096", verbose=False):
        context = zmq.Context()
        self._sub_socket = context.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.CONFLATE, False)
        self._sub_socket.connect(ip_address)
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, b'')

        self._subscriber_thread = threading.Thread(target=self._update_value)
        self._subscriber_thread.start()

        self._value = None
        self.verbose = verbose
        self.last_message = None

    @property
    def message(self):
        if self._value is None and self.verbose:
            print("The subscriber has not received a message")
        self.last_message = self._value
        return self._value

    def _update_value(self):
        while True:
            message = self._sub_socket.recv()
            self._value = np.frombuffer(message).astype(np.float32)


class ZMQPublisher:
    """
    Creates a thread that publishes to a ZMQ subscriber
    """
    def __init__(self, ip_address="tcp://192.168.1.3:2096"):
        context = zmq.Context()
        self._pub_socket = context.socket(zmq.PUB)
        self._pub_socket.bind(ip_address)
        self.last_message = None

    def send_message(self, message):
        self.last_message = message
        self._pub_socket.send(message.astype(np.float64).tobytes())


class CameraReader:
    def __init__(self, dev="/dev/video0", api=cv2.CAP_V4L2, fourcc="MJPG", size=(1280, 360), fps=20):
        self.dev, self.api = dev, api
        self.fourcc, self.size, self.fps = fourcc, size, fps
        self.cap = None
        self._lock = threading.Lock()
        self._stop = False
        self.latest_frame = None
        self.latest_ts = 0.0
        self._t = None

    def start(self):
        self.cap = cv2.VideoCapture(self.dev, self.api)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开 {self.dev}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self.latest_frame = frame
                self.latest_ts = time.time()

    def get_latest(self, copy_frame=True):
        with self._lock:
            f = self.latest_frame
            ts = self.latest_ts
        if f is None:
            return None, 0.0
        return (f.copy() if copy_frame else f), ts

    def stop(self):
        self._stop = True
        if self._t is not None:
            self._t.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


# 设置命令行参数
args = parse_args()

# 初始化机器人和夹爪
robot = RobotInterface(ip_address="127.0.0.1", port=50051)
gripper = GripperInterface(ip_address="127.0.0.1", port=50052)

# 获取当前关节位置
q_current = robot.get_joint_positions()
print("当前关节角度:", q_current)

# 移动机器人到目标关节位置
q_target = [0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0]
robot.move_to_joint_positions(q_target)
print("移动后的关节角度:", robot.get_joint_positions())
print(gripper.get_state())
gripper.goto(0.01, 0.2, 0.1)

# 初始化 ZMQ 订阅和发布器
subscriber = ZMQSubscriber(ip_address="tcp://127.0.0.1:6001")
publisher = ZMQPublisher(ip_address="tcp://127.0.0.1:6003")
gripper_subscriber = ZMQSubscriber(ip_address="tcp://127.0.0.1:6004")
state_publisher = ZMQPublisher(ip_address="tcp://127.0.0.1:6002")

# 初始化摄像头
cam = CameraReader(dev="/dev/video0", api=cv2.CAP_V4L2, fourcc="MJPG", size=(2560, 720), fps=20)
cam.start()

# 数据收集参数
init_time = -100
record_count = 0
save_freq = 20  # 保存数据的频率（每秒）
first_save = False

# 存储数据的列表
qpos_list = []
action_list = []
image_list = []

# 主控制循环（500 Hz）
num = 0
tt = 0
last_frame = None
last_gripper_mode = None

def save_data_on_exit():
    # 保存数据到文件
    if len(image_list) > 0:
        Observation = dict()
        Observation['image_diagonal_view'] = image_list
        Observation['action'] = action_list
        Observation['qpos'] = qpos_list

        # 检查目录是否存在，不存在则创建
        if not os.path.exists(args.dataset_folder):
            os.makedirs(args.dataset_folder)

        # 保存到新的文件
        count = sum(1 for f in os.listdir(args.dataset_folder) if f.endswith(".hdf5"))
        with h5py.File(f'{args.dataset_folder}/episode_{count}.hdf5', 'w') as file:
            for key, value in Observation.items():
                file.create_dataset(key, data=value, compression='gzip', compression_opts=9)
        print(f'{args.dataset_folder}/episode_{count}.hdf5 saved')


try:
    dt = 1.0 / 500  # 500 Hz
    while True:
        tt += 1
        q_target = subscriber.message  # np.array, dtype=float32

        if q_target is not None and len(q_target) == 7:
            num += 1

            if num <= 4:
                robot.move_to_joint_positions(q_target)
                if num == 4:
                    robot.start_joint_impedance(adaptive=False)
                continue

            q_tensor = torch.from_numpy(q_target).float()
            robot.update_desired_joint_positions(q_tensor)
            
            # 根据订阅的消息控制夹爪
            gripper.goto(gripper_subscriber.message * 0.15, 0.5, 0.5, False)
            
            # 存储动作和关节位置数据
            action = np.zeros(8)
            action[0:7] = subscriber.message
            action[7] = gripper_subscriber.message[0]

            qpos = np.zeros(8)
            qpos[0:7] = robot.get_joint_positions()
            qpos[7] = gripper.get_state().width

            # 定期保存数据
            if time.time() - init_time > 1.0 / save_freq * record_count:
                if not first_save:
                    init_time = time.time()
                    first_save = True
                record_count += 1

                frame, ts = cam.get_latest(copy_frame=True)
                if frame is not None:
                    H, W, C = frame.shape
                    frame = frame[:, W // 2:, :]
                    frame = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)

                    qpos_list.append(copy.deepcopy(qpos))
                    action_list.append(copy.deepcopy(action))
                    image_list.append(copy.deepcopy(frame))

            # 发布力矩和状态
            tau = np.array(robot.get_robot_state().motor_torques_external, dtype=np.float32)
            publisher.send_message(tau)
            state_publisher.send_message(np.array(robot.get_joint_positions(), dtype=np.float32))

except KeyboardInterrupt:
    print("程序中断，正在保存数据...")
    save_data_on_exit()
    print("数据保存完毕，程序退出")
