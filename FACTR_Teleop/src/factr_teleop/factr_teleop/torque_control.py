import torch
import time
import numpy as np
from polymetis import RobotInterface,GripperInterface
import threading
import zmq
import copy
import h5py
import cv2




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
        while(True):
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

# ------------------------------
# 1. 初始化机器人
# ------------------------------
robot = RobotInterface(ip_address="127.0.0.1", port=50051)
gripper = GripperInterface(ip_address="127.0.0.1",port=50052)

q_current = robot.get_joint_positions() 
print("当前关节角度:", q_current) 
q_target = [0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0] 
robot.move_to_joint_positions(q_target) 
print("移动后的关节角度:", robot.get_joint_positions())
print(gripper.get_state())
gripper.goto(0.01,0.2,0.1)
# 启动关节阻抗控制（非阻塞）
# robot.start_joint_impedance(adaptive=True)

# ------------------------------
# 2. 初始化 ZMQ 订阅
# ------------------------------
subscriber = ZMQSubscriber(ip_address="tcp://127.0.0.1:6001")
publisher = ZMQPublisher(ip_address="tcp://127.0.0.1:6003")
gripper_subscriber = ZMQSubscriber(ip_address="tcp://127.0.0.1:6004")
state_publisher = ZMQPublisher(ip_address="tcp://127.0.0.1:6002")

# ------------------------------
# 3. 实时控制循环（500 Hz）
# ------------------------------



class CameraReader:
    def __init__(self, dev="/dev/video0", api=cv2.CAP_V4L2, fourcc="MJPG",
                 size=(1280, 360), fps=20):
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
        # 按设备支持实际设置（确保这组参数是 v4l2-ctl 列出的）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # 尽量降缓存
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                # 轻微等待，避免占满 CPU；如果持续失败再考虑重启 cap
                time.sleep(0.005)
                continue
            with self._lock:
                self.latest_frame = frame
                self.latest_ts = time.time()

    def get_latest(self, copy_frame=True):
        with self._lock:
            f = self.latest_frame
            #print(np.shape(f))
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


init_time = -100
record_count = 0
save_freq = 20  # 每秒保存数据的频率
first_save = False



cam = CameraReader(dev="/dev/video0", api=cv2.CAP_V4L2, fourcc="MJPG",
                   size=(2560, 720), fps=20)   # 如需试 /dev/video1 就改这里  # <<<
cam.start()  # <<< 只 start 一次


num=0

qpos_list = []
action_list = []
image_list = []

tt = 0

last_frame = None

last_gripper_mode = None

try:
    
    dt = 1.0 / 500 # 500 Hz
    # next_time = time.perf_counter()
    
    while True:
        #print(tt)
        tt+=1
        q_target = subscriber.message  # np.array, dtype=float32
        
        

        
        if q_target is not None and len(q_target) == 7:
            # 转成 torch.Tensor 并更新关节阻抗控制目标
            num+=1
            if num<=4:
                robot.move_to_joint_positions(q_target)
                # time.sleep(0.2)  # 第一次收到命令后，等待2秒再执行，确保机器人已经进入阻抗控制模式
                # 启动关节阻抗控制（非阻塞）
                if num==4:
                    robot.start_joint_impedance(adaptive=False)
                continue
            q_tensor = torch.from_numpy(q_target).float()
            robot.update_desired_joint_positions(q_tensor)
            # gripper.goto(gripper_subscriber.message[0],0.2,0.1,False)
            # gripper.grasp(0.2,0.1,gripper_subscriber.message,blocking=False)
            
            
            '''
            if gripper_subscriber.message >= 0.04:
                print("open gripper")
                gripper.goto(0.078,0.2,0.1,False)
            else:
                print("close gripper")
                gripper.grasp(0.2,0.1,blocking=False)
            '''
            
            '''
            want_open = (gripper_subscriber.message >= 0.04)  # 注意：g_msg是numpy数组，所以用 g_msg[0]

            if want_open and last_gripper_mode != "open":
                print("open gripper (edge)")
                gripper.goto(0.078, 0.2, 0.1, False)
                last_gripper_mode = "open"

            elif (not want_open) and last_gripper_mode != "close":
                print("close gripper (edge)")
                gripper.grasp(0.2, 0.1, blocking=False)
                last_gripper_mode = "close"
            '''
            
            #print(gripper_subscriber.message)
            gripper.goto(gripper_subscriber.message*0.15, 0.5, 0.5, False)
            
            action = np.zeros(8)
            action[0:7] = subscriber.message
            action[7] = gripper_subscriber.message[0]
            
            qpos = np.zeros(8)
            qpos[0:7] = robot.get_joint_positions()
            qpos[7] = gripper.get_state().width


            # frame, ts = cam.get_latest(copy_frame=True) 
            # if(frame is not None):
            #     last_frame = copy.deepcopy(frame)
            #     H, W, C = last_frame.shape
            #     last_frame = last_frame[:, W//2:]
                
            #with open ("gripper.txt","a") as f:
            #     f.write(f"{gripper_subscriber.message}\n")
            #with open ("arm.txt","a") as f:
            #     f.write(f"{subscriber.message}\n")
            if(time.time()-init_time>1.0/save_freq*record_count):
                if first_save==False:
                    init_time = time.time()
                    first_save=True
                record_count+=1
                
                #with open ("action_test.txt","a") as f:
                #    f.write(f"{action} {(time.time()-init_time)*save_freq}\n")
                #with open ("qpos_test.txt","a") as f:
                #    f.write(f"{qpos} {(time.time()-init_time)*save_freq}\n")
                frame, ts = cam.get_latest(copy_frame=True) 
                H, W, C = frame.shape
                #print(np.shape(frame))
                frame = frame[:, W//2:,:]
                H, W, C = frame.shape
                frame = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
                #print(np.shape(frame))
                qpos_list.append(copy.deepcopy(qpos))
                action_list.append(copy.deepcopy(action))
                image_list.append(copy.deepcopy(frame))
                #print(np.shape(qpos_list))
            
            tau = np.array(robot.get_robot_state().motor_torques_external, dtype=np.float32)
            publisher.send_message(tau)
            state_publisher.send_message(np.array(robot.get_joint_positions(), dtype=np.float32))
            
        #print(len(image_list))
        if(len(image_list) ==100400 and len(image_list)>0):
            #print("********")
            Observation = dict()
            print(np.shape(image_list))
            print(np.shape(action_list))
            print(np.shape(qpos_list))
            Observation['image_diagonal_view'] = image_list
            
            Observation['action'] = action_list
            Observation['qpos'] = qpos_list
            #with h5py.File('dataset/grasp/episode_' + str(self.success_num) + '.hdf5', 'w') as file:
            
            import os
            folder = "dataset/cup"
            count = sum(1 for f in os.listdir(folder) if f.endswith(".hdf5"))
            
            with h5py.File(f'dataset/cup/episode_{count}.hdf5', 'w') as file:
                for key, value in Observation.items():
                    file.create_dataset(key, data=value, compression='gzip', compression_opts=9)
        # robot.update_desired_joint_positions(torch.tensor([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0], dtype=torch.float32))
        # 精确控制 500 Hz
        # next_time += dt
        # sleep_time = next_time - time.perf_counter()
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        # else:
        #     # 如果处理太慢，直接跳过
        #     next_time = time.perf_counter()
        # time.sleep(0.002)

except KeyboardInterrupt:

    print("程序中断，停止控制")


# while True:
#     tau = robot.get_robot_state().motor_torques_external
#     print("当前关节力矩:", tau)
#     tag=True
#     for items in tau:
#         if abs(items)>1.0:
#             print("force detected")
#             tag=False
#     if tag==False:
#         break
#     time.sleep(0.01)  # 100Hz读取
