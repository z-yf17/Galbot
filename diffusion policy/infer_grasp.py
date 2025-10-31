import random
import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import math
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from dm_control import mujoco
import h5py
import torch
import cv2
import argparse
import pickle
#from policy import ACTPolicy
from policy import DiffusionPolicy

import time

# 设置种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 执行推理并测试
def inference():
    #for i in range(18200, 20001, 200):  # 从18200到20000，每次递增200
    for i in range(96000,100000,500):
        model_file = f"ckpt/grasp_target_2/policy_step_{i}_seed_0.ckpt"
        print(f"Loading model: {model_file}")
        
        # 设置种子
        set_seed(42)
        
        # 初始化环境
        env = Env_q()
        env.load_model()
        env.policy.deserialize(torch.load(model_file))
        
        success_num = 0
        result = []
        
        # 测试100次
        for test_num in range(100):
            tt = time.time()
            print(f"### Test {test_num + 1} ###")
            
            env.initial_model()
            env.initial_episode()
            env.infer(f'video/grasp/episode_{test_num + 1}.mp4')
            success = env.judge_succeed()
            success_num += success
            result.append(success)
            print(f"Successes: {success_num}/{test_num + 1}")
            print(f"Time taken: {time.time() - tt} seconds")
            
            # 保存每次测试的结果
        save_results(result, i, test_num)
        
        # 每次完成模型的所有测试后，打印进度
        print(f"Finished testing model epoch {i}")

# 保存结果到文件
def save_results(result, model_epoch, test_num):
    with open('inference_result/result-grasp_target.txt', 'a') as file:
        file.write(f"Model epoch: {model_epoch}, Test: {test_num + 1}\n")
        file.write(f"Result: {result[-1]}\n")  # 保存当前测试的结果
        file.write(f"Success rate up to this test: {sum(result) / len(result):.4f}\n")
        file.write("\n")

class Env_q():
    def __init__(self, path_q=None):
        super().__init__()
        self.path_q = path_q if path_q else "franka_emika_panda/scene_grasp.xml"
        self.physics_q = mujoco.Physics.from_xml_path(self.path_q)
        self.render_q = False
        self.step_id_q = 0
        self.sample_indent = 2
        self.success_num = 0
        self.trajectory = []
        self.qpos = []
        self.initial_episode()
        self.infer_step = 200
        self.diffusion_policy = None
        self.past_action = []
        self.interval = 10
        self.img_list = []
        self.action_list = []
        
        self.result = []
        self.policy_config = {'lr': 2e-5,
                         'camera_names': ['image_diagonal_view'],
                         'action_dim': 8,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': 64,
                         'num_queries': 64,
                         'num_inference_timesteps': 50,
                         'ema_power': 0.99,
                         'vq': False,
                         }
        self.config = {
            'num_steps': 20000,

            'state_dim': 8,

            'temporal_agg': False,
            'camera_names': 'image_diagonal_view',
            #'actuator_config': actuator_config,
        }

        self.load_model()
        with open('ckpt/grasp_target_2/dataset_stats.pkl', 'rb') as file:
            self.norm_stats = pickle.load(file)

    def initial_model(self):
        rand = np.random.uniform(-0.03, 0.03, 7)
        new_state = np.zeros(31)
        new_state[0:len(np.array(self.physics_q.named.data.qpos))] = self.physics_q.named.data.qpos.copy()
        new_state[0:7] = [0, 0.46, 0, -1.47, 0, 2.3, 3.1415926 / 4] + rand
        new_state[7:9] = 0
        self.physics_q.set_state(np.array(new_state))
        self.physics_q.forward()

    def initial_episode(self):
        x = 0.6 + random.random()*0.15
        y = -0.15 + random.random()*0.3

        np.copyto(self.physics_q.data.qpos[9 + 7 * 0: 9 + 7 * 1], [x,y,0.02,1,0,0,0])
        self.physics_q.step()

    def infer(self, save_path):
        self.img_list = []
        action_agg = []
        for i in range(800):
            action_agg.append([])

        count = 0
        for t in range(int(self.infer_step) * self.sample_indent):
            img, qpos = self.get_env_state()
            qpos = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            cond = np.expand_dims(img, axis=0)
            cond = np.expand_dims(cond, axis=0)
            cond = torch.tensor(cond.copy()).float() / 255
            qpos = np.expand_dims(qpos, axis=0)

            if t % self.sample_indent == 0:
                ts = int(t / self.sample_indent)
                if ts % self.interval == 0:
                    action_list = self.policy(torch.tensor(qpos).float().cuda(), cond.cuda())[0].detach().cpu().numpy()
                    
                    tt = action_list * self.norm_stats["action_std"] + self.norm_stats["action_mean"]

                    

                    #print(tt)
                    #exit()
                    for i in range(len(action_list)):
                        action_agg[ts + i].append(action_list[i])

                target_qpos = np.mean(action_agg[ts], axis=0)
                #target_qpos = target_qpos * self.norm_stats["action_std"] + self.norm_stats["action_mean"]

                target_qpos = ((target_qpos + 1) / 2) * (self.norm_stats["action_max"] - self.norm_stats["action_min"]) + self.norm_stats["action_min"]
                
                
                self.action_list = self.action_list[1:]

                count = count + 1

                pd_value = target_qpos - self.physics_q.data.qpos[0:8].copy()
                self.physics_q.data.ctrl[0:8] = target_qpos.copy()

                img_show = np.concatenate(
                    (self.get_camera_img_q("left_side"), self.get_camera_img_q("diagonal_view")), axis=1)
                self.img_list.append(img_show)
            self.physics_q.step()

        if self.judge_succeed():
            save_path = save_path.replace('task2', 'task2_successful')
        else:
            save_path = save_path.replace('task2', 'task2_failed')
        print(save_path)
        self.images_to_video(self.img_list, save_path)

    def load_model(self):
        enc_layers = 4
        dec_layers = 7
        nheads = 8

        self.policy = DiffusionPolicy(self.policy_config)

    def get_camera_img_q(self, camera_name, width=640, height=480):
        width, height = 120, 160
        camera_id = self.physics_q.model.name2id(camera_name, 'camera')
        image = self.physics_q.render(width, height, camera_id=camera_id)
        return image

    def get_env_state(self):
        image_left_side = self.get_camera_img_q("diagonal_view")
        image_left_side = np.transpose(image_left_side, (2, 0, 1))
        qpos = self.physics_q.data.qpos[0:8].copy()
        return image_left_side, qpos

    def images_to_video(self, image_list, output_file, fps=30):
        frame = image_list[0]
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for image in image_list:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image_bgr)
        video.release()

    def judge_succeed(self):
        red_box_index = self.physics_q.model.name2id('red_box', 'body')
        red_box_position = self.physics_q.data.xpos[red_box_index]

        print(red_box_position)


        if(red_box_position[2]>0.1):
            self.success = True
            print("success")
            return True
        else:
            self.success = False
            print('failed')
            return False

if __name__ == '__main__':
    inference()
