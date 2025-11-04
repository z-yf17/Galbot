import pathlib
import os

### Task parameters
DATA_DIR = '/home/galbot/zyf/act/dataset'
SIM_TASK_CONFIGS = {
    'sim_grasp_target': {
        'dataset_dir': DATA_DIR + '/grasp_target',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['image_diagonal_view']
    },
    'sim_wipe_target': {
        'dataset_dir': DATA_DIR + '/wipe_target',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['image_diagonal_view']
    },
    'sim_cup': {
        'dataset_dir': DATA_DIR + '/cup',
        'num_episodes': 60,
        'episode_len': 500,
        'camera_names': ['image_diagonal_view']
    },
}

### Simulation envs fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/'  # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2

### Hyperparameters for training
TRAINING_PARAMS = {
    'policy_class': 'ACT',
    'lr': 2e-5,
    'chunk_size': 64,  # The size of the chunk of data to process
    'batch_size': 8,   # The batch size for training
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'num_steps': 100000,
    'kl_weight': 10,
    'seed': 0,
    'eval_every': 500,
    'validate_every': 500,
    'save_every': 500
}

### Policy-specific configuration
POLICY_CONFIGS = {
    'ACT': {
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'lr': TRAINING_PARAMS['lr'],
        'num_queries': TRAINING_PARAMS['chunk_size'],
        'kl_weight': TRAINING_PARAMS['kl_weight'],
        'hidden_dim': TRAINING_PARAMS['hidden_dim'],
        'dim_feedforward': TRAINING_PARAMS['dim_feedforward'],
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'camera_names': ['image_diagonal_view'],
        'vq': False,
        'action_dim': 16
    },
    'Diffusion': {
        'lr': TRAINING_PARAMS['lr'],
        'camera_names': ['image_diagonal_view'],
        'action_dim': 8,
        'observation_horizon': 1,
        'action_horizon': 8,
        'prediction_horizon': TRAINING_PARAMS['chunk_size'],
        'num_queries': TRAINING_PARAMS['chunk_size'],
        'num_inference_timesteps': 50,
        'ema_power': 0.99,
        'vq': False
    },
    'CNNMLP': {
        'lr': TRAINING_PARAMS['lr'],
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'num_queries': 1,
        'camera_names': ['image_diagonal_view']
    }
}

### Actuator configuration
ACTUATOR_CONFIG = {
    'actuator_network_dir': None,
    'history_len': 10,
    'future_len': 10,
    'prediction_len': 10,
}

### Main configuration
MAIN_CONFIG = {
    'num_steps': TRAINING_PARAMS['num_steps'],
    'eval_every': TRAINING_PARAMS['eval_every'],
    'validate_every': TRAINING_PARAMS['validate_every'],
    'save_every': TRAINING_PARAMS['save_every'],
    'ckpt_dir': 'ckpt/cup',  # Example path, can be modified based on the task
    'resume_ckpt_path': None,  # Path to resume checkpoint, if required
    'episode_len': SIM_TASK_CONFIGS['sim_cup']['episode_len'],
    'state_dim': 8,
    'lr': TRAINING_PARAMS['lr'],
    'policy_class': TRAINING_PARAMS['policy_class'],
    'onscreen_render': False,  # False by default
    'policy_config': POLICY_CONFIGS[TRAINING_PARAMS['policy_class']],
    'task_name': 'sim_cup',
    'seed': TRAINING_PARAMS['seed'],
    'temporal_agg': False,
    'camera_names': SIM_TASK_CONFIGS['sim_cup']['camera_names'],
    'real_robot': False,  # Not using real robot
    'load_pretrain': False,  # No pretraining
    'actuator_config': ACTUATOR_CONFIG,
    'batch_size': TRAINING_PARAMS['batch_size'],  # Ensure batch size is included in main config
    'chunk_size': TRAINING_PARAMS['chunk_size']
}
