import torch
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange  # kept for potential downstream use
from torchvision import transforms  # kept for potential downstream use

from utils import load_data
from utils import compute_dict_mean, set_seed
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy


def get_auto_index(dataset_dir):
    """Return the first unused integer index for files like qpos_{i}.npy."""
    max_idx = 1000
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def main(args):
    set_seed(1)

    # === Keep only parameters used by your two commands ===
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy']              # --policy (ACT / Diffusion / CNNMLP)
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    lr = args['lr']
    seed = args['seed']
    chunk_size = args['chunk_size']
    hidden_dim = args.get('hidden_dim', None)
    dim_feedforward = args.get('dim_feedforward', None)
    kl_weight = args.get('kl_weight', None)

    # Fixed periodicities (previously CLI flags)
    eval_every = 10000
    validate_every = 5000
    save_every = 5000

    # Task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim or task_name == 'all':
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']         # stored in config; not used for evaluation here
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # Fixed model-side parameters
    state_dim = 8
    lr_backbone = 1e-5
    backbone = 'resnet18'

    # Build policy_config matching your commands
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'lr': lr,
            'num_queries': chunk_size,
            'kl_weight': kl_weight,
            'hidden_dim': hidden_dim,
            'dim_feedforward': dim_feedforward,
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': camera_names,
            'action_dim': 8,
            # Stable defaults to keep things simple/compatible
            'vq': False,
            'vq_class': None,
            'vq_dim': None,
            'no_encoder': False,
        }
    elif policy_class == 'Diffusion':
        policy_config = {
            'lr': lr,
            'camera_names': camera_names,
            'action_dim': 8,
            'observation_horizon': 1,
            'action_horizon': 8,
            'prediction_horizon': chunk_size,
            'num_queries': chunk_size,
            'num_inference_timesteps': 20,
            'ema_power': 0.99,
            'vq': False,
        }
    elif policy_class == 'CNNMLP':
        policy_config = {
            'lr': lr,
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'num_queries': 1,
            'camera_names': camera_names,
        }
    else:
        raise NotImplementedError(f"Unknown policy: {policy_class}")

    # Minimal training config
    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': lr,
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': seed,
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': False,  # fixed as False for simplicity
    }

    # Prepare output dir & persist config
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    # Data
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        name_filter,
        camera_names,
        batch_size_train,
        batch_size_val,
        chunk_size,
        False,                     # skip_mirrored_data: fixed as False
        config['load_pretrain'],   # False
        policy_class,
        stats_dir_l=stats_dir,
        sample_weights=sample_weights,
        train_ratio=train_ratio
    )

    # Save dataset statistics
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Train
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    # All three policies delegate optimizer creation to their own method
    return policy.configure_optimizers()


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.cuda()
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)

    # No resume/load_pretrain here to keep things minimal
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = np.inf
    best_ckpt_info = None

    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps + 1)):
        # Validation
        if step % validate_every == 0:
            print('validating')
            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)
                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # Periodic checkpoint (no online evaluation)
        if (step > 0) and (step % eval_every == 0):
            ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)

        # Train step
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()

        # Late-stage frequent saves (kept from original logic)
        if step % save_every == 0 and step > 90000:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

    # Final save
    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info


def repeater(data_loader):
    """Endless iterator over a dataloader, logging epoch boundaries."""
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Only keep parameters used by your commands
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)

    # Keep --policy to match your commands (ACT / Diffusion / CNNMLP)
    parser.add_argument('--policy', type=str, required=True)

    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=False)       # used by ACT
    parser.add_argument('--dim_feedforward', type=int, required=False)  # used by ACT
    parser.add_argument('--kl_weight', type=int, required=False)        # used by ACT

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_steps', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)

    main(vars(parser.parse_args()))
