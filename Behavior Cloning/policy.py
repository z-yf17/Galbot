import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

from collections import OrderedDict
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D


from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel


'''
class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        self.camera_names = args_override['camera_names']

        self.observation_horizon = args_override['observation_horizon']  # 仍保持原始用法
        self.action_horizon = args_override['action_horizon']
        self.prediction_horizon = args_override['prediction_horizon']
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.ema_power = args_override['ema_power']
        self.lr = args_override['lr']
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override['action_dim']  # 14 + 2
        # 仍保持 obs_dim 不变：拼接(每相机64) + qpos(8)
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 8

        # ── 视觉主干 ───────────────────────────────────────────────────────────────
        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbones.append(ResNet18Conv(**{
                'input_channel': 3, 'pretrained': True, 'input_coord_conv': False
            }))
            pools.append(SpatialSoftmax(**{
                'input_shape': [512, 6, 10], 'num_kp': self.num_kp,
                'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0
            }))
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)

        backbones = replace_bn_with_gn(backbones)  # 保持原有替换

        # ── 噪声预测网络（保持条件维度不变，以兼容现有pipeline）──────────────
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # ── 调度器：v_prediction ────────────────────────────────────────────────
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='v_prediction'
        )

        # ── 轻量融合增强（新增，保持输出维度不变）──────────────────────────────
        # 复用的图像归一化（避免每次forward新建）
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # 每相机特征(64维)的 LayerNorm，统一尺度
        self.vis_lns = nn.ModuleList([
            nn.LayerNorm(self.feature_dimension) for _ in self.camera_names
        ])

        # qpos 的 LayerNorm（默认8维；若你的qpos不是8维，请相应修改obs_dim与这里的维度）
        self.qpos_ln = nn.LayerNorm(8)

        # 相机级可学习权重（不同视角“话语权”不同）
        self.cam_logits = nn.Parameter(torch.zeros(len(self.camera_names)))  # sigmoid后∈(0,1)

        # 分支全局可学习权重：视觉 vs qpos
        self.branch_logits = nn.Parameter(torch.zeros(2))  # [w_vis, w_qpos]

        # 轻量 FiLM：qpos -> (γ, β) 对“拼接后的视觉向量”做仿射调制；不改变维度
        self.use_film = True
        self.vis_dim = self.feature_dimension * len(self.camera_names)
        self.film = nn.Linear(8, self.vis_dim * 2)
        # 避免调制过强的小系数（可在 args_override 中通过 film_*_scale 覆盖）
        self.film_gamma_scale = args_override.get('film_gamma_scale', 0.1)
        self.film_beta_scale = args_override.get('film_beta_scale', 0.1)

        # 训练期模态Dropout：偶尔遮蔽某一路，防止只走qpos捷径
        self.moddrop_p = args_override.get('moddrop_p', 0.15)

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    # ───────────────────────────────────────────────────────────────────────────
    # 统一的 obs_cond 构造：强化“视觉+qpos”的融合，同时保持尺寸不变
    # ───────────────────────────────────────────────────────────────────────────
    def _build_obs_cond(self, nets, qpos, image, actions):
        B = qpos.shape[0]
        C = len(self.camera_names)

        cam_feats = []
        for cam_id in range(C):
            cam_image = image[:, cam_id]
            # 若输入是uint8，则转为float并/255；否则假设已在[0,1]
            if cam_image.dtype == torch.uint8:
                cam_image = cam_image.float() / 255.0
            else:
                cam_image = cam_image.float()
            cam_image = self.normalize(cam_image)

            cam_features = nets['policy']['backbones'][cam_id](cam_image)
            pool_features = nets['policy']['pools'][cam_id](cam_features)
            pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = nets['policy']['linears'][cam_id](pool_features)

            # 每相机LayerNorm，统一特征尺度
            out_features = self.vis_lns[cam_id](out_features)
            cam_feats.append(out_features)

        # 拼接所有相机特征： (B, 64*C)
        vis_feat = torch.cat(cam_feats, dim=1)

        # （可选）FiLM 调制视觉：qpos->(γ,β)，只影响内容，不改维度
        if self.use_film:
            qpos_norm = self.qpos_ln(qpos)  # (B,8)
            film_params = self.film(qpos_norm)  # (B, 2*vis_dim)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)  # (B,vis_dim),(B,vis_dim)
            gamma = torch.tanh(gamma) * self.film_gamma_scale
            beta = torch.tanh(beta) * self.film_beta_scale
            vis_feat = vis_feat * (1.0 + gamma) + beta  # (B,vis_dim)

        # 相机级权重：对每个相机的64D特征乘以独立权重
        if C > 0:
            cam_w = torch.sigmoid(self.cam_logits)  # (C,)
            vis_feat_split = vis_feat.view(B, C, self.feature_dimension)   # (B,C,64)
            vis_feat = (vis_feat_split * cam_w.view(1, C, 1)).reshape(B, -1)  # (B,64*C)

        # qpos 轻量归一化
        qpos_feat = self.qpos_ln(qpos)  # (B,8)

        # 训练期模态Dropout（不改维度，只置零）
        if (actions is not None) and self.training:
            r = torch.rand((), device=qpos.device)
            if r < self.moddrop_p:
                vis_feat = torch.zeros_like(vis_feat)
            elif r < 2 * self.moddrop_p:
                qpos_feat = torch.zeros_like(qpos_feat)

        # 分支全局权重（sigmoid标量），仍不改维度
        w_vis, w_q = torch.sigmoid(self.branch_logits)
        vis_feat = vis_feat * w_vis
        qpos_feat = qpos_feat * w_q

        # 仍然是“视觉拼接 + qpos”→ 与原实现同维度
        obs_cond = torch.cat([vis_feat, qpos_feat], dim=1)  # (B, 64*C + 8) == self.obs_dim
        return obs_cond

    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        if actions is not None:  # training time
            nets = self.nets

            # === 构造 obs_cond（增强融合，但维度不变） ===
            obs_cond = self._build_obs_cond(nets, qpos, image, actions)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device, dtype=actions.dtype)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()

            # forward diffusion
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # predict v (since prediction_type='v_prediction')
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            # target v = sqrt(alpha_cumprod)*eps - sqrt(1-alpha_cumprod)*x0
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=actions.device, dtype=actions.dtype)
            alpha_prod_t = alphas_cumprod[timesteps]  # (B,)
            alpha_t = torch.sqrt(alpha_prod_t).view(B, *([1] * (actions.ndim - 1)))   # (B,1,1)
            sigma_t = torch.sqrt(1.0 - alpha_prod_t).view(B, *([1] * (actions.ndim - 1)))  # (B,1,1)
            target = alpha_t * noise - sigma_t * actions

            # L2 loss with padding mask
            all_l2 = F.mse_loss(noise_pred, target, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {'l2_loss': loss, 'loss': loss}

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict

        else:  # inference time
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim

            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model

            # === 构造 obs_cond（增强融合，但维度不变） ===
            obs_cond = self._build_obs_cond(nets, qpos, image, actions=None)

            # initialize action from Gaussian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device, dtype=qpos.dtype
            )
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict v
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                # reverse step
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status




'''
class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        self.camera_names = args_override['camera_names']

        self.observation_horizon = args_override['observation_horizon'] ### TODO TODO TODO DO THIS
        self.action_horizon = args_override['action_horizon'] # apply chunk size
        self.prediction_horizon = args_override['prediction_horizon'] # chunk size
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.ema_power = args_override['ema_power']
        self.lr = args_override['lr']
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override['action_dim'] # 14 + 2
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 8 # camera features and proprio

        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': True, 'input_coord_conv': False}))
            pools.append(SpatialSoftmax(**{'input_shape': [512, 6, 10], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)
        
        backbones = replace_bn_with_gn(backbones) # TODO


        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim*self.observation_horizon
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler —— 改为 v_prediction，其它保持不变
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='v_prediction'
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        if actions is not None: # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                cam_image = normalize(cam_image)
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                #print(np.shape(cam_features))
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device, dtype=actions.dtype)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict v (since prediction_type='v_prediction')
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            
            # ====== 目标为 v：v = sqrt(alpha_cumprod)*eps - sqrt(1-alpha_cumprod)*x0 ======
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=actions.device, dtype=actions.dtype)
            alpha_prod_t = alphas_cumprod[timesteps]                        # (B,)
            alpha_t = torch.sqrt(alpha_prod_t).view(B, *([1] * (actions.ndim - 1)))   # (B,1,1)
            sigma_t = torch.sqrt(1.0 - alpha_prod_t).view(B, *([1] * (actions.ndim - 1)))  # (B,1,1)
            target = alpha_t * noise - sigma_t * actions

            # L2 loss
            all_l2 = F.mse_loss(noise_pred, target, reduction='none')
            #all_l2 = F.l1_loss(noise_pred, target, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict
        else: # inference time
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            
            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model
            
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                cam_image = normalize(cam_image)
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            #print([np.max(all_features[0].detach().cpu().numpy()), np.min(all_features[0].detach().cpu().numpy())])
            obs_cond = torch.cat(all_features + [qpos], dim=1)
            #print([np.max(qpos.detach().cpu().numpy()), np.min(qpos.detach().cpu().numpy())])

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device, dtype=qpos.dtype)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise (here: v)
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status

    
    



class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        #image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            #print(is_pad)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
    
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
