import os
import copy

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

from third_parties.lpips import LPIPS

from core.train import create_lr_updater
from core.data import create_dataloader
from core.utils.network_util import set_requires_grad
from core.utils.train_util import cpu_data_to_gpu, Timer
from core.utils.image_util import tile_images, to_8b_image
from core.nets import VitExtractor
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import math
from scipy.ndimage.morphology import distance_transform_edt
from configs import cfg
import clip
from torchvision import transforms

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch
    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs

def _unpack_depths(depths, patch_masks, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch

    patch_imgs = torch.zeros(patch_masks.shape).to(depths.device) # (N_patch, H, W)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = depths[div_indices[i]:div_indices[i+1]]

    return patch_imgs

def _unpack_alpha(alphas, patch_masks, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch

    patch_imgs = torch.zeros(patch_masks.shape).to(alphas.device) # (N_patch, H, W)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = alphas[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


class Trainer(object):
    def __init__(self, network, optimizer):
        print('\n********** Init Trainer ***********')

        network = network.cuda().deploy_mlps_to_secondary_gpus()
        self.network = network
        
        if cfg.train.with_vit:
            if cfg.train.use_huggingface:
                import transformers
                self.use_huggingface = True
                vit_model = cfg.train.get('vit_model', 'facebook/vit-mae-large')
                if 'mae' in vit_model:
                    self.vit = transformers.ViTMAEModel.from_pretrained(vit_model).cuda()
                elif 'CLIP' in vit_model or 'clip' in vit_model:
                    self.vit = transformers.CLIPVisionModelWithProjection.from_pretrained(vit_model).cuda()
                print('use vit model:', vit_model)
                self.vit.eval()
                set_requires_grad(self.vit, requires_grad=False)
            else:
                self.use_huggingface = False
                vit = VitExtractor(model_name=cfg.train.get('vit_model', 'dino_vits16'), device='cuda')
                self.vit = vit.cuda()
                self.vit.eval()
                set_requires_grad(self.vit, requires_grad=False)
        if cfg.train.with_clip:
            clip_model = cfg.train.get('clip_model', 'ViT-L/14')
            print('clip_model', clip_model)
            self.perceptor, preprocess = clip.load(clip_model, jit=False, download_root='clip_ckpts', device=torch.device("cpu"))
            self.perceptor = self.perceptor.eval().requires_grad_(False).cuda()
            self.clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            self.clip_transform = transforms.Compose([
                transforms.Resize((cfg.train.vit_resolution, cfg.train.vit_resolution)),
                self.clip_normalizer
            ])

        self.optimizer = optimizer
        self.update_lr = create_lr_updater()

        if cfg.pretrained != '':
            self.load_pretrained(f'{cfg.pretrained}')

        if cfg.resume and Trainer.ckpt_exists(cfg.load_net):
            self.load_ckpt(f'{cfg.load_net}')
        else:
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1

        self.timer = Timer()

        if "lpips" in cfg.train.lossweights.keys():
            self.lpips = LPIPS(net='vgg')
            set_requires_grad(self.lpips, requires_grad=False)
            self.lpips = nn.DataParallel(self.lpips).cuda()

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(data_type='progress')
        self.log_dict = dict()

        print('************************************')


    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.tar')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################3
    ## Training 

    def get_vit_feature(self, x):

        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).reshape(1, 3, 1, 1)
        x = F.interpolate(x, size=(cfg.train.vit_resolution, cfg.train.vit_resolution))
        x = (x - mean) / std
        if self.use_huggingface:
            outputs = self.vit(x)
            return outputs.image_embeds
        feature = self.vit.get_feature_from_input(x)[-1]
        return feature[:, 0, :]

    def get_clip_feature(self, x):
        x = self.clip_transform(x)
        feat = self.perceptor.encode_image(x)
        return feat


    def get_img_rebuild_loss(self, loss_names, rgb, target):
        losses = {}

        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "lpips" in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)

        return losses

    def get_loss(self, net_output, 
                 patch_masks, bgcolor, targets, div_indices, target_patch_masks, sample_novel_view, smpl_masks=None, is_front=None, yaw_delta=0., body_part=None, direction=None):

        lossweights = cfg.train.lossweights
        loss_names = list(lossweights.keys())
        rgb = net_output['rgb']
        alpha = net_output['alpha']
        if not sample_novel_view:
            losses = self.get_img_rebuild_loss(
                            loss_names, 
                            _unpack_imgs(rgb, patch_masks, bgcolor,
                                        targets, div_indices), 
                            targets)

        else:
            losses = dict()
            rgb_patches = _unpack_imgs(rgb, patch_masks, bgcolor,
                                        targets, div_indices)
            alpha_patches = _unpack_alpha(alpha, patch_masks, div_indices)

            if 'vit' in loss_names:
                feat_real = self.get_vit_feature(targets.permute(0, 3, 1, 2))
                feat_fake = self.get_vit_feature(rgb_patches.permute(0, 3, 1, 2))
                losses['vit'] = F.mse_loss(feat_real, feat_fake)
            if 'vit_cos' in loss_names:
                feat_real = self.get_vit_feature(targets.permute(0, 3, 1, 2))
                feat_fake = self.get_vit_feature(rgb_patches.permute(0, 3, 1, 2))
                cosine = torch.cosine_similarity(torch.mean(feat_real, dim=0), torch.mean(feat_fake, dim=0), dim=0)
                losses['vit_cos'] = 1.0 - cosine
            if 'clip' in loss_names:
                feat_real = self.get_clip_feature(targets.permute(0, 3, 1, 2))
                feat_fake = self.get_clip_feature(rgb_patches.permute(0, 3, 1, 2))
                cosine = torch.cosine_similarity(torch.mean(feat_real, dim=0), torch.mean(feat_fake, dim=0), dim=0)
                losses['clip'] = 1.0 - cosine
                    
                if cfg.train.get('yaw_delta_loss_weight', False):
                    losses['clip'] = losses['clip'] * math.cos(yaw_delta / 180. * math.pi / 2)
                    print('yaw_delta_loss_weight: {}'.format(math.cos(yaw_delta / 180. * math.pi / 2)))
                if 'clip_loss_warmup' in cfg.train:
                    if self.iter < cfg.train.clip_loss_warmup[0]:
                        losses['clip'] = losses['clip'] * 0
                    elif self.iter < cfg.train.clip_loss_warmup[1]:
                        losses['clip'] = losses['clip'] * (self.iter - cfg.train.clip_loss_warmup[0]) / (cfg.train.clip_loss_warmup[1] - cfg.train.clip_loss_warmup[0])
            if 'sil_l2' in loss_names:
                sil_l2loss = (smpl_masks - alpha_patches) ** 2
                if cfg.train.sil_loss_inside_only:
                    sil_l2loss = sil_l2loss * smpl_masks
                losses['sil_l2'] = sil_l2loss.mean()
            if 'sil_edge' in loss_names:
                kernel_size=7
                power=cfg.train.get('edt_power', 0.25)
                def compute_edge(x):
                    return F.max_pool2d(x, kernel_size, 1, kernel_size // 2) - x
                gt_edge = compute_edge(smpl_masks.unsqueeze(1)).cpu().numpy()
                pred_edge = compute_edge(alpha_patches.unsqueeze(1))
                edt = torch.tensor(distance_transform_edt(1 - (gt_edge > 0)) ** (power * 2), dtype=torch.float, device=pred_edge.device)
                if cfg.train.sil_loss_inside_only:
                    pred_edge = pred_edge * smpl_masks.unsqueeze(1)
                sil_edgeloss = torch.sum(pred_edge * edt) / (pred_edge.sum()+1e-7)
                losses['sil_edge'] = sil_edgeloss


        train_losses = [
            weight * losses[k] for k, weight in lossweights.items() if k in losses
        ]
        loss_names = [k for k in loss_names if k in losses]
        return {loss_names[i]: train_losses[i] for i in range(len(loss_names))}

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1
        self.network.train()
        cfg.perturb = cfg.train.perturb


    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        self.train_begin(train_dataloader=train_dataloader)
        self.timer.begin()
        data_time = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            data_gpu_time = time.time()
            if self.iter > cfg.train.maxiter:
                break
            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            if data.get('render_for_reference', False):
                with torch.no_grad():
                    batch_ref = dict()
                    for k in batch:
                        batch_ref[k] = batch[k]
                    batch_ref['ray_mask'] = batch_ref['ray_mask_ref']
                    batch_ref['rays'] = batch_ref['rays_ref']
                    batch_ref['near'] = batch_ref['near_ref']
                    batch_ref['far'] = batch_ref['far_ref']
                    batch_ref['patch_div_indices'] = batch_ref['patch_div_indices_ref']
                    data.pop('ray_mask_ref')
                    data.pop('rays_ref')
                    data.pop('far_ref')
                    data.pop('patch_div_indices_ref')
                    data_ref = cpu_data_to_gpu(batch_ref, exclude_keys=EXCLUDE_KEYS_TO_GPU)
                    data_ref.pop('ray_mask_ref')
                    data_ref.pop('rays_ref')
                    data_ref.pop('far_ref')
                    data_ref.pop('patch_div_indices_ref')
                    self.network.eval()
                    net_output_ref = self.network(**data_ref)
                    self.network.train()
                rgb_ref = net_output_ref['rgb'].detach()
                alpha_ref = net_output_ref['alpha'].detach()
                rgb_patches = _unpack_imgs(rgb_ref, data_ref['target_patch_masks'], data['bgcolor']/255,
                                            data_ref['target_patches'], data_ref['patch_div_indices'])
                alpha_patches = _unpack_alpha(alpha_ref, data_ref['target_patch_masks'], data_ref['patch_div_indices'])
                data['target_patch_masks'] = (alpha_patches > 1e-3)
                data['target_patches'] = rgb_patches
            net_output = self.network(**data)
            self.optimizer.zero_grad()
            train_loss = 0
            loss_dict = self.get_loss(
                net_output=net_output,
                patch_masks=data['patch_masks'],
                bgcolor=data['bgcolor'] / 255.,
                targets=data['target_patches'],
                div_indices=data['patch_div_indices'],
                target_patch_masks=data.get('target_patch_masks', None),
                sample_novel_view=data.get('sample_novel_view', False),
                smpl_masks=data.get('smpl_masks', None),
                is_front=data.get('is_front', None),
                yaw_delta=data.get('yaw_delta', 0.),
                body_part=data.get('body_part', None),
                direction=data.get('direction', None)
                )
            for name, loss in loss_dict.items():
                train_loss = train_loss + loss
            train_loss.backward()
            self.optimizer.step()
            for k, v in loss_dict.items():
                if k in self.log_dict:
                    self.log_dict[k].append(v.item())
                else:
                    self.log_dict[k] = [v.item()]
            if self.iter % cfg.train.log_interval == 0:
                loss_str = f"Loss: {train_loss.item():.4f} ["
                for k, v in self.log_dict.items():
                    loss_str += f"{k}: {sum(v)/len(v):.4f} "
                self.log_dict = dict()
                loss_str += "]"

                log_str = 'Epoch: {} [Iter {}, {}/{} ({:.0f}%), {}] {}'
                log_str = log_str.format(
                    epoch, self.iter,
                    batch_idx * cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str)
                print(log_str)

            is_reload_model = False
            if self.iter in [100, 300, 1000, 2500] or \
                self.iter % cfg.progress.dump_interval == 0:
                is_reload_model = self.progress()

            if not is_reload_model:
                if self.iter % cfg.train.save_checkpt_interval == 0:
                    self.save_ckpt('latest')

                if cfg.save_all:
                    if self.iter % cfg.train.save_model_interval == 0:
                        self.save_ckpt(f'iter_{self.iter}')

                self.update_lr(self.optimizer, self.iter)

                self.iter += 1
    
    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.

    def progress_end(self):
        self.network.train()
        cfg.perturb = cfg.train.perturb

    def progress(self):
        self.progress_begin()

        print('Evaluate Progress Images ...')

        images = []
        is_empty_img = False
        for _, batch in enumerate(tqdm(self.prog_dataloader)):

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.network(**data)

            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs

            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            images.append(np.concatenate([rendered, truth], axis=1))

             # check if we create empty images (only at the begining of training)
            if self.iter <= 5000 and \
                np.allclose(rendered, np.array(cfg.bgcolor), atol=5.):
                is_empty_img = True
                break

        tiled_image = tile_images(images)
        
        Image.fromarray(tiled_image).save(
            os.path.join(cfg.logdir, "prog_{:06}.jpg".format(self.iter)))

        if is_empty_img:
            print("Produce empty images; reload the init model.")
            self.load_ckpt('init')
            
        self.progress_end()

        return is_empty_img


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Save checkpoint to {path} ...")

        torch.save({
            'iter': self.iter,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_pretrained(self, path):
        print(f"Load pretrained model from {path} ...")
        ckpt = torch.load(path, map_location='cuda:0')
        if cfg.train.get('init_cnl_mlp_rgb', False):
            print('init_cnl_mlp_rgb')
            mlp_weights = self.network.cnl_mlp.module.output_linear[0].weight
            mlp_bias = self.network.cnl_mlp.module.output_linear[0].bias
            ckpt['network']['cnl_mlp.module.output_linear.0.weight'][:3] = mlp_weights[:3]
            ckpt['network']['cnl_mlp.module.output_linear.0.bias'][:3] = mlp_bias[:3]
        self.network.load_state_dict(ckpt['network'], strict=False)

    def load_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path)
        self.iter = ckpt['iter'] + 1

        self.network.load_state_dict(ckpt['network'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])
