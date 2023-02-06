import os
import pickle

import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.utils.data
import random
from easydict import EasyDict as edict
from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes, \
    get_body_direction
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox, \
    _update_extrinsics

from configs import cfg
from third_parties.smpl.smpl_numpy import SMPL
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
    }
    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            start_frame=0,
            src_type="zju_mocap",
            **_):

        print('[Dataset Path]', dataset_path) 
        self.src_type = src_type
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')

        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()

        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints,   
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')

        framelist = self.load_train_frames()
        if 'end_frame' in cfg.train:
            framelist = framelist[start_frame: cfg.train.end_frame+1]
        else:
            framelist = framelist[start_frame:]
        self.framelist = framelist[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
        self.frame_id_set = list(set([f.split('_')[1] for f in framelist]))
        self.latent_index_map = {f: i for i, f in enumerate(self.frame_id_set)}
        if 'single_frame_id' in cfg.train:
            print('framelist:', self.framelist)
            self.single_frame_id = self.framelist.index(cfg.train.single_frame_id)
            print('self.single_frame_id', self.single_frame_id)
        else:
            self.single_frame_id = -1
        print(f' -- Total Frames: {self.get_total_frames()}')
        self.cameras = self.load_train_cameras()
        self.mesh_infos = self.load_train_mesh_infos()
        if cfg.task in ['mydemo', 'fashion']:
            self.smpl_dataset_path = dataset_path
            self.smpl_cameras = self.load_smpl_train_cameras()
            self.smpl_image_dir = os.path.join(self.smpl_dataset_path, 'images')
            self.smpl_image_dir = os.path.join(self.smpl_dataset_path, 'masks')
        else:
            if cfg.train.use_smpl_data:
                self.smpl_dataset_path = dataset_path + '_smpl'
                self.smpl_cameras = self.load_smpl_train_cameras()
                self.smpl_image_dir = os.path.join(self.smpl_dataset_path, 'images')
                self.smpl_image_dir = os.path.join(self.smpl_dataset_path, 'masks')
                assert 'smpl_masks' in keyfilter

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.ray_shoot_mode = ray_shoot_mode
        self.sample_novel_ratio = cfg.train.sample_novel_ratio
        self.num_smpl_cameras = cfg.train.get('num_smpl_cameras', -1)
        self.use_smpl_data = cfg.train.get('use_smpl_data', False)
        self.patch_dict_config = cfg.train.get('patch_dict_config', None)
        self.patch_cfg = cfg.patch
        if 'train_patch_cfgs' in cfg.train:
            assert isinstance(cfg.train.train_patch_cfgs, list)
            self.train_patch_cfgs = cfg.train.train_patch_cfgs
        else:
            self.train_patch_cfgs = [self.patch_cfg]
        self.sample_cams = cfg.train.get('sample_cams', None)
        self.textureless_ratio = cfg.train.get('textureless_ratio', 0.)
        self.sample_directions = cfg.train.get('sample_directions', None)
        if self.textureless_ratio > 0:
            assert cfg.use_normal_map
        self.num_frame = len(self.framelist)
        self.text_prompts = cfg.train.get('text_prompts', None)

        smpl_model = SMPL(sex='neutral', model_dir='third_parties/smpl/models/')
        self.smpl_weights = smpl_model.weights
       

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        cameras = None
        if cfg.task in ['mydemo', 'fashion']:
            data = np.load(os.path.join(self.dataset_path, 'gt_params.npz'))
            if 'focal_l' in data:
                focal = data['focal_l']
                K = np.eye(3, dtype='float32')
                K[0, 0] = focal
                K[1, 1] = focal
                img_size = Image.open(os.path.join(self.dataset_path, 'gt_image.png')).size[0]
                K[:2, 2] = img_size / 2.
                E = np.eye(4)
            else:
                K = data['intrinsics']
                E = data['extrinsics']
            cameras = {f: dict(intrinsics=K, extrinsics=E) for f in self.framelist}
        elif cfg.train.use_input_annot:
            with open(os.path.join(self.dataset_path, 'input_camera.pkl'), 'rb') as f:
                camera = pickle.load(f) 
            cameras = {f: camera for f in self.framelist}
        else:
            with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
                cameras = pickle.load(f)
        return cameras

    def load_smpl_train_cameras(self):
        cameras = None
        with open(os.path.join(self.smpl_dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox
        if cfg.task in ['mydemo', 'fashion']:
            data = np.load(os.path.join(self.dataset_path, 'gt_params.npz'))
            poses = data['pose'][0]
            betas = data['shape'][0]
            Th = data['global_t'][0]
            if 'global_r' in data:
                Rh = data['global_r'][0]
            else:
                Rh = np.array([0., 0., 0.])
            smpl_model = SMPL(sex='neutral', model_dir='third_parties/smpl/models/')
            vertices, joints = smpl_model(poses, betas)
            tpose_vertices, tpose_joints = smpl_model(np.zeros_like(poses), betas)
            demo_mesh_info = dict(
                poses=poses,
                betas=betas,
                tpose_joints=tpose_joints,
                tpose_vertices=tpose_vertices,
                bbox=bbox,
                Rh=Rh,
                Th=Th,
                joints=joints,
                vertices=vertices,
            )
            mesh_infos.update({'gt': demo_mesh_info})
        elif cfg.train.use_input_annot:
            with open(os.path.join(self.dataset_path, 'input_mesh_info.pkl'), 'rb') as f:
                annot = pickle.load(f)
            annot['bbox'] = self.skeleton_to_bbox(annot['joints'])
            mesh_infos.update({'gt': anno})

        return mesh_infos

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        framelist = [split_path(ipath)[1] for ipath in img_paths]
        if cfg.task in ['mydemo', 'fashion']:
            framelist = list(set(['_'.join(f.split('_')[:-1]) for f in framelist]))
        return framelist
    
    def query_dst_skeleton(self, frame_name, use_smpl=False):
        mesh_info = None
        if cfg.task in ['mydemo', 'fashion']:
            if use_smpl:
                mesh_info = self.mesh_infos[frame_name + '_00']
            else:
                mesh_info = self.mesh_infos['gt']
        else:
            if (not use_smpl) and cfg.train.use_input_annot:
                mesh_info = self.mesh_infos['gt']
            else:
                mesh_info = self.mesh_infos[frame_name]
        ret = {
            'poses': mesh_info['poses'].astype('float32'),
            'dst_tpose_joints': \
                mesh_info['tpose_joints'].astype('float32'),
            'bbox': mesh_info['bbox'].copy(),
            'Rh': mesh_info['Rh'].astype('float32'),
            'Th': mesh_info['Th'].astype('float32')
        }

        return ret

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, near, far

    @staticmethod
    def resize_by_mask(mask, K, target_H, target_W, img=None, padding=0, bg_color=0, ref_mask=None):
        if ref_mask is None:
            ref_mask = mask
        H, W = ref_mask.shape
        ys, xs = np.where(ref_mask > 0)
        min_x, max_x = xs.min(), xs.max()+1
        min_y, max_y = ys.min(), ys.max()+1
        if isinstance(padding, float):
            padding = np.round(padding * max(max_y - min_y, max_x-min_x)).astype(np.int)
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        if max_x - min_x > max_y - min_y:
            min_y = min_y + ((max_y - min_y) - (max_x - min_x)) // 2
            max_y = min_y + max_x - min_x
        if max_y - min_y > max_x - min_x:
            min_x = min_x + ((max_x - min_x) - (max_y - min_y)) // 2
            max_x = min_x + max_y - min_y
        trans = np.eye(3)
        trans[2, 0] = -min_x
        trans[2, 1] = -min_y
        scale = np.eye(3)
        scale[0, 0] = target_W / (max_x - min_x)
        scale[1, 1] = target_H / (max_y - min_y)

        K = (K.T @ trans @ scale).T
        lp = max(0, -min_x)
        rp = max(0, W - max_x)
        up = max(0, -min_y)
        dp = max(0, H - max_y)
        min_x += lp
        max_x += lp
        min_y += up
        max_y += up
        mask = np.pad(mask, ((up, dp), (lp, rp)), 'constant')
        mask = mask[min_y:max_y, min_x:max_x]
        mask = cv2.resize(mask[:, :, None].astype(np.float), (target_H, target_W), interpolation=cv2.INTER_NEAREST)
        mask = mask[:, :] > 0.
        if img is not None:
            img = np.pad(img, ((up, dp), (lp, rp), (0, 0)), mode='edge')
            img = img[min_y:max_y, min_x:max_x]
            img = cv2.resize(img, (target_H, target_W))

        return mask, K, img
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W,
            subject_only=False,
            bbox_only=False,
            sH=None, sW=None, get_subject_mask=False):

        assert subject_mask.dtype == np.bool
        assert bbox_mask.dtype == np.bool

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            # let p = self.patch_cfg.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if subject_only:
                candidate_mask = subject_mask
            elif bbox_only:
                candidate_mask = bbox_mask
            elif np.random.rand(1)[0] < self.patch_cfg.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            if sH is not None and sW is not None:
                ray_indices, mask, xy_min, xy_max = \
                self._get_patch_ray_indices_stride(ray_mask, candidate_mask, 
                                            patch_size, H, W, sH, sW, get_subject_mask)
                assert len(ray_indices) > 0

            else:    
                ray_indices, mask, xy_min, xy_max = \
                    self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                                patch_size, H, W)
                assert len(ray_indices) > 0
            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)

            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        if sH is not None and sW is not None:
            patch_info.update({
                'stride': np.array([sW, sH])
            })
        patch_div_indices = np.array(patch_div_indices)
        assert len(select_inds) > 0
        return select_inds, patch_info, patch_div_indices

    
    def _get_patch_ray_indices_stride(
        self,
        ray_mask,
        candidate_mask, 
        patch_size, 
        H, W,
        sH, sW,
        get_subject_mask
    ):
        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

        valid_ys, valid_xs = np.where(candidate_mask)

        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        x_min = np.clip(a=center_x - (patch_size[0] // 2) * sW,
                        a_min=0, a_max=W-patch_size[0] * sW)
        y_min = np.clip(a=center_y - (patch_size[1] // 2) * sH,
                        a_min=0, a_max=H-patch_size[1] * sH)

        x_max = x_min + (patch_size[0] - 1) * sW + 1
        y_max = y_min + (patch_size[1] - 1) * sH + 1

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max:sH, x_min:x_max:sW] = True
        sel_ray_mask = sel_ray_mask.reshape(-1)
        if get_subject_mask:
            inter_mask = np.bitwise_and(sel_ray_mask, candidate_mask.reshape(-1))
        else:
            inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]

        inter_mask = inter_mask.reshape(H, W)
        assert len(select_inds) > 0 and inter_mask[y_min:y_max:sH, x_min:x_max:sW].sum() > 0, "{}, {}".format(center_x, center_y)
        return select_inds, \
                inter_mask[y_min:y_max:sH, x_min:x_max:sW], \
                np.array([x_min, y_min]), np.array([x_max, y_max])


    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = tuple([ps // 2 for ps in patch_size])
        x_min = np.clip(a=center_x-half_patch_size[0], 
                        a_min=0, 
                        a_max=W-patch_size[0])
        x_max = x_min + patch_size[0]
        y_min = np.clip(a=center_y-half_patch_size[1],
                        a_min=0,
                        a_max=H-patch_size[1])
        y_max = y_min + patch_size[1]

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max])
    
    def load_image(self, frame_name, bg_color, resize=True):
        if cfg.task in ['mydemo', 'fashion'] or cfg.train.use_input_annot:
            imagepath = os.path.join(self.dataset_path, 'gt_image.png')
            maskpath = os.path.join(self.dataset_path, 'gt_mask.png')
        else:
            imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
            maskpath = os.path.join(self.dataset_path, 
                                    'masks', 
                                    '{}.png'.format(frame_name))        

        orig_img = np.array(load_image(imagepath))
        alpha_mask = np.array(load_image(maskpath))
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1. and resize:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask

    def load_segment(self, file_path, resize=True):
        segment_img = np.array(Image.open(file_path))
        segments = dict()
        for seg_key, seg_color in cfg.segments.items():
            seg_map = (segment_img[:, :, 0] == seg_color).astype(np.float)
            if cfg.resize_img_scale != 1. and (not cfg.train.resize_to_patchsize_by_mask):
                seg_map = cv2.resize(seg_map[:, :, None], None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_NEAREST)
            segments[seg_key] = seg_map
        segments['full'] = (segment_img[:, :, 0] > 0).astype(np.float)
        return segments

    def get_total_frames(self):
        return cfg.train.maxiter
        #return len(self.framelist)

    def sample_patch_rays(self, img, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, near, far):
        patch_size = self.patch_cfg.size
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=self.patch_cfg.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=patch_size, 
                H=H, W=W)
        assert len(select_inds) > 0

        rays_o, rays_d, ray_img, near, far = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, near, far)
        
        targets = []
        for i in range(self.patch_cfg.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, near, far, \
                target_patches, patch_masks, patch_div_indices

    def sample_patch_rays_novel(self, img, H, W,
                          subject_mask, bbox_mask, ray_mask, ray_mask_ref,
                          rays_o, rays_o_ref, rays_d, rays_d_ref, ray_img, ray_img_ref, near, near_ref, far, far_ref, smpl_mask=None, render_for_reference=False):

        sH = self.patch_cfg.get('sH', None)
        sW = self.patch_cfg.get('sW', None)
        patch_size = self.patch_cfg.size
        if 'size_h' in self.patch_cfg and 'size_w' in self.patch_cfg:
            patch_size = (self.patch_cfg.size_w, self.patch_cfg.size_h)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        select_inds_ref, patch_info_ref, patch_div_indices_ref = \
            self.get_patch_ray_indices(
                N_patch=self.patch_cfg.N_patches, 
                ray_mask=ray_mask_ref, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=patch_size, 
                H=H, W=W, subject_only=not render_for_reference, sW=sW, sH=sH, get_subject_mask=not render_for_reference)

        select_inds_novel, patch_info_novel, patch_div_indices_novel = \
            self.get_patch_ray_indices(
                N_patch=self.patch_cfg.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=patch_size, 
                H=H, W=W, bbox_only=True, sW=sW, sH=sH)

        rays_o, rays_d, ray_img, near, far = self.select_rays(
            select_inds_novel, rays_o, rays_d, ray_img, near, far)
        if render_for_reference:
            rays_o_ref, rays_d_ref, ray_img_ref, near_ref, far_ref = self.select_rays(
                select_inds_ref, rays_o_ref, rays_d_ref, ray_img_ref, near_ref, far_ref)
        else:
            rays_o_ref, rays_d_ref, ray_img_ref, near_ref, far_ref = rays_o, rays_d, ray_img, near, far
        
        targets = []
        for i in range(self.patch_cfg.N_patches):
            x_min, y_min = patch_info_ref['xy_min'][i] 
            x_max, y_max = patch_info_ref['xy_max'][i]
            targets.append(img[y_min:y_max:sH, x_min:x_max:sW])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info_novel['mask']  # boolean array (N_patches, P, P)
        target_patch_masks = patch_info_ref['mask']  # boolean array (N_patches, P, P)
        rets = [rays_o, rays_o_ref, rays_d, rays_d_ref, ray_img, ray_img_ref, near, near_ref, far, far_ref, \
                target_patches, patch_masks, target_patch_masks, patch_div_indices_novel, patch_div_indices_ref]
        if smpl_mask is not None:
            smpl_patch_masks = []
            for i in range(self.patch_cfg.N_patches):
                x_min, y_min = patch_info_novel['xy_min'][i] 
                x_max, y_max = patch_info_novel['xy_max'][i]
                smpl_patch_masks.append(smpl_mask[y_min:y_max:sH, x_min:x_max:sW])
            smpl_patch_masks = np.stack(smpl_patch_masks, axis=0) # (N_patches, P, P)
            rets.append(smpl_patch_masks)
        else:
            rets.append(None)
        return rets

    def __len__(self):
        return self.get_total_frames()

    def set_random_color(self):
        print('set random color!')
        self.bgcolor = (np.random.rand(3) * 255.).astype('float32')

    def sample_camera(self, frame_idx, Rh, Th, sample_directions):
        D = None
        if self.sample_cams != None:
            cam = random.choice(self.sample_cams)
        elif sample_directions != None:
            body_directions = []
            for cam in range(0, self.num_smpl_cameras):
                smpl_frame_name = '{}_{:02d}'.format(frame_idx, cam)
                E = self.smpl_cameras[smpl_frame_name]['extrinsics']
                direction, yaw = get_body_direction(E, Rh, Th, inv=cfg.get('body_direction_inv', False))
                body_directions.append((direction, yaw))
            D = random.choices(population=list(sample_directions.keys()), weights=list(sample_directions.values()), k=1)[0]
            candidates = [(i, yaw) for i, (d, yaw) in enumerate(body_directions) if d == D]
            cam, yaw = random.choice(candidates)
        else:
            cam = random.choice(range(0, self.num_smpl_cameras))
        return cam, D

    def sample_nearest_camera(self, frame_idx, Rh, Th, sample_directions, sampled_camera):
        body_directions = []
        for cam in range(0, self.num_smpl_cameras):
            smpl_frame_name = '{}_{:02d}'.format(frame_idx, cam)
            E = self.smpl_cameras[smpl_frame_name]['extrinsics']
            body_directions.append(get_body_direction(E, Rh, Th, inv=cfg.get('body_direction_inv', False)))
        sampled_yaw = body_directions[sampled_camera][1]
        min_diff = 180
        res = 0
        for i, (d, yaw) in enumerate(body_directions):
            if d in sample_directions:
                diff = (yaw - sampled_yaw + 360) % 360
                diff = min(360 - diff, diff)
                if diff < min_diff:
                    res = i
                    min_diff = diff
        return res

    def __getitem__(self, idx):
        idx = idx % self.num_frame
        self.patch_cfg = edict(random.choice(self.train_patch_cfgs))
        frame_name = self.framelist[idx]
        if self.single_frame_id != -1:
            frame_name = self.framelist[self.single_frame_id]
        results = {
            'frame_name': frame_name,
        }

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        sample_novel_view = random.random() < self.sample_novel_ratio
        img, alpha = self.load_image(frame_name, bgcolor, resize=not (sample_novel_view and cfg.train.resize_to_patchsize_by_mask))
        img = (img / 255.).astype('float32')
        alpha_mask = alpha[:, :, 0] > 0.

        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        if not (sample_novel_view and cfg.train.resize_to_patchsize_by_mask):
            K[:2] *= cfg.resize_img_scale

        E = self.cameras[frame_name]['extrinsics']
        body_part = 'none'
        cam_direction = 'none'
        render_for_reference = False
        if sample_novel_view:
            K_ref = K.copy()
            E_ref = E.copy()
            render_frame_name = self.framelist[idx]
            dst_skel_info = self.query_dst_skeleton(render_frame_name, use_smpl=True)
            dst_bbox = dst_skel_info['bbox']
            dst_poses = dst_skel_info['poses']
            dst_tpose_joints = dst_skel_info['dst_tpose_joints']
            body_part = 'full'
            assert self.num_smpl_cameras > 1
            if cfg.task in ['mydemo', 'fashion'] or cfg.train.use_input_annot:
                segments_ref = self.load_segment(os.path.join(self.dataset_path, 'gt_segments.png'))
            elif cfg.task in ['zju_mocap', 'thuman']:
                segments_ref = self.load_segment(os.path.join(self.smpl_dataset_path, 'segments', '{}_{:02d}.png'.format(frame_name, 0)))
            elif cfg.task in ['h36m']:
                segments_ref = self.load_segment(os.path.join(self.smpl_dataset_path, 'segments_view3', '{}_{:02d}.png'.format(frame_name, 0)))
            if cfg.train.get('sample_body_parts', None):
                assert cfg.train.resize_to_patchsize_by_mask
                sample_body_parts_cfg = cfg.train.get('sample_body_parts', None)
                probs = [sample_body_parts_cfg[k]['prob'] for k in sample_body_parts_cfg]
                #print(list(sample_body_parts_cfg.keys()), probs)
                body_part = random.choices(list(sample_body_parts_cfg.keys()), weights=probs, k=1)[0]
                #print(body_part)
                sample_directions = sample_body_parts_cfg[body_part].get('directions', None)
            else:
                body_part = 'full'
                sample_directions = self.sample_directions
            smpl_cam = -1
            sample_times = 0
            while smpl_cam < 0:
                smpl_cam, cam_direction = self.sample_camera(self.framelist[idx], Rh=dst_skel_info['Rh'], Th=dst_skel_info['Th'], sample_directions=sample_directions)
                smpl_frame_name = '{}_{:02d}'.format(self.framelist[idx], smpl_cam)
                segments_smpl = self.load_segment(os.path.join(self.smpl_dataset_path,'segments','{}.png'.format(smpl_frame_name)))
                smpl_maskpath = os.path.join(self.smpl_dataset_path, 
                                        'masks', 
                                        '{}.png'.format(smpl_frame_name))
                smpl_mask = np.array(load_image(smpl_maskpath))
                if segments_smpl[body_part].sum() < cfg.train.min_bodypart_pixels:
                    smpl_cam = -1
                    sample_times += 1
                if sample_times > 10:
                    body_part='full'
            if cfg.resize_img_scale != 1. and (not cfg.train.resize_to_patchsize_by_mask):
                smpl_mask = cv2.resize(smpl_mask, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LINEAR)
            smpl_mask = (smpl_mask[:, :, 0] > 0.).astype(np.float32)
            K = self.smpl_cameras[smpl_frame_name]['intrinsics'][:3, :3].copy()
            if not (sample_novel_view and cfg.train.resize_to_patchsize_by_mask):
                K[:2] *= cfg.resize_img_scale
            E = self.smpl_cameras[smpl_frame_name]['extrinsics']
            if cfg.train.get('render_for_reference', None):
                if body_part in cfg.train.render_for_reference and cam_direction in cfg.train.render_for_reference[body_part]:
                    render_for_reference = True
                    ref_cam = self.sample_nearest_camera(self.framelist[idx], Rh=dst_skel_info['Rh'], Th=dst_skel_info['Th'], sample_directions=cfg.train.render_for_reference[body_part][cam_direction], sampled_camera=smpl_cam)
                    ref_smpl_frame_name = '{}_{:02d}'.format(self.framelist[idx], ref_cam)
                    ref_smpl_maskpath = os.path.join(self.smpl_dataset_path, 
                                            'masks', 
                                            '{}.png'.format(ref_smpl_frame_name))
                    alpha_mask = np.array(load_image(ref_smpl_maskpath))
                    segments_ref = self.load_segment(os.path.join(self.smpl_dataset_path, 
                                        'segments', 
                                        '{}.png'.format(ref_smpl_frame_name)))
                    if cfg.resize_img_scale != 1. and (not cfg.train.resize_to_patchsize_by_mask):
                        alpha_mask = cv2.resize(alpha_mask, None, 
                                        fx=cfg.resize_img_scale,
                                        fy=cfg.resize_img_scale,
                                        interpolation=cv2.INTER_LINEAR)
                    alpha_mask = alpha_mask[:, :, 0] > 0
                    K_ref = self.smpl_cameras[ref_smpl_frame_name]['intrinsics'][:3, :3].copy()
                    if not (sample_novel_view and cfg.train.resize_to_patchsize_by_mask):
                        K_ref[:2] *= cfg.resize_img_scale
                    E_ref = self.smpl_cameras[ref_smpl_frame_name]['extrinsics']
                    #print(smpl_frame_name, ref_smpl_frame_name)
                if segments_ref[body_part].sum()  < cfg.train.min_bodypart_pixels:
                    body_part = 'full'
            E_ref = apply_global_tfm_to_camera(
                    E=E_ref, 
                    Rh=dst_skel_info['Rh'],
                    Th=dst_skel_info['Th'])
            R_ref = E_ref[:3, :3]
            T_ref = E_ref[:3, 3]
        results.update({'sample_novel_view': sample_novel_view, 'body_part': body_part, 'direction': cam_direction,'render_for_reference': render_for_reference})
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]
        yaw = np.arctan2(R[0, 2], R[2, 2])/np.pi * 180
        is_front = not (yaw < 90 and yaw >= -90)
        if sample_novel_view:
            yaw_ref = np.arctan2(R_ref[0, 2], R_ref[2, 2])/np.pi * 180
            yaw_delta = (yaw - yaw_ref + 360) % 360
            yaw_delta = min(yaw_delta, 360 - yaw_delta)
        else:
            yaw_delta = 0.
        results.update({'yaw_delta': yaw_delta})
        results.update({'is_front': is_front})


        if sample_novel_view:
            if body_part == 'full' and cfg.train.resize_to_patchsize_by_mask:
                padding = 0.05
                alpha_mask, K_ref, img = self.resize_by_mask(alpha_mask, K_ref, cfg.patch.size, cfg.patch.size, padding=padding, img=img, bg_color=bgcolor)
                smpl_mask, K, _ = self.resize_by_mask(smpl_mask, K, cfg.patch.size, cfg.patch.size, padding=padding, bg_color=bgcolor)
                smpl_mask = smpl_mask.astype(np.float)
                H, W = cfg.patch.size, cfg.patch.size
            else:
                padding = 0.05
                if body_part == 'head':
                    padding = 0.5
                if body_part in cfg.symmetric and random.random() > cfg.train.get('flip_ratio', 1.0) and segments_smpl[cfg.symmetric[body_part]].sum() >= cfg.train.min_bodypart_pixels:
                    alpha_mask, K_ref, img = self.resize_by_mask(alpha_mask, K_ref, cfg.patch.size, cfg.patch.size, padding=padding, img=img, bg_color=bgcolor, ref_mask=segments_ref[cfg.symmetric[body_part]])
                    alpha_mask = alpha_mask[:, ::-1]
                    img = img[:, ::-1]
                else:
                    alpha_mask, K_ref, img = self.resize_by_mask(alpha_mask, K_ref, cfg.patch.size, cfg.patch.size, padding=padding, img=img, bg_color=bgcolor, ref_mask=segments_ref[body_part])
                smpl_mask, K, _ = self.resize_by_mask(smpl_mask, K, cfg.patch.size, cfg.patch.size, padding=padding, bg_color=bgcolor, ref_mask=segments_smpl[body_part])
                assert (alpha_mask > 0).sum() > 0, f"{frame_name}, {body_part}"
                assert (smpl_mask > 0).sum() > 0 
                smpl_mask = smpl_mask.astype(np.float)
                H, W = cfg.patch.size, cfg.patch.size
            

            rays_o_ref, rays_d_ref = get_rays_from_KRT(H, W, K_ref, R_ref, T_ref)
            rays_o_ref = rays_o_ref.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
            rays_d_ref = rays_d_ref.reshape(-1, 3)
            near_ref, far_ref, ray_mask_ref = \
                rays_intersect_3d_bbox(dst_bbox, rays_o_ref, rays_d_ref)
            rays_o_ref = rays_o_ref[ray_mask_ref]
            rays_d_ref = rays_d_ref[ray_mask_ref]
            ray_img_ref = img.reshape(-1, 3)
            ray_img_ref = ray_img_ref[ray_mask_ref]

            near_ref = near_ref[:, None].astype('float32')
            far_ref = far_ref[:, None].astype('float32')
            

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            if sample_novel_view:
                rays_o, rays_o_ref, rays_d, rays_d_ref, ray_img, _, near, near_ref, far, far_ref, \
                target_patches, patch_masks, target_patch_masks, patch_div_indices, patch_div_indices_ref, smpl_patch_masks = \
                    self.sample_patch_rays_novel(img=img, H=H, W=W,
                                        subject_mask=alpha_mask,
                                        bbox_mask=ray_mask.reshape(H, W),
                                        ray_mask=ray_mask,
                                        ray_mask_ref=ray_mask_ref,
                                        rays_o=rays_o, 
                                        rays_o_ref=rays_o_ref,
                                        rays_d=rays_d, 
                                        rays_d_ref=rays_d_ref,
                                        ray_img=ray_img, 
                                        ray_img_ref=ray_img_ref,
                                        near=near, 
                                        near_ref=near_ref,
                                        far=far,
                                        far_ref=far_ref,
                                        smpl_mask=smpl_mask,
                                        render_for_reference=render_for_reference)
                batch_rays_ref = np.stack([rays_o_ref, rays_d_ref], axis=0) 
            else:
                rays_o, rays_d, ray_img, near, far, \
                target_patches, patch_masks, patch_div_indices = \
                    self.sample_patch_rays(img=img, H=H, W=W,
                                        subject_mask=alpha_mask,
                                        bbox_mask=ray_mask.reshape(H, W),
                                        ray_mask=ray_mask,
                                        rays_o=rays_o, 
                                        rays_d=rays_d, 
                                        ray_img=ray_img, 
                                        near=near, 
                                        far=far)
                smpl_patch_masks = None
        else:
            assert False, f"Invalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 
        #print('batch_rays.shape', batch_rays.shape)

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})
            if sample_novel_view and render_for_reference:
                results.update({
                    'ray_mask_ref': ray_mask_ref,
                    'rays_ref': batch_rays_ref,
                    'near_ref': near_ref,
                    'far_ref': far_ref,
                })

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'target_patches': target_patches})
                if sample_novel_view:
                    results.update({
                        'target_ray_mask': ray_mask_ref,
                        'target_patch_masks': target_patch_masks
                    })
                    if render_for_reference:
                        results.update({
                            'patch_div_indices_ref': patch_div_indices_ref
                        })

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img
        
        if 'smpl_masks' in self.keyfilter and smpl_patch_masks is not None:
            results['smpl_masks'] = smpl_patch_masks

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            cnl_gtfms = get_canonical_global_tfms(
                            self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms,
                'dst_poses': dst_poses
            })
            
        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })

        #print(results['target_patches'])
        #print(results['target_patches'].shape)
        for k, v in results.items():
            assert v is not None, f'{k} is None!'
        
        return results
