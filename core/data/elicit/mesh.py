import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data
from PIL import Image

from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_projection_depth
from core.utils.file_util import list_files, split_path

from configs import cfg
from third_parties.smpl.smpl_numpy import SMPL


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    if os.path.exists(msk_path):
        msk = np.array(load_image(msk_path))[:, :, 0]
        msk = (msk != 0).astype(np.uint8)
    else:
        msk = None

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    if os.path.exists(msk_path):
        msk_cihp = np.array(load_image(msk_path))[:, :, 0]
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
    else:
        msk_cihp = None
    if msk is None:
        msk = msk_cihp
    if msk_cihp is None:
        msk_cihp = msk
    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


class Dataset(torch.utils.data.Dataset):
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'h36m': {'rotate_axis': 'z', 'inv_angle': True},
        'thuman': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
    }

    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            skip=1,
            bgcolor=None,
            src_type="zju_mocap",
            **_):

        print('[Dataset Path]', dataset_path) 

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
        self.framelist = framelist[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]  

        frame_id_set = list(set([f.split('_')[1] for f in framelist]))
        self.latent_index_map = {f: i for i, f in enumerate(frame_id_set)}

        cameras = self.load_train_cameras()
        mesh_infos = self.load_train_mesh_infos()

        self.train_frame_idx = framelist.index(cfg.mesh.frame_name)
        print(f' -- Frame Name: {cfg.mesh.frame_name}, Frame Idx: {self.train_frame_idx}')
        self.total_frames = 1

        self.train_frame_name = framelist[self.train_frame_idx]
        self.train_camera = cameras[framelist[self.train_frame_idx]]
        self.train_mesh_info = mesh_infos[framelist[self.train_frame_idx]]
        self.train_latent_index = self._get_frame_latent_index(self.train_frame_name)
        if cfg.freeview.get('use_gt_camera', False):
            self.use_gt_camera = True
            gt_cameras = self.load_gt_cameras()
            self.total_frames = len(gt_cameras)
            self.gt_cameras = gt_cameras
            self.camera_frames = sorted(list(gt_cameras.keys()))
            annots = np.load(os.path.join(cfg.source_data_dir, 'annots.npy'), allow_pickle=True).item()
            self.gt_views = annots['ims'][int(self.train_frame_name.split('_')[-1])]['ims']
        else:
            self.use_gt_camera = False
        print(f' -- Total Rendered Frames: {self.total_frames}')

        self.bgcolor = bgcolor if bgcolor is not None else [0., 0., 0.]
        self.keyfilter = keyfilter
        self.src_type = src_type

        
    @staticmethod
    def _load_bigpose():
        big_poses = np.zeros([24, 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1)
        return big_poses

    def _get_frame_latent_index(self, frame_name):
        if cfg.frame_latent_index >= 0:
            return cfg.frame_latent_index
        if not cfg.use_pdf_data:
            return 0
        if frame_name == 'gt':
            idx = int(self.latent_index_map[frame_name])
        else:
            idx = int(self.latent_index_map[(frame_name.split('_')[1])])
        if len(self.latent_index_map) > cfg.num_latent_code:
            idx = int(idx / len(self.latent_index_map) * cfg.num_latent_code)
        return idx

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        if cfg.task in ['mydemo', 'fashion'] and cfg.mesh.get('render_gt_view', True):
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
        if cfg.task in ['mydemo', 'fashion'] and cfg.mesh.get('render_gt_view', True):
            data = np.load(os.path.join(self.dataset_path, 'gt_params.npz'))
            poses = data['pose'][0]
            betas = data['shape'][0]
            Th = data['global_t'][0]
            Rh = np.array([0., 0., 0.])
            smpl_model = SMPL(sex='neutral', model_dir='third_parties/smpl/models/')
            vertices, joints = smpl_model(poses, betas)
            tpose_vertices, tpose_joints = smpl_model(np.zeros_like(poses), betas)
            bbox = self.skeleton_to_bbox(joints)
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
            for k in mesh_infos:
                mesh_infos[k] = demo_mesh_info
        return mesh_infos

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self):
        ret = {
            'poses': self.train_mesh_info['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.train_mesh_info['tpose_joints'].astype('float32'),
            'bbox': self.train_mesh_info['bbox'].copy(),
            'Rh': self.train_mesh_info['Rh'].astype('float32'),
            'Th': self.train_mesh_info['Th'].astype('float32')
        }
        return ret

    def get_camera(self):
        E = self.train_camera['extrinsics'].copy()
        K = self.train_camera['intrinsics'].copy()
        K[:2] *= cfg.resize_img_scale
        return K, E

    def load_image(self, frame_name, bg_color):
        if cfg.mesh.get('render_gt_view', True):
            imagepath = os.path.join(self.dataset_path, 'gt_image.png')
            maskpath = os.path.join(self.dataset_path, 'gt_mask.png')
        else:
            imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
            maskpath = os.path.join(self.dataset_path, 
                                    'masks', 
                                    '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        alpha_mask = np.array(load_image(maskpath))
        
        if 'distortions' in self.train_camera:
            K = self.train_camera['intrinsics']
            D = self.train_camera['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                             fx=cfg.resize_img_scale,
                             fy=cfg.resize_img_scale,
                             interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask


    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        frame_name = self.train_frame_name
        results = {
            'frame_name': frame_name,
        }

        bgcolor = np.array(self.bgcolor, dtype='float32')
        img, alpha = self.load_image(frame_name, bgcolor)
        alpha = alpha[..., 0]
        img = img / 255.
        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton()
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_Rh = dst_skel_info['Rh']
        dst_Th = dst_skel_info['Th']
        K, E = self.get_camera()
        if cfg.mesh.get('for_tet', False):
            dst_bbox = self.skeleton_to_bbox(dst_skel_info['bigpose_joints'])
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_Rh,
                Th=dst_Th)
        R = E[:3, :3]
        T = E[:3, 3]
        min_xyz = dst_bbox['min_xyz']
        max_xyz = dst_bbox['max_xyz']
        voxel_size = cfg.voxel_size
        x = np.arange(min_xyz[0], max_xyz[0] + voxel_size[0],
                    voxel_size[0])
        y = np.arange(min_xyz[1], max_xyz[1] + voxel_size[1],
                    voxel_size[1])
        z = np.arange(min_xyz[2], max_xyz[2] + voxel_size[2],
                    voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)
        shape = pts.shape
        _, pts_2d = get_projection_depth(pts.reshape(-1, 3), K, R, T)
        pts_2d[:, 0] = np.clip(pts_2d[:, 0], 0, W - 1)
        pts_2d[:, 1] = np.clip(pts_2d[:, 1], 0, H - 1)
        inside = (alpha>0)[pts_2d[:, 1], pts_2d[:, 0]]
        inside = inside.reshape(*shape[:-1])

        if 'pts' in self.keyfilter:
            results.update({
                'pts': pts,
                'inside': inside,
                'dst_bbox_min_xyz': dst_bbox['min_xyz'],
                'dst_bbox_max_xyz': dst_bbox['max_xyz'],
            })
        

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints)
            cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms,
                'dst_poses': dst_poses
            })             

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = \
                self.motion_weights_priors.copy()

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



        return results
