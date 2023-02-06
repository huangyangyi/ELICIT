import neural_renderer as nr
import numpy as np

import torch

import os
import os.path as osp
import tqdm
from PIL import Image
from tqdm import tqdm
import cv2
import pickle
import yaml
import json

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '387.yaml',
                    'the path of config file')



def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


with open('../../third_parties/smpl/smpl_vert_segmentation.json', 'r') as f:
    smpl_segment = json.load(f)
segment_groups = {
    'body': ['spine1', 'spine2', 'spine', 'hips'],
    'rightArm': ['rightArm', 'rightShoulder'],
    'leftArm': ['leftArm', 'leftShoulder'],
    'rightHand': ['rightForeArm', 'rightHand'],
    'leftHand': ['leftForeArm', 'leftHand'],
    'rightUpLeg': ['rightUpLeg'],
    'leftUpLeg': ['leftUpLeg'],
    'rightLeg': ['rightLeg', 'rightFoot', 'rightToeBase'],
    'leftLeg': ['leftLeg', 'leftFoot', 'leftToeBase'],
    'head': ['head'],
    'neck': ['neck'],
}
colors ={
    'body': 20 * 1,
    'rightArm': 20 * 2,
    'leftArm': 20 * 3,
    'rightHand': 20 * 4,
    'leftHand': 20 * 5,
    'rightUpLeg': 20 * 6,
    'leftUpLeg': 20 * 7,
    'rightLeg': 20 * 8,
    'leftLeg': 20 * 9,
    'head': 20 * 10,
    'neck': 20 * 11,
}
from third_parties.smpl.smpl_numpy import SMPL
smpl_model = SMPL(sex='neutral', model_dir='../../third_parties/smpl/models/')

def apply_global_tfm_to_camera(E, Rh, Th):
    r""" Get camera extrinsics that considers global transformation.

    Args:
        - E: Array (3, 3)
        - Rh: Array (3, )
        - Th: Array (3, )
        
    Returns:
        - Array (3, 3)
    """

    global_tfms = np.eye(4)  #(4, 4)
    global_rot = cv2.Rodrigues(Rh)[0].T
    global_trans = Th
    global_tfms[:3, :3] = global_rot
    global_tfms[:3, 3] = -global_rot.dot(global_trans)
    return E.dot(np.linalg.inv(global_tfms))

def get_camrot(campos, lookat=None, inv_camera=False):
    r""" Compute rotation part of extrinsic matrix from camera posistion and
         where it looks at.

    Args:
        - campos: Array (3, )
        - lookat: Array (3, )
        - inv_camera: Boolean

    Returns:
        - Array (3, 3)

    Reference: http://ksimek.github.io/2012/08/22/extrinsic/
    """

    if lookat is None:
        lookat = np.array([0., 0., 0.], dtype=np.float32)

    # define up, forward, and right vectors
    up = np.array([0., 1., 0.], dtype=np.float32)
    if inv_camera:
        up[1] *= -1.0
    forward = lookat - campos
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    camrot = np.array([right, up, forward], dtype=np.float32)
    return camrot
def norm_np_arr(arr):
    return arr / np.linalg.norm(arr)
def lookat(eye, target, up):
    zaxis = norm_np_arr(eye - target)
    xaxis = norm_np_arr(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis)
    viewMatrix = np.array([
        [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
        [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
        [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, eye)],
        [0       , 0       , 0       , 1                  ]
    ])
    viewMatrix = np.linalg.inv(viewMatrix)
    return viewMatrix, xaxis, yaxis, zaxis
def render_for_nerf(v, f, K, E, Rh, Th, mask=None, textures=None, focal=1260, img_size=1024):
    texture_size = 8
    batch_size = v.shape[0]
    vertices = v.clone()
    faces = torch.from_numpy(f.astype(np.int32)).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
    if textures is None:
        if mask is None:
            textures = torch.ones(vertices.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
        else:
            textures = torch.zeros(vertices.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
            textures[:, mask] = 1.
    else:
        textures = textures.cuda()
    E = apply_global_tfm_to_camera(
                E=E, 
                Rh=Rh,
                Th=Th)
    R = E[:3, :3]
    #print(np.arctan2(R[0, 2], R[2, 2])/np.pi * 180)
    t = E[:3, 3]
    K = torch.tensor(K, dtype=torch.float).unsqueeze(0).cuda()
    R = torch.tensor(R, dtype=torch.float).unsqueeze(0).cuda()
    t = torch.tensor(t, dtype=torch.float).unsqueeze(0).cuda()
    renderer = nr.Renderer(camera_mode='projection', image_size=img_size, K=K, R=R, t=t, orig_size=1024, light_intensity_ambient=1., light_intensity_directional=0.).cuda()
    images_list = []
    transformation_list = []
    images, _, _ = renderer(vertices, faces, textures)
    images_list.append(images)
    images = torch.cat(images_list, 0)
    detached_images = (images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).squeeze(0)
    return detached_images

def render_for_nerf_batch(v, f, K, E, Rhs, Ths, mask=None, textures=None, focal=1260, img_size=1024):
    texture_size = 8
    batch_size = v.shape[0]
    vertices = v.clone()
    faces = torch.from_numpy(f.astype(np.int32)).cuda()
    if textures is None:
        if mask is None:
            textures = torch.ones(vertices.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
        else:
            textures = torch.zeros(vertices.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
            textures[:, mask] = 1.
    else:
        textures = textures.cuda()
    Rs = []
    ts = []
    for Rh, Th in zip(Rhs, Ths):
        E1 = apply_global_tfm_to_camera(
                    E=E, 
                    Rh=Rh,
                    Th=Th)
        R = E1[:3, :3]
        t = E1[:3, 3]
        Rs.append(R)
        ts.append(t)
    #print(Rs[0])
    #print(ts[0])
    Ks = torch.tensor(K, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    Rs = torch.tensor(np.array(Rs), dtype=torch.float).cuda()
    ts = torch.tensor(np.array(ts), dtype=torch.float).cuda().unsqueeze(1)
    #print(vertices.shape, Rs.shape, Ks.shape, ts.shape)
    renderer = nr.Renderer(camera_mode='projection', image_size=img_size, K=Ks, R=Rs, t=ts, orig_size=1024, light_intensity_ambient=1., light_intensity_directional=0.).cuda()
    images, _, _ = renderer(vertices, faces, textures)
    detached_images = (images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    return detached_images



with open('../../third_parties/smpl/segment_faces.pkl', 'rb') as f:
    segment_faces = pickle.load(f)
    
_, _, texture_default = nr.load_obj('../../third_parties/smpl/smpl_uv.obj', load_texture=True, texture_size=8)



def main(argv):
    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sample_interval = cfg['smpl_interval']
    source_dir = cfg['dataset']['zju_mocap_path']
    output_dir = cfg['output']['dir']
    output_name = cfg['output']['name']
    select_view = cfg['training_view']
    n_frames = cfg['max_frames']

    print('Processing {}'.format(subject))
    input_dir = os.path.join(source_dir, subject)
    output_path = os.path.join(output_dir, output_name)
    smpl_params_custom = np.load(osp.join(output_path, 'gt_params.npz'))
    custom_betas = smpl_params_custom['shape']
    custom_poses = smpl_params_custom['pose']
    out_img_dir  = osp.join(output_path, 'images')
    out_mask_dir = osp.join(output_path, 'masks')
    out_segment_dir = osp.join(output_path, 'segments')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_segment_dir, exist_ok=True)

    anno_path = os.path.join(input_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()

    data = smpl_params_custom
    poses = data['pose'][0]
    betas = data['shape'][0]
    rotmat = data.get('rotmat', None)
    Th = data['global_t'][0]
    Rh = np.array([0., 0., 0.])
    if 'global_r' in data:
        Rh = data['global_r'][0]
    focal = data['focal_l']
    img_size = 1024
    K = np.eye(3, dtype='float32')
    K[0, 0] = focal
    K[1, 1] = focal
    K[:2, 2] = img_size / 2.
    E = np.eye(4)
    _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
    vertices, joints = smpl_model(poses, betas)
    face = smpl_model.faces
    textures = torch.zeros(1, face.shape[0], 8, 8, 8, 3, dtype=torch.float32)
    for seg_key, seg_faces in segment_faces.items():
        textures[:, seg_faces] = colors[seg_key]
    segment_out = render_for_nerf(torch.tensor(vertices, dtype=torch.float).cuda().unsqueeze(0), 
                                face, K=K, E=E, Rh=Rh, Th=Th, textures=textures)
    Image.fromarray(segment_out).save(osp.join(output_path, 'gt_segments.png'))

    # load cameras
    cams = annots['cams']
    cam_Ks = np.array(cams['K']).astype('float32')
    cam_Rs = np.array(cams['R']).astype('float32')
    cam_Ts = np.array(cams['T']).astype('float32') / 1000.
    cam_Ds = np.array(cams['D']).astype('float32')
    smpl_params = np.load(os.path.join(input_dir, 'new_params', '{}.npy'.format(0)), allow_pickle=True).item()
    Th = smpl_params['Th'][0]  #(3,)

    n_cams = len(cams['K'])
    Ks = cam_Ks     #(N, 3, 3)
    Ds = cam_Ds[:, :, 0]
    Es = np.stack([np.eye(4) for _ in range(n_cams)])  #(N, 4, 4)
    cam_Ts = cam_Ts[:, :3, 0]
    Es[:, :3, :3] = cam_Rs
    Es[:, :3, 3]= cam_Ts
    
    batch_size = 4
    cameras = {}
    mesh_infos = {}
    all_betas = []
    for fs in tqdm(range(0, n_frames+1, batch_size*sample_interval)):
        for i in range(n_cams):
            ft = min(fs + batch_size*sample_interval, n_frames+1)
            K = Ks[i]
            E = Es[i]
            batch_out_names = []
            batch_Rh = []
            batch_Th = []
            batch_vertices = []
            batch_faces = []
            for fi in range(fs, ft, sample_interval):
                out_name = 'frame_{:06d}_{:02d}'.format(fi, i)
                if fi < n_frames:
                    smpl_params = np.load('{}/new_params/{}.npy'.format(input_dir, fi), allow_pickle=True).item()
                    betas = smpl_params['shapes'][0] #(10,)
                    poses = smpl_params['poses'][0]  #(72,)
                elif fi == n_frames:
                    smpl_params = np.load('{}/new_params/{}.npy'.format(input_dir, fi-1), allow_pickle=True).item()
                    poses = custom_poses[0]
                if custom_betas is not None:
                    betas = custom_betas[0]
                Rh = smpl_params['Rh'][0]  #(3,)
                Th = smpl_params['Th'][0]  #(3,)

                all_betas.append(betas)
                # write camera info
                cameras[out_name] = {
                        'intrinsics': K,
                        'extrinsics': E,
                        #'distortions': D
                }
                # write mesh info
                _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
                vertices, joints = smpl_model(poses, betas)
                mesh_infos[out_name] = {
                    'Rh': Rh,
                    'Th': Th,
                    'poses': poses,
                    'joints': joints, 
                    'tpose_joints': tpose_joints
                }
                face = smpl_model.faces
                #print(get_body_direction(E, Rh, Th))
                batch_out_names.append(out_name)
                batch_Rh.append(Rh)
                batch_Th.append(Th)
                batch_vertices.append(vertices)
                batch_faces.append(face)
            batch_faces = np.stack(batch_faces)
            batch_Rh = np.stack(batch_Rh)
            batch_Th = np.stack(batch_Th)
            batch_vertices = np.stack(batch_vertices)
            image_outs = render_for_nerf_batch(torch.tensor(batch_vertices, dtype=torch.float).cuda(), 
                                        batch_faces, K=K, E=E, Rhs=batch_Rh, Ths=batch_Th, textures=texture_default.unsqueeze(0).repeat(batch_faces.shape[0], 1, 1, 1, 1, 1))
            textures = torch.zeros(batch_faces.shape[0], face.shape[0], 8, 8, 8, 3, dtype=torch.float32)
            for seg_key, seg_faces in segment_faces.items():
                textures[:, seg_faces] = colors[seg_key]
            segments = render_for_nerf_batch(torch.tensor(batch_vertices, dtype=torch.float).cuda(), 
                                    batch_faces, K=K, E=E, Rhs=batch_Rh, Ths=batch_Th, textures=textures)
            for image_out, segment, out_name in zip(image_outs, segments, batch_out_names):
                segment = Image.fromarray(segment)
                segment.save(osp.join(out_segment_dir, '{}.png'.format(out_name)))

                mask = Image.fromarray((image_out[:, :, 0] > 0).astype(np.uint8)*255)
                image_out = Image.fromarray(image_out)
                image_out.save(osp.join(out_img_dir, "{}.png".format(out_name)))
                mask.save(osp.join(out_mask_dir, "{}.png".format(out_name)))
                #print(osp.join(out_img_dir, "{}.png".format(out_name)))

        with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
            pickle.dump(cameras, f)

        # write mesh infos
        with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
            pickle.dump(mesh_infos, f)
        
        # write canonical joints
        avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
        _, template_joints = smpl_model(np.zeros(72), avg_betas)
        with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
            pickle.dump(
                {
                    'joints': template_joints,
                }, f)

if __name__ == '__main__':
    app.run(main)
