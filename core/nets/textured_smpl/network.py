import torch
import torch.nn as nn
import torch.nn.functional as F
import neural_renderer as nr
import json
import pickle
import numpy as np
from third_parties.smpl.smpl_numpy import SMPL

from core.utils.network_util import MotionBasisComputer
from core.nets.human_nerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp
from core.nets.human_nerf.ops.grid_sample_3d import grid_sample_3d

from configs import cfg

def render_for_nerf(v, f, K, R, t, mask=None, textures=None, focal=1260, img_size=1024, render_size=1024, bgcolor=[0,0,0]):
    texture_size = 8
    vertices = v.unsqueeze(0)
    batch_size = 1
    faces = torch.from_numpy(f.astype(np.int32)).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
    if textures is None:
        if mask is None:
            textures = torch.ones(vertices.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        else:
            textures = torch.zeros(vertices.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
            textures[:, mask] = 1.
    else:
        textures = textures
    renderer = nr.Renderer(camera_mode='projection', image_size=render_size, K=K.unsqueeze(0), R=R.unsqueeze(0), t=t.unsqueeze(0), orig_size=render_size, 
    light_intensity_ambient=1., light_intensity_directional=0.,background_color=bgcolor).cuda()
    images_list = []
    transformation_list = []
    rgbs, _, alphas = renderer(vertices, faces, textures)
    return rgbs, alphas

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.texture_size = 8
        # motion basis computer
        smpl_model = SMPL(sex='neutral', model_dir='third_parties/smpl/models/')
        self.face = smpl_model.faces
        _, _, texture_default = nr.load_obj('smpl_uv.obj', load_texture=True, texture_size=self.texture_size)
        self.texture = torch.nn.Parameter(torch.tensor(texture_default, dtype=torch.float32), requires_grad=True).cuda()

    def deploy_mlps_to_secondary_gpus(self):
        return self
    def forward(self,
                vertices,
                K,
                R,
                t,
                bgcolor,
                img_size,
                render_size,
                **kwargs):
        texture = F.relu(self.texture)
        rgbs, alphas = render_for_nerf(vertices, self.face, K, R, t, 
        textures=texture.unsqueeze(0), bgcolor=bgcolor/255, img_size=img_size, render_size=render_size)
        ret =  dict(rgb_patches=rgbs.permute(0, 2, 3, 1), alpha_patches=alphas)
        return ret
