import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import MotionBasisComputer
from core.nets.human_nerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp
from core.nets.human_nerf.ops.grid_sample_3d import grid_sample_3d

from configs import cfg

def init_decoder_layer(layer:nn.Module):
    layer.load_state_dict({'weight': torch.tensor([
        #   R       G       B
        [0.298, 0.207, 0.208],  # L1
        [0.187, 0.286, 0.173],  # L2
        [-0.158, 0.189, 0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]).T.contiguous()})

def grid_sample_3d_py(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    iz = ((iz + 1) / 2) * (ID - 1);
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix);
        iy_tnw = torch.floor(iy);
        iz_tnw = torch.floor(iz);

        ix_tne = ix_tnw + 1;
        iy_tne = iy_tnw;
        iz_tne = iz_tnw;

        ix_tsw = ix_tnw;
        iy_tsw = iy_tnw + 1;
        iz_tsw = iz_tnw;

        ix_tse = ix_tnw + 1;
        iy_tse = iy_tnw + 1;
        iz_tse = iz_tnw;

        ix_bnw = ix_tnw;
        iy_bnw = iy_tnw;
        iz_bnw = iz_tnw + 1;

        ix_bne = ix_tnw + 1;
        iy_bne = iy_tnw;
        iz_bne = iz_tnw + 1;

        ix_bsw = ix_tnw;
        iy_bsw = iy_tnw + 1;
        iz_bsw = iz_tnw + 1;

        ix_bse = ix_tnw + 1;
        iy_bse = iy_tnw + 1;
        iz_bse = iz_tnw + 1;

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
            total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires,
                                        cfg.non_rigid_motion_mlp.i_embed)
        self.non_rigid_mlp = \
            load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                skips=cfg.non_rigid_motion_mlp.skips)
        self.non_rigid_mlp = \
            nn.DataParallel(
                self.non_rigid_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.secondary_gpus[0])

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires,
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn

        # canonical mlp
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cnl_pos_embed_size,
                mlp_depth=cfg.canonical_mlp.mlp_depth,
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips,
                rgb_ch=3 if not cfg.vae_mode else 4)
        self.cnl_mlp = \
            nn.DataParallel(
                self.cnl_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.primary_gpus[0])

        # pose decoder MLP
        self.pose_decoder = \
            load_pose_decoder(cfg.pose_decoder.module)(
                embedding_size=cfg.pose_decoder.embedding_size,
                mlp_width=cfg.pose_decoder.mlp_width,
                mlp_depth=cfg.pose_decoder.mlp_depth)

        if cfg.train.freeze_rigid_motion:
            self.freeze_rigid_motion()
        
        if cfg.train.train_albedo_only:
            for name, param in self.named_parameters():
                if not 'output_linear' in name:
                    param.requires_grad = False

        if cfg.vae_mode:
            if not cfg.vae_ft:
                from diffusers import AutoencoderKL
                self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder='vae').to('cuda').eval()
                for param in self.vae.parameters():
                    param.requires_grad = False
                self.decoder_layer = None
            else:
                for param in self.parameters():
                    param.requires_grad=False
                self.decoder_layer = nn.Linear(4, 3, bias=False).to('cuda')
                init_decoder_layer(self.decoder_layer)
                for param in self.decoder_layer.parameters():
                    param.requires_grad=True
                
    
    def freeze_rigid_motion(self):
        for param in self.motion_basis_computer.parameters():
            param.requires_grad = False
        for param in self.mweight_vol_decoder.parameters():
            param.requires_grad = False
        for param in self.pose_decoder.parameters():
            param.requires_grad = False
        

    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self

    def _query_mlp(
            self,
            pos_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
            pos_flat=pos_flat,
            pos_embed_fn=pos_embed_fn,
            non_rigid_mlp_input=non_rigid_mlp_input,
            non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
            chunk=chunk)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
            raws_flat,
            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])
        #xyz_flat = result['xyz']
        #output['xyz'] = torch.reshape(
        #    xyz_flat,
        #    list(pos_xyz.shape[:-1]) + [xyz_flat.shape[-1]])
        return output

    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))

    def _apply_mlp_kernals(
            self,
            pos_flat,
            pos_embed_fn,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            chunk):
        raws = []
        xyz_cnl = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]

            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(
                        non_rigid_mlp_input, total_elem)
                )
                xyz = result['xyz']

            xyz_embedded = pos_embed_fn(xyz)
            raws += [self.cnl_mlp(
                pos_embed=xyz_embedded)]
            xyz_cnl.append(xyz)

        output = {}
        output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0])
        #output['xyz'] = torch.cat(xyz_cnl, dim=0).to(cfg.primary_gpus[0])

        return output

    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            #print(i, rays_flat.shape[0])
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    
    def _batchify_pts(self, pts_flat, **kwargs):
        all_ret = {}
        for i in range(0, pts_flat.shape[0], cfg.chunk):
            #print(i, rays_flat.shape[0])
            ret = self._render_pts(pts_flat=pts_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def _raw2outputs(self, raw, raw_mask, z_vals, rays_d, bgcolor=None, points=None, acc_thres=0.9):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[..., :1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        if cfg.vae_mode:
            if cfg.vae_ft:
                rgb = self.decoder_layer(raw[..., :-1])
                rgb = (rgb + 1) / 2
                rgb = torch.clamp(rgb, 0, 1)
            else:
                rgb = raw[..., :-1]
        else:
            rgb = torch.sigmoid(raw[..., :-1])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[..., -1], dists)  # [N_rays, N_samples]
        if cfg.train.train_albedo_only:
            alpha = alpha.detach()
        alpha = alpha * raw_mask[:, :, 0]

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha),
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        acc_map = torch.sum(weights, -1)
        acc_map[acc_map < cfg.render_th] = 0.


        far = z_vals.max()
        depth_map = torch.sum(weights * z_vals, -1) + (1.-acc_map) * far
        rgb_map = rgb_map + (1.-acc_map[..., None]) * bgcolor[None, ...]/255.

        if points is not None:
            mask = acc_map > acc_thres
            valid_weights = weights[mask]
            valid_points = points[mask]
            weights_max = valid_weights.max(dim=1, keepdim=True)[0]
            mask1 = (valid_weights == weights_max)
            max_points = valid_points[mask1]
            return rgb_map, acc_map, weights, depth_map, max_points

        return rgb_map, acc_map, weights, depth_map

    @staticmethod
    def _rgb_density2outputs(rgb, density, raw_mask, z_vals, rays_d, bgcolor=None, points=None, acc_thres=0.9):
        def _density2alpha(density, dists):
            return 1.0 - torch.exp(-density*dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[..., :1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        alpha = _density2alpha(density, dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha),
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.-acc_map[..., None]) * bgcolor[None, :]/255.

        if points is not None:
            mask = acc_map > acc_thres
            valid_weights = weights[mask]
            valid_points = points[mask]
            weights_max = valid_weights.max(dim=1, keepdim=True)[0]
            mask1 = (valid_weights == weights_max)
            max_points = valid_points[mask1]
            return rgb_map, acc_map, weights, depth_map, max_points

        return rgb_map, acc_map, weights, depth_map

    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3)  # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1]

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(
                motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                * cnl_bbox_scale_xyz[None, :] - 1.0
            #weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :],
            #                        grid=pos[None, None, None, :, :],
            #                        padding_mode='zeros', align_corners=True)
            weights = grid_sample_3d(motion_weights[None, i:i+1, :, :, :], pos[None, None, None, :, :].detach())
            weights = weights[0, 0, 0, 0, :, None]
            weights_list.append(weights)
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights,
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(
                motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
            torch.stack(weighted_motion_fields, dim=0), dim=0
        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}

        if 'x_skel' in output_list:  # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list:  # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask

        return results

    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]
        return rays_o, rays_d, near, far

    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples])

    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)

        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals

    def _render_pts(
        self,
        pts_flat,
        motion_scale_Rs,
        motion_Ts,
        motion_weights_vol,
        cnl_bbox_min_xyz,
        cnl_bbox_scale_xyz,
        pos_embed_fn,
        non_rigid_pos_embed_fn,
        non_rigid_mlp_input=None,
        bgcolor=None,
        **_):
        mv_output = self._sample_motion_fields(
            pts=pts_flat[:, None, :],
            motion_scale_Rs=motion_scale_Rs[0],
            motion_Ts=motion_Ts[0],
            motion_weights_vol=motion_weights_vol,
            cnl_bbox_min_xyz=cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']

        query_result = self._query_mlp(
            pos_xyz=cnl_pts,
            non_rigid_mlp_input=non_rigid_mlp_input,
            pos_embed_fn=pos_embed_fn,
            non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        raw = query_result['raws']
        return {
            'alpha': raw[:, 0, 3] * pts_mask[:, 0, 0],
            'rgb': torch.sigmoid(raw[:, 0, :3])
        }
        
        

    def _render_rays(
            self,
            ray_batch,
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            return_canonical_points=False,
            use_normal_map=False,
            textureless=False,
            **_):

        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        mv_output = self._sample_motion_fields(
            pts=pts,
            motion_scale_Rs=motion_scale_Rs[0],
            motion_Ts=motion_Ts[0],
            motion_weights_vol=motion_weights_vol,
            cnl_bbox_min_xyz=cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']

        query_result = self._query_mlp(
            pos_xyz=cnl_pts,
            non_rigid_mlp_input=non_rigid_mlp_input,
            pos_embed_fn=pos_embed_fn,
            non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        raw = query_result['raws']
        if 'xyz' in query_result:
            canonical_points = query_result['xyz']

        if return_canonical_points:
            rgb_map, acc_map, _, depth_map, points = \
                self._raw2outputs(raw, pts_mask, z_vals, rays_d,
                                  bgcolor, points=canonical_points)
            ret_dict = {'rgb': rgb_map,
                        'alpha': acc_map,
                        'depth': depth_map,
                        'points': points}
        else:
            rgb_map, acc_map, _, depth_map = \
                self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)
            ret_dict = {'rgb': rgb_map,
                        'alpha': acc_map,
                        'depth': depth_map}
        return ret_dict

    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
            dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts

    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    def forward(self,
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                rays=None,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):
        #print('textureless: {}'.format(kwargs.get('textureless', False)))
        dst_Rs = dst_Rs[None, ...]
        dst_Ts = dst_Ts[None, ...]
        dst_posevec = dst_posevec[None, ...]
        cnl_gtfms = cnl_gtfms[None, ...]
        motion_weights_priors = motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)

            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                dst_Rs_no_root,
                refined_Rs)
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
            dst_Rs=dst_Rs,
            dst_Ts=dst_Ts,
            cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol = motion_weights_vol[0]  # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': motion_weights_vol
        })
        if rays is not None:
            rays_o, rays_d = rays
            rays_shape = rays_d.shape

            rays_o = torch.reshape(rays_o, [-1, 3]).float()
            rays_d = torch.reshape(rays_d, [-1, 3]).float()
            packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)
            #print('kwargs.keys()', kwargs.keys())

            all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
            for k in all_ret:
                if k == 'points':
                    continue
                k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
                all_ret[k] = torch.reshape(all_ret[k], k_shape)
        elif 'pts' in kwargs:
            pts = kwargs['pts']
            all_ret = self._batchify_pts(pts_flat=pts.reshape(-1, 3), **kwargs)
            for k in all_ret:
                k_shape = list(pts.shape[:-1]) + list(all_ret[k].shape[1:])
                all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret
