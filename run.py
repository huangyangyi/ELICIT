import skimage
import os, shutil

import torch
import numpy as np
from tqdm import tqdm
import cv2

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from configs import cfg, args
from third_parties.lpips import LPIPS
import mcubes, trimesh
import PIL

EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask']


def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


def get_loss(lpips, rgb, target):
    lpips_loss = lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2).cuda()), 
                       scale_for_lpips(target.permute(0, 3, 1, 2).cuda()))
    return torch.mean(lpips_loss).cpu().detach().numpy()


def load_network():
    model = create_network()
    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_depth_map(depth_vals, ray_mask, width, height):
    depth_map = np.zeros((height * width), dtype='float32')
    depth_map[ray_mask] = depth_vals / depth_vals.max()
    return depth_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None, depth=None):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')
    depth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    if depth is not None:
        depth_map = unpack_depth_map(depth, ray_mask, width, height)
        depth_image  = to_8b3ch_image(depth_map)
    return rgb_image, alpha_image, truth_image, depth_image


def _freeview(
        data_type='freeview',
        folder_name=None):
    cfg.perturb = 0.

    model = load_network()
    test_loader = create_dataloader(data_type)
    writer = ImageWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net),
                exp_name=folder_name, 
                fps=cfg.fps)

    model.eval()
    lpips = LPIPS(net='vgg').cuda()
    #all_points = []
    psnr_l = []
    ssim_l = []
    lpips_l = []
    for batch in tqdm(test_loader):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            net_output = model(**data, 
                               iter_val=cfg.eval_iter, use_normal_map=cfg.get('use_normal_map', False), textureless=cfg.get('textureless', False))

        rgb = net_output['rgb']
        alpha = net_output['alpha']
        depth = net_output.get('depth', None)
        if depth is not None:
            depth = depth.data.cpu().numpy()

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        target_rgbs = batch.get('target_rgbs', None)

        rgb_img, alpha_img, _, depth_img = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy(), depth=depth)
        depth_map = unpack_depth_map(depth, ray_mask, width, height)
        alpha_map = unpack_alpha_map(alpha.data.cpu().numpy(), ray_mask, width, height)
        writer.append_numpy(dict(depth=depth_map, alpha=alpha_map))
        
        imgs = [rgb_img]
        if cfg.show_truth and target_rgbs is not None:
            target_rgbs = to_8b_image(target_rgbs.numpy())
            imgs.append(target_rgbs)
        if cfg.show_alpha:
            imgs.append(alpha_img)
        if cfg.show_depth:
            imgs.append(depth_img)

        pred_img_norm = rgb_img / 255
        gt_img_norm = target_rgbs / 255
        if isinstance(gt_img_norm, torch.Tensor):
            gt_img_norm = gt_img_norm.cpu().numpy()
        #pred_img_norm = pred_img_norm.cpu().numpy()
        psnr_l.append(psnr_metric(pred_img_norm.reshape(-1, 3)[ray_mask.cpu().numpy().astype(np.bool)], gt_img_norm.reshape(-1, 3)[ray_mask.cpu().numpy().astype(np.bool)]))
        x, y, w, h = cv2.boundingRect(ray_mask.reshape(gt_img_norm.shape[:2]).cpu().numpy().astype(np.uint8)*255)
        pred_img_norm = pred_img_norm[y:y + h, x:x + w]
        gt_img_norm = gt_img_norm[y:y + h, x:x + w]
        ssim_l.append(skimage.metrics.structural_similarity(pred_img_norm, gt_img_norm, multichannel=True))
        lpips_loss = get_loss(lpips=lpips, rgb=torch.from_numpy(pred_img_norm).float().unsqueeze(0), target=torch.from_numpy(gt_img_norm).float().unsqueeze(0))
        lpips_l.append(lpips_loss)

        img_out = np.concatenate(imgs, axis=1)
        print(psnr_l[-1], ssim_l[-1], lpips_l[-1])
        writer.append(img_out)

    writer.finalize()
    print ('PSNR:', np.array(psnr_l).mean())
    print ('SSIM:', np.array(ssim_l).mean())
    print ('LPIPS:', np.array(lpips_l).mean())



def run_mesh():
    print('Warning: experimental feature!')
    data_type='mesh'
    folder_name=f"mesh_{cfg.mesh.frame_name}" \
            if not cfg.render_folder_name else cfg.render_folder_name
    if cfg.mesh.render_gt_view:
        folder_name = 'mesh_gt'
    cfg.perturb = 0.

    model = load_network()
    test_loader = create_dataloader(data_type)
    output_dir = os.path.join(cfg.logdir, cfg.load_net, folder_name)
    if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]
        mesh_kwargs = dict()
        if cfg.mesh.get('for_tet', False):
            bigpose_dst_Rs=batch['bigpose_dst_Rs']
            bigpose_dst_Ts=batch['bigpose_dst_Ts']
            bigpose_cnl_gtfms=batch['bigpose_cnl_gtfms']
            bigpose=batch['bigpose']
            bigpose_vertices=batch['bigpose_vertices']
            dst_Rs=batch['dst_Rs']
            dst_Ts=batch['dst_Ts']
            cnl_gtfms=batch['cnl_gtfms']
            dst_poses=batch['dst_poses']
            pvertices=batch['pvertices']
            batch.update(dict(
                dst_Rs=bigpose_dst_Rs,
                dst_Ts=bigpose_dst_Ts,
                cnl_gtfms=bigpose_cnl_gtfms,
                dst_poses=bigpose,
                pvertices=bigpose_vertices,
            ))
        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter, return_canonical_points=False, use_normal_map=cfg.get('use_normal_map', False), textureless=cfg.get('textureless', False))

        alpha = net_output['alpha'].cpu().detach().numpy()
        pts = batch['pts'].detach().numpy()
        inside = batch['inside'].cpu().detach().numpy()
        cube = np.zeros_like(alpha)
        cube[inside == 1] = alpha[inside == 1]
        np.save(os.path.join(cfg.logdir, cfg.load_net, folder_name, f'{idx:06d}_density.npy'), dict(pts=pts, alpha=cube))
        cube = np.pad(cube, 10, mode='constant')    
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        vertices = (vertices - 10) * cfg.voxel_size
        vertices = vertices + batch['dst_bbox_min_xyz'].detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh = max(mesh.split(), key=lambda m: len(m.vertices))

        batch['pts'] = torch.tensor(mesh.vertices, dtype=torch.float32, device=batch['pts'].device)
        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
        with torch.no_grad():
            net_output_new = model(**data, iter_val=cfg.eval_iter, return_canonical_points=False, use_normal_map=cfg.get('use_normal_map', False), textureless=cfg.get('textureless', False))
        colrs = net_output_new['rgb'].cpu().detach().numpy()

        import pymeshlab
        mesh_kwargs['v_color_matrix'] = np.concatenate((colrs, np.ones((colrs.shape[0], 1), dtype=colrs.dtype)), axis=-1).astype(np.float64)
        m = pymeshlab.Mesh(mesh.vertices, mesh.faces, **mesh_kwargs)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, "level_set")
        # UV map and turn vertex coloring into a texture
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10240)
        ms.compute_texmap_from_color(textname=f"tex_{idx:06d}")
        ms.save_current_mesh(os.path.join(cfg.logdir, cfg.load_net, folder_name, f'{idx:06d}.obj'))



def run_freeview():
    _freeview(
        data_type='freeview',
        folder_name=f"freeview_{cfg.freeview.frame_name}" \
            if not cfg.render_folder_name else cfg.render_folder_name)



def run_movement(render_folder_name='movement'):
    cfg.perturb = 0.

    model = load_network()
    test_loader = create_dataloader('movement')
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net),
        exp_name=render_folder_name,
        fps=cfg.fps)

    model.eval()
    lpips = LPIPS(net='vgg').cuda()
    #all_points = []
    psnr_l = []
    ssim_l = []
    lpips_l = []

    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter, return_canonical_points=False, use_normal_map=cfg.get('use_normal_map', False), textureless=cfg.get('textureless', False))

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        if 'rgb_patches' in net_output and 'alpha_patches' in net_output:
            rgb = net_output['rgb_patches'].reshape(-1,3)[ray_mask]
            alpha = net_output['alpha_patches'].reshape(-1)[ray_mask]
        else:
            rgb = net_output['rgb']
            alpha = net_output['alpha']
        rgb_img, alpha_img, truth_img, _ = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])

        pred_img_norm = rgb_img / rgb_img.max()
        gt_img_norm = truth_img / truth_img.max()
        
        psnr_l.append(psnr_metric(pred_img_norm.reshape(-1, 3)[ray_mask.cpu().numpy().astype(np.bool)], gt_img_norm.reshape(-1, 3)[ray_mask.cpu().numpy().astype(np.bool)]))
        x, y, w, h = cv2.boundingRect(ray_mask.reshape(gt_img_norm.shape[:2]).cpu().numpy().astype(np.uint8)*255)
        pred_img_norm = pred_img_norm[y:y + h, x:x + w]
        gt_img_norm = gt_img_norm[y:y + h, x:x + w]
        ssim_l.append(skimage.metrics.structural_similarity(pred_img_norm, gt_img_norm, multichannel=True))
        lpips_loss = get_loss(lpips=lpips, rgb=torch.from_numpy(pred_img_norm).float().unsqueeze(0), target=torch.from_numpy(gt_img_norm).float().unsqueeze(0))
        lpips_l.append(lpips_loss)
        print(psnr_l[-1], ssim_l[-1], lpips_l[-1])
        
        imgs = [rgb_img]
        if cfg.show_truth:
            imgs.append(truth_img)
        if cfg.show_alpha:
            imgs.append(alpha_img)
        if cfg.show_depth:
            imgs.append(depth_img)
            
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=f"{idx:06d}")
    
    #torch.save(all_points, 'points.pth')
    writer.finalize()

    print ('PSNR:', np.array(psnr_l).mean())
    print ('SSIM:', np.array(ssim_l).mean())
    print ('LPIPS:', np.array(lpips_l).mean())
        
if __name__ == '__main__':
    globals()[f'run_{args.type}']()
