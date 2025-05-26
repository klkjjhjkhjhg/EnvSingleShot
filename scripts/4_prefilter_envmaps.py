import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import os.path as osp
import torch, math
import torch.nn.functional as F
import tqdm
import pickle
from utils.render_util import PlanarSVBRDF
from utils.utils import torch_dot, torch_norm
from utils.env_utils import mask, mask_hemisphere
import nvdiffrast.torch as dr
import torchvision
import numpy as np
import argparse

def conv_pix(env, n, roughness, mat_util):
    r = v = n
    sN = 32
    l, _, _, pdfL, _ = mat_util.importanceSampling(n, roughness*2-1, v, sN)
    
    l = torch_norm(l, dim=-3)
    NoL = torch_dot(n, l).clip(0,1)

    #* Convert lighting direction to env map uv
    x, y, z = l[:,:,:,0], l[:,:,:,1], l[:,:,:,2]
    theta = torch.acos(z)
    phi = torch.atan2(y, x)
    u = 1 - phi / (2 * torch.pi) + 0.25
    flag = u > 1.0
    u[flag] = u[flag] - 1.0
    v = theta / torch.pi
    
    #* Fetch env map color
    uv = torch.stack([u, v], dim=-1).view(1, sN**2, -1, 2)
    env = env.permute(0,2,3,1).contiguous()
    light = dr.texture(env, uv, boundary_mode='clamp', filter_mode='linear')
    
    h, w = n.shape[2:]
    light = light.view(1, sN, sN, h, w, 3).permute(0, 1, 2, 5, 3, 4).contiguous()
    mento_carlo = light * NoL# / pdfL
    pix_val = mento_carlo.sum((1,2)) / NoL.sum((1,2))
    return pix_val

def _load_env_data(env_path, scale=False):
        env_data = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
        if env_data is None:
            raise ValueError(f"Cannot load image from {env_path}")

        if env_path.lower().endswith('.png'):
            env_data = env_data.astype(np.float32) / 255.0
        else:
            env_data = env_data.astype(np.float32)

        if env_data.shape[-1] == 4:
            env_data = env_data[:, :, :3]
        if env_data.shape[-1] == 3:
            env_data = env_data[:, :, ::-1]
        env_data = env_data if not scale else scale_light(env_data, 2, 50)
        return env_data
 
def conv_env(env, roughness, mat_util, device='cuda'):
    env = env.to(device)
    h, w = env.shape[2:]
    u = torch.linspace(0, 1, h).to(device)
    v = torch.linspace(0, 1, w).to(device)
    
    u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
    uv_coords = torch.stack([u_grid, v_grid], dim=-1)  # (h, w, 2)

    # 将 UV 坐标转换为球面方向
    phi = -((uv_coords[:, :, 1]-0.25) * 2 * torch.pi)
    phi[:, int(w * 0.75):] = phi[:, int(w * 0.75):] + 2 * torch.pi
    theta = (uv_coords[:, :, 0]) * torch.pi
    directions = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=0).unsqueeze(0)  # (1, 3, h, w)
    
    pS = 256
    if h >= pS:
        #* Conv env maps by patch:
        mipmap = torch.zeros(1,3,h,w).to(device)
        for i in range(h // pS):
            for j in range(w // (pS*2)):
                mipmap[:,:,i*pS:((i+1)*pS), j*pS*2:((j+1)*pS*2)] = conv_pix(env, directions[:, :, i*pS:((i+1)*pS), j*pS*2:((j+1)*pS*2)], roughness, mat_util)
    else:
        mipmap = conv_pix(env, directions, roughness, mat_util)
    return mipmap

def scale_light(x, k, m):
    scale = np.mean(x, axis=-1, keepdims=True) ** k * m
    return x * scale

def main(env_folder, save_root): 
    nLods = 8    
    saveRoot = save_root
    os.makedirs(saveRoot, exist_ok=True)
    vis_path = os.path.join(saveRoot,"vis_mips")
    os.makedirs(vis_path, exist_ok=True)
   
    paths = sorted(os.listdir(env_folder))
    newp = []
    for p in paths:
        if p.endswith('.png') and '.' != p[0]:
            newp.append(p)
    paths = [osp.join(env_folder, p) for p in newp]

    mat_util = PlanarSVBRDF(device='cuda')
    for p in tqdm.tqdm(paths, desc='LoD'):
        # gt_env = cv2.imread(p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        gt_env = _load_env_data(p, scale=True)
        gt_env = gt_env.copy()
        gt_env = torch.from_numpy(gt_env).permute(2,0,1).contiguous()
        gt_env = mask_hemisphere(gt_env)
        gt_env = mask(gt_env, {"ball_center": [0.0, 0.0, 0.0]}) #* Mask the active lighting area in the env map
        gt_env = mask(gt_env, {"ball_center": [0.75, -0.75, 0.0]}) #* Mask the active lighting area in the env map
        gt_env = mask(gt_env, {"ball_center": [-0.75, 0.75, 0.0]}) #* Mask the active lighting area in the env map
        gt_env = mask(gt_env, {"ball_center": [0.75, 0.75, 0.0]}) #* Mask the active lighting area in the env map
        gt_env = mask(gt_env, {"ball_center": [-0.75, -0.75, 0.0]}) #* Mask the active lighting area in the env map
        gt_env = torch.flip(gt_env, [0])
        mipmaps = [gt_env.clone()]
        vis_mips = [gt_env.clone()[None]]
        env = gt_env[None]
        for i in range(nLods):
            #* Downsample env map
            env = F.interpolate(env, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)

            #* Prefilter
            env_mip = conv_env(env, (i+1)/nLods, mat_util)
            env_mip = mask_hemisphere(env_mip[0]).unsqueeze(0)

            #* Save
            mipmaps.append(env_mip.squeeze(0).clone().cpu())
            vis_mips.append(F.interpolate(env_mip.clone().cpu(), scale_factor=2**(i+1), mode='nearest'))
            
        #* Visualize
        torchvision.utils.save_image(torch.flip(torch.cat(vis_mips), dims=[1]), osp.join(vis_path, osp.splitext(osp.basename(p))[0]+'.png'))
        filename = osp.splitext(osp.basename(p))[0]
        pickle.dump(mipmaps, open(osp.join(saveRoot, filename+'.bin'), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expDir', default='expDir')
    opt = parser.parse_args()
    expDir = opt.expDir
    imgFolder = os.path.join(expDir, "envmap")
    saveFolder = os.path.join(expDir, "mipmaps")
    os.makedirs(saveFolder, exist_ok=True)
    main(imgFolder, saveFolder)