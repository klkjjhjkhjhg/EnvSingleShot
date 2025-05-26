import torch
import torch.nn.functional as F

def mask(env_map, config={}, ret_mask=False):
    cfg = {
        "polygon": [
            [-3.375, -2, 3.3919],
            [0.6250, -2, 3.3919],
            [0.6250,  2, 3.3919],
            [-3.375,  2, 3.3919],
        ],
        "ball_center": [1.375, 0.0, 0.0],
        "ball_radius": 0.225,
        "square_len": 4
    }
    cfg.update(config)
    p = torch.tensor(cfg['polygon'], dtype=torch.float32)
    c = torch.tensor(cfg['ball_center'], dtype=torch.float32)
    r = torch.tensor(cfg['ball_radius'], dtype=torch.float32)
    aabb = light_aabb(p, r, c)
    ret, inlight = mask_env_map(env_map, aabb, c, p, cfg['square_len'])
    if ret_mask:
        return ret, inlight
    else:
        return ret
    
def get_pattern(env_map, config={}):
    cfg = {
        "polygon": [
            [-3.375, -2, 3.3919],
            [0.6250, -2, 3.3919],
            [0.6250,  2, 3.3919],
            [-3.375,  2, 3.3919],
        ],
        "ball_center": [1.375, 0.0, 0.0],
        "ball_radius": 0.225
    }
    lenx, leny = 4, 4
    cfg.update(config)
    p = torch.tensor(cfg['polygon'], dtype=torch.float32)
    c = torch.tensor(cfg['ball_center'], dtype=torch.float32)
    r = torch.tensor(cfg['ball_radius'], dtype=torch.float32)
    dirx = p[1] - p[0]
    diry = p[3] - p[0]
    dirx_norm = dirx / torch.norm(dirx)
    diry_norm = diry / torch.norm(diry)
    
    u, v = torch.meshgrid(torch.linspace(0, 1, 256), torch.linspace(0, 1, 256), indexing='ij')
    uv_coords = (torch.stack([v, 1-u, torch.zeros_like(u)], dim=-1)) * lenx
    offset = uv_coords[..., 0:1] * dirx_norm.view(1,1,3) + uv_coords[..., 1:2] * diry_norm.view(1,1,3)
    xl = p[None, None, 0] + offset
    
    directions = xl - c  # Shape: (4, 3)
    
    # Normalize direction vectors to unit length
    directions_norm = directions / torch.norm(directions, dim=-1, keepdim=True)  # (num_pixels, 3)

    # Convert normalized direction vectors to spherical coordinates
    x, y, z = directions_norm[:, :, 0], directions_norm[:, :, 1], directions_norm[:, :, 2]
    theta = torch.acos(z)  # Polar angle
    phi = torch.atan2(y, x)  # Azimuthal angle
    
    # Convert spherical coordinates to UV mapping
    envu = 1 - phi / (2 * torch.pi) + 0.25
    flag = envu > 1.0
    envu[flag] = envu[flag] - 1.0
    envv = theta / torch.pi
    _, envH, envW = env_map.shape
    envv = (envv * envH).long()
    envu = (envu * envW).long()
    pattern = env_map[:, envv, envu] / 255.
    pattern = (pattern ** 2.2) * 255
    return pattern.cpu().permute(1,2,0).contiguous().numpy()

def light_aabb(polygon, ball_radius, ball_center):
    """
    遍历 AABB 的所有像素，将遮挡区域设置为零。
    
    Args:
        env_map (torch.Tensor): 环境贴图 (H, W)。
        aabb (torch.Tensor): UV 的轴对齐边界框 [[u_min, v_min], [u_max, v_max]]。
        ball_center (torch.Tensor): 球心坐标 (3,)。
        ball_radius (float): 球半径。
        light_square (torch.Tensor): 光源正方形四个顶点坐标 (4, 3)。
        
    Returns:
        torch.Tensor: 更新后的环境贴图。
    """
    # Calculate direction vectors from the sphere center to each vertex
    directions = polygon - ball_center  # Shape: (4, 3)
    
    # Normalize direction vectors to unit length
    directions_norm = directions / torch.norm(directions, dim=-1, keepdim=True)  # (num_pixels, 3)

    # Convert normalized direction vectors to spherical coordinates
    x, y, z = directions_norm[:, 0], directions_norm[:, 1], directions_norm[:, 2]
    theta = torch.acos(z)  # Polar angle
    phi = torch.atan2(y, x)  # Azimuthal angle
    
    # Convert spherical coordinates to UV mapping
    u = phi / (2 * torch.pi) + 0.75
    v = theta / torch.pi
    uv_coords = torch.stack([v, u], dim=-1)  # Shape: (4, 2)
    
    aabb = torch.tensor([[uv_coords[:,0].min(), 0.0], 
                        [uv_coords[:,0].max(), 1.0]], dtype=torch.float32)
    
    return aabb

def mask_env_map(env_map, aabb, ball_center, light_square, square_len=4):
    """
    遍历 AABB 的所有像素，将遮挡区域设置为零。
    
    Args:
        env_map (torch.Tensor): 环境贴图 (H, W)。
        aabb (torch.Tensor): UV 的轴对齐边界框 [[u_min, v_min], [u_max, v_max]]。
        ball_center (torch.Tensor): 球心坐标 (3,)。
        ball_radius (float): 球半径。
        light_square (torch.Tensor): 光源正方形四个顶点坐标 (4, 3)。
        
    Returns:
        torch.Tensor: 更新后的环境贴图。
    """
    # 生成 AABB 内所有像素点的 UV 坐标
    # h, w = torch.ceil(env_map.shape[1] * (aabb[1, 0]-aabb[0, 0])).int(), torch.ceil(env_map.shape[2] * (aabb[1, 1]-aabb[0, 1])).int()
    h, w = env_map.shape[1:]
    u_vals = torch.linspace(0.0, 1.0, env_map.shape[1])
    v_vals = torch.linspace(0.0, 1.0, env_map.shape[2])
    u_grid, v_grid = torch.meshgrid(u_vals, v_vals, indexing='ij')
    uv_coords = torch.stack([u_grid, v_grid], dim=-1).reshape(-1, 2)  # (num_pixels, 2)

    # 将 UV 坐标转换为球面方向
    phi = -((uv_coords[:, 1]-0.25) * 2 * torch.pi)
    phi[int(512 * 0.75):] = phi[int(512 * 0.75):] + 2 * torch.pi
    theta = (uv_coords[:, 0]) * torch.pi
    directions = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=-1)  # (num_pixels, 3)

    # 射线起点：球心
    ray_origins = ball_center.unsqueeze(0).expand_as(directions)  # (num_pixels, 3)

    # 调用 intersect 函数检测射线与平面交点
    ray_dirs = directions  # 射线方向
    hits, t_vals = intersect(light_square, ray_dirs, ray_origins)

    # 判断交点是否在正方形光源的平面范围内
    intersection_points = ray_origins + t_vals.unsqueeze(-1) * ray_dirs  # 交点坐标 (num_pixels, 3)
    in_light = check_in_square(intersection_points, light_square, square_len)  # 自定义函数判断交点是否在正方形内

    # 更新环境贴图：将遮挡区域设置为零
    in_light = in_light.view(h, w) & hits.view(h, w)
    
    # h_range = torch.round(aabb[:, 0] * env_map.shape[1]).int()
    in_light = in_light.unsqueeze(0).expand(3, *in_light.shape)
    env_map[:, :, :][in_light] = 0  # 遮挡区域置零

    return env_map, in_light

def mask_hemisphere(env_map):
    h, w = env_map.shape[1], env_map.shape[2]
    u_vals = torch.linspace(0.0, 1.0, env_map.shape[1])
    v_vals = torch.linspace(0.0, 1.0, env_map.shape[2])
    u_grid, v_grid = torch.meshgrid(u_vals, v_vals, indexing='ij')
    uv_coords = torch.stack([u_grid, v_grid], dim=-1).reshape(-1, 2)  # (num_pixels, 2)

    # 将 UV 坐标转换为球面方向
    theta = (uv_coords[:, 0]) * torch.pi
    mask = theta > torch.pi / 2
    in_light = mask.view(h, w)
    env_map[:, in_light] = 0
    return env_map
    
def intersect(light_square, ray_dirs, ray_origins, eps=1e-6):
    """
    计算射线与平面交点。

    Args:
        light_square (torch.Tensor): 光源平面顶点 (4, 3)。
        ray_dirs (torch.Tensor): 射线方向 (num_pixels, 3)。
        eps (float): 防止除零的小值。

    Returns:
        torch.Tensor: 是否相交 (num_pixels,)。
        torch.Tensor: 相交参数 t 值 (num_pixels,)。
    """
    # 光源平面的法向量
    v1 = F.normalize(light_square[1] - light_square[0], dim=-1)
    v2 = F.normalize(light_square[3] - light_square[0], dim=-1)
    tangent_normal = torch.cross(v2, v1, dim=-1) # 平面法向量 (3,)

    # 计算射线与平面相交的参数 t
    cosTheta = torch.sum(tangent_normal * ray_dirs, dim=-1)  # (num_pixels,)
    cosTheta[torch.abs(cosTheta) < eps] = eps  # 防止除零
    tangentPlane = torch.cat([tangent_normal, -(light_square[0]*tangent_normal).sum(-1, keepdims=True)], dim=0)

    t_vals = - (tangentPlane.unsqueeze(0)*torch.cat([ray_origins, torch.ones_like(ray_origins[:, 0:1])], dim=-1)).sum(-1) / cosTheta
    # 返回是否相交 (t > 0) 和 t 值
    hits = t_vals > 0
    return hits, t_vals

def check_in_square(points, light_square, square_len=4, border=0.3):
    """
    判断点是否在正方形光源平面内。

    Args:
        points (torch.Tensor): 交点坐标 (num_pixels, 3)。
        light_square (torch.Tensor): 光源正方形顶点 (4, 3)。

    Returns:
        torch.Tensor: 点是否在正方形内 (num_pixels,)。
    """
    p0 = light_square[0]
    dirx = light_square[1] - p0
    diry = light_square[3] - p0

    # Normalize dirx and diry to unit length
    dirx_norm = dirx / torch.norm(dirx)
    diry_norm = diry / torch.norm(diry)
    
    # Project points onto dirx and diry
    proj_x = torch.sum((points - p0) * dirx_norm, dim=-1)
    proj_y = torch.sum((points - p0) * diry_norm, dim=-1)
    # Check if projections are within the range [-4, 4]
    in_square = (proj_x >= 0-border) & (proj_x <= square_len+border) & (proj_y >= 0-border) & (proj_y <= square_len+border)

    return in_square