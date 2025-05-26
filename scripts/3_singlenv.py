import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import xml.etree.ElementTree as et
import cv2

# 计算多边形的中心
def calculate_polygon_center(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    z_coords = [point[2] for point in polygon]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    center_z = sum(z_coords) / len(z_coords)
    return [center_x, center_y, center_z]

# 调整多边形中心
def adjust_polygon_center(polygon, dx, dy, dz):
    new_polygon = [[x + dx, y + dy, z + dz] for x, y, z in polygon]
    return new_polygon

def param2polygon(center, dirx, diry, halfx, halfy):
    ex = dirx * halfx
    ey = diry * halfy
    polygon = [
        center - ex - ey,   
        center + ex - ey,   
        center + ex + ey,   
        center - ex + ey,   
    ]
    return polygon

def adjust_polygon_dir(polygon, dx, dy):
    polygon = np.array(polygon)
    dirx = polygon[1] - polygon[0]
    diry = polygon[3] - polygon[0]
    halfx = halfy = 2
    center = calculate_polygon_center(polygon)
    
    dirx[2] = dirx[2] + dx
    dirx = dirx / np.linalg.norm(dirx)
    diry[2] = diry[2] + dy
    diry = diry / np.linalg.norm(diry)
    
    new_polygon = param2polygon(center, dirx, diry, halfx, halfy)
    return new_polygon

def sample_cone_vectors(l, r, num_samples=64):
    # Normalize the input vectors l to ensure they are unit vectors
    l = l / np.linalg.norm(l, axis=1, keepdims=True)  # Normalize each row
    
    # Generate random samples for theta and phi
    theta = np.random.rand(num_samples) * np.radians(r)
    phi = 2 * np.pi * np.random.rand(num_samples)
    
    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Stack the random spherical samples into a (num_samples, 3) matrix
    vectors = np.vstack([x, y, z]).T  # Shape: (num_samples, 3)
    
    # Calculate the axis of rotation (cross product of z-axis and each l_i)
    z_axis = np.array([0, 0, 1])
    axis = np.cross(np.tile(z_axis, (l.shape[0], 1)), l)  # Shape: (n, 3)
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = axis / axis_norm  # Normalize each row (axis)
    
    # Calculate the angle between the z-axis and each l_i
    angle = np.arccos(np.dot(l, z_axis))  # Shape: (n,)
    
    # Precompute sin and cos of the angles
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rotation matrix using Rodrigues' rotation formula for each vector
    R = np.empty((l.shape[0], 3, 3))  # Shape: (n, 3, 3)
    
    # Rodrigues' rotation formula components
    R[:, 0, 0] = cos_angle + axis[:, 0]**2 * (1 - cos_angle)
    R[:, 0, 1] = axis[:, 0]*axis[:, 1]*(1 - cos_angle) - axis[:, 2]*sin_angle
    R[:, 0, 2] = axis[:, 0]*axis[:, 2]*(1 - cos_angle) + axis[:, 1]*sin_angle
    
    R[:, 1, 0] = axis[:, 1]*axis[:, 0]*(1 - cos_angle) + axis[:, 2]*sin_angle
    R[:, 1, 1] = cos_angle + axis[:, 1]**2 * (1 - cos_angle)
    R[:, 1, 2] = axis[:, 1]*axis[:, 2]*(1 - cos_angle) - axis[:, 0]*sin_angle
    
    R[:, 2, 0] = axis[:, 2]*axis[:, 0]*(1 - cos_angle) - axis[:, 1]*sin_angle
    R[:, 2, 1] = axis[:, 2]*axis[:, 1]*(1 - cos_angle) + axis[:, 0]*sin_angle
    R[:, 2, 2] = cos_angle + axis[:, 2]**2 * (1 - cos_angle)
    
    # Reshape vectors to (num_samples, 1, 3) to match with R's shape for broadcasting
    vectors = np.expand_dims(vectors, axis=-1)  # Shape: (num_samples, 3, 1)
    
    # Perform batch matrix multiplication to rotate all vectors
    rotated_vectors = np.matmul(R[:,None], vectors[None]).squeeze(-1)  # Shape: (n, num_samples, 3)

    return rotated_vectors

def mask_env_map_with_color(env, threshold=20, threshold_rgb= 20):
    def linear_to_display_p3(linear_color):
        # 对于小于等于0.0031308的颜色，使用12.92倍
        # 对于大于0.0031308的颜色，应用1.055次方的转换
        return np.where(linear_color <= 0.0031308, 
                        12.92 * linear_color, 
                        1.055 * np.power(linear_color, 1/2.4) - 0.055)
    dominant_colors = np.array([
        [9.1953745, 245.94664, 57.18312],  # 示例颜色 1
        [190.30363, 4.0078087, 250.1099],  # 示例颜色 2
        [9.534288, 3.9379532, 250.07562]   # 示例颜色 3
    ])
    dominant_colors = linear_to_display_p3(dominant_colors / 255.) * 255

    # 转为浮点数组以计算欧几里得距离
    pixels = env.reshape(-1, 3).astype(float)

    # 初始化掩码
    mask = np.zeros(pixels.shape[0], dtype=bool)

    # 对每个主要颜色计算距离并更新掩码
    scale = np.arange(0.1, 10.0, 0.05)
    for s in scale:
        for color in dominant_colors:
            mask |= np.linalg.norm(pixels*s - color, axis=1) < threshold
            
    # 对某个单色通道小于阈值的像素进行筛选
    # for c in range(3):
    #     mask |= (pixels[:,c] < threshold_rgb)

    # 将符合条件的像素置零
    pixels[mask] = 0

    # 重塑回原图像形状
    filtered_img_array = pixels.reshape(env.shape)
    return filtered_img_array


def quaternionToRotation(Q):
    # q = a + bi + cj + dk
    a = float(Q[0])
    b = float(Q[1])
    c = float(Q[2])
    d = float(Q[3])

    R = np.array([[2*a**2-1+2*b**2, 2*b*c+2*a*d,     2*b*d-2*a*c],
                  [2*b*c-2*a*d,     2*a**2-1+2*c**2, 2*c*d+2*a*b],
                  [2*b*d+2*a*c,     2*c*d-2*a*b,     2*a**2-1+2*d**2]])
    return np.transpose(R)

def load_camera(txt_path):
    c = 0
    with open(txt_path, 'r') as camParams: #相机内参.焦距/图像中心/图像尺寸
        for p in camParams.readlines():
            c += 1
            if c <= 3: # skip comments
                continue
            else:
                line = p.strip().split(' ')
                imgW, imgH = int(line[2]), int(line[3])
                fx = float(line[4])
                cxp, cyp = int(line[5]), int(line[6])
    fy = fx
    return imgW, imgH, fx, fy, cxp, cyp

def load_image(txt_path):
    paramDict = {}
    c = 0
    with open(txt_path, 'r') as camPoses:
        for cam in camPoses.readlines():
            c += 1
            if c <= 3: # skip comments
                continue
            elif c == 4:
                numImg = int(cam.strip().split(',')[0].split(':')[1])
                print('Number of images:', numImg)
            else:
                if c % 2 == 1:
                    line = cam.strip().split(' ')
                    R = quaternionToRotation(line[1:5])
                    paramDict['Rot'] = R #计算正交单位旋转矩阵
                    paramDict['Trans'] = np.array([float(line[5]), float(line[6]), float(line[7])]) #三维平移矢量
                    paramDict['Origin'] = -np.matmul(np.transpose(R), paramDict['Trans']) #外参，告诉世界坐标是怎样经过旋转和平移，然后落到另一个相机坐标上，包括相机的俯仰角。
                    paramDict['Target'] = R[2, :] + paramDict['Origin'] #相机视线方向
                    paramDict['Up'] = -R[1, :] #相机头顶方向
                    paramDict['cId'] = int(line[0]) #相机id
                    paramDict['imgName'] = line[-1] #相机id
    return paramDict

def load_ball(txt_path):
    ballLocDict = {}
    with open(txt_path, 'r') as locFile:
        for line in locFile.readlines():
            pos = line.strip().split(' ')
            # imgId top bottom left right
            ballLocDict[pos[0]] = [int(pos[1]), int(pos[2]), int(pos[3]), int(pos[4])]
    return ballLocDict

def get_sfm_params(param_id):
    envParamDict = {
        # 'sfm_4f/':   [0.95, 0.3, 0.1 , 0.8, 0.2, 0.3],
        # 'sfm_3127/': [0.95, 0.2, 0.35, 0.7, 0.2, 0.3]
        'sfm_4f/':   [0.95, 0.4, 0.1 , 0.8, 0.4, 0.5],
        'sfm_3127/': [0.95, 0.2, 0.35, 0.7, 0.4, 0.5]
        }

    if param_id == 0:
        param = envParamDict['sfm_4f/']
    elif param_id == 1:
        param = envParamDict['sfm_3127/']
    rTHout = param[0]
    rTHin = param[1]
    topTH = param[2]
    bottomTH = param[3]
    leftTH = param[4]
    rightTH = param[5]
    return rTHout, rTHin, topTH, bottomTH, leftTH, rightTH
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expDir', default='expDir')
    opt = parser.parse_args()
    expDir = opt.expDir
    imgFolder = os.path.join(expDir, "raw_images")
    
    print("The intermediate images are stored in tmp/pattern_detect. Please check these images and adjust the segementation if necessary.")

    os.makedirs(os.path.join(expDir, 'envmap'), exist_ok=True)
    os.makedirs(os.path.join(expDir, 'patterns'), exist_ok=True)
    for p in sorted(os.listdir(imgFolder)):
        imgball_path = os.path.join(imgFolder,p) 
        img_name = os.path.basename(imgball_path)
        img_nameno = os.path.splitext(os.path.basename(imgball_path))[0]
        if '.' == p[0] or os.path.exists(os.path.join(expDir, f'patterns/{img_nameno}.png')):
            continue
        balltxt = os.path.join(expDir, f'sparse/{img_nameno}_ballLoc.txt')
        imgtxt = os.path.join(expDir, f'sparse/{img_nameno}_images.txt')
        cameratxt = os.path.join(expDir,f'sparse/{img_nameno}_camera.txt')
        envSavePath = os.path.join(expDir,f'envmap/{img_nameno}_env.png')
        patternSavePath = os.path.join(expDir,f'patterns/{img_nameno}.png')
        
        resFolder = os.path.join(expDir, 'resources')
        os.makedirs(resFolder, exist_ok=True)
        outFolder = os.path.join(expDir, 'envmap')
        xmlFile  = "resources/misc/planarlc.xml"
        objs = "resources/misc/objs"
        cmd = f"cp -r {objs}/* {resFolder}/"
        os.system(cmd)
        
        tree = et.parse(xmlFile)
        root = tree.getroot()
        val = root.findall('default')[4]
        
        plane, mirrorball, lighting = root.findall('shape')
        plane.findall('string')[0].set('value', os.path.join(resFolder, 'plane.obj'))
        lighting.find('string').set('value', os.path.join(resFolder, 'ipad_real.obj'))
        pattern = root.findall('default')[5]
        pattern.set('value', os.path.join(resFolder, 'pattern.png'))
        assert(val.attrib['name'] == 'envPath')
        val.set('value', envSavePath)
        tree.write(os.path.join(outFolder, f'{img_nameno}.xml'))
        
        # print("Input the scale of current envmap:")
        # env_scale = int(input())
        showStepByStep = False
        checkFolder = os.path.join(expDir, 'checkEnv')
        if showStepByStep: os.makedirs(checkFolder, exist_ok=True)

        envH = 256
        envW = 512
        rTHout, rTHin, topTH, bottomTH, leftTH, rightTH = get_sfm_params(0)

        mirrorBallSize = 0.45

        # Read camera file
        imgW, imgH, fx, fy, cxp, cyp = load_camera(cameratxt)
        # Read images file
        paramDict = load_image(imgtxt)
        # Read ball file
        ballLocDict = load_ball(balltxt)

        # find mirror ball center找镜球的中心
        minSTD = np.inf
        ballx = []
        bally = []
        ballz = []
        rBallW = mirrorBallSize/2 # (m) * scale
        centerList = []
        for imgId, pos in ballLocDict.items():
            R = paramDict['Rot']
            t = paramDict['Trans']
            top, bottom, left, right = pos[0], pos[1], pos[2], pos[3]
            cxBall, cyBall = (right+left)/2/fx, (top+bottom)/2/fy # 球心归一化像素坐标
            cx, cy = cxp / fx, cyp / fy # 像面中心归一化像素坐标
            rxBall, ryBall = (right-left)/fx/2, (bottom-top)/fy/2, # 球半径归一化像素坐标
            dBallCenterImg = np.sqrt((cxBall-cx)**2 + (cyBall-cy)**2 + 1**2) # 球心深度
            # similar triangle
            dxBallCenterW = rBallW / rxBall * np.sqrt(rxBall**2+dBallCenterImg**2)
            dyBallCenterW = rBallW / ryBall * np.sqrt(ryBall**2+dBallCenterImg**2)
            dBallCenterW = (dxBallCenterW + dyBallCenterW) / 2
            ballCenter = np.array([(cxBall-cx), (cyBall-cy), 1]) / dBallCenterImg * dBallCenterW # in unit
            translatedCenter = np.matmul(np.transpose(R), ballCenter-t)
            centerList.append(translatedCenter)
            ballx.append(translatedCenter[0])
            bally.append(translatedCenter[1])
            ballz.append(translatedCenter[2])
        centers = np.stack(centerList, axis=0)
        std = np.std(centers, axis=0)
        std = np.sqrt(np.sum(std**2))
        if std < minSTD:
            minSTD = std
            scale = 1.0
            ballCenterW = np.mean(centers, axis=0)
        print('Mirror ball center: ', ballCenterW)
        print('Optimal scale:', scale)
        print('Creating env map for {}'.format(img_nameno))
        # After finding camera center, reconstruct envionment map
        # First step: find valid range of rays shooting from the camera
        rBallW = mirrorBallSize/2 * scale # (m) * scale
        envMap = np.zeros((envH*envW, 3))
        color_img = cv2.imread(imgball_path)[:,:,::-1]
        if color_img.shape[0] > 2000:
            color_img = cv2.resize(color_img, dsize=(0,0), fx=0.5, fy=0.5)

        R = paramDict['Rot']
        t = paramDict['Trans']
        C = paramDict['Origin']
        target = paramDict['Target']
        up = paramDict['Up']
        ballCenterC = np.matmul(R, ballCenterW) + t
        ballCenteru, ballCenterv = ballCenterC[0]/ballCenterC[2]*fx+cxp, ballCenterC[1]/ballCenterC[2]*fy+cyp  #球心坐标转到相机坐标系下再映射到图像坐标系
        # create a mask for valid pixels
        theta = np.arcsin(rBallW/np.sqrt(np.sum(ballCenterC**2)))
        rBallImg = fx * np.tan(theta) #球体以相关的角度映射到图形平面的半径
        u = np.arange(imgW)
        v = np.arange(imgH)
        uu, vv = np.meshgrid(u, v)
        m1 = ((uu-ballCenteru)**2 + (vv-ballCenterv)**2) < (rBallImg * rTHout)**2
        ball_img = color_img[int(ballCenterv-rBallImg * rTHout):int(ballCenterv+rBallImg * rTHout), int(ballCenteru-rBallImg * rTHout):int(ballCenteru+rBallImg * rTHout)]
        #遮罩修改
        maskOri = m1
        mask = np.reshape(maskOri, (imgW*imgH))
        dist = np.sqrt(((uu-ballCenteru)**2 + (vv-ballCenterv)**2))
        dist = np.reshape(dist, (imgW*imgH))[mask] # dist to ball center on each pixel
        # Compute view direction in world coord.计算视线方向在世界坐标系中的表示
        xx, yy = np.meshgrid(np.linspace(-1, 1, imgW), np.linspace(-1, 1, imgH))
        xx = np.reshape(xx * imgW/2/fx, (imgW*imgH))[mask] # 1-d
        yy = np.reshape(yy * imgH/2/fy, (imgW*imgH))[mask] # 1-d
        zz = np.ones((xx.size), dtype=np.float32) # 1-d
        v = np.stack([xx, yy, zz], axis=1).astype(np.float32) # m x 3
        v = v / np.maximum(np.sqrt(np.sum(v**2, axis=1))[:, np.newaxis], 1e-6)
        v = np.expand_dims(v, axis=2) # m x 3 x 1
        # Transform to world coord from cam params
        vW = -np.matmul(np.expand_dims(np.transpose(R), axis=0), v).squeeze(2) # m x 3

        # Compute normal by 3d geometry 计算球体表面上与视线方向交点的法线方向
        pa = np.sum(vW**2, axis=1) # m
        pb = np.sum(-2 * vW * (C - ballCenterW), axis=1) # m
        pc = np.sum((C - ballCenterW)**2) - rBallW**2 # 1
        root = (-pb - np.sqrt(pb**2-4*pa*pc))/(2*pa) # m
        normal = C - vW * np.expand_dims(root, axis=1) - ballCenterW # m x 3
        normal = normal / np.expand_dims(np.sqrt(np.sum(normal**2, axis=1)), axis=1) # m x 3
        normal_img = np.zeros((imgH*imgW,3), dtype=np.float32)
        normal_img[mask] = normal
        normal_img[~mask] = (color_img.reshape(-1, 3)[~mask] / 255)*2-1
        normal_img = normal_img.reshape(imgH, imgW, 3)

        # Reflection计算与视线方向和法线相关的反射方向
        cos_theta = 2 * np.sum(vW * (normal), axis=1) # m
        vWr = normal * np.expand_dims(cos_theta, axis=1) - vW # m x 3
        vWr = vWr / np.expand_dims(np.sqrt(np.sum(vWr**2, axis=1)), axis=1)

        inTH = 2
        nS = 128
        sample_vecs = sample_cone_vectors(vWr, inTH, nS)
        vWr = sample_vecs.reshape(-1, 3)
        # Compute angle from reflected directions to env (u,v)
        theta = np.arccos(vWr[:, 2])
        phi = np.arctan2(vWr[:, 1], vWr[:, 0])
        envu = 1 - phi / (2 * np.pi) + 0.25
        flag = envu > 1.0
        envu[flag] = envu[flag] - 1.0
        envv = theta / np.pi

        envv = (envv * envH).astype(np.int64)
        envu = (envu * envW).astype(np.int64)

        # 原始数据
        envidx = envv * envW + envu  # 索引
        color = np.reshape(color_img, (imgW * imgH, 3))[mask, :]  # m x 3
        color = np.tile(color[:, None], (1, nS, 1)).reshape(-1, 3)  # m x 3 (扩展后的颜色)

        # 初始化数组
        count = np.bincount(envidx, minlength=(envH * envW))  # 每个索引的累积计数
        envMap = np.zeros((envH * envW, 3), dtype=np.float64)  # 累积颜色值

        # 使用 np.add.at 累加颜色到 envMap
        np.add.at(envMap, envidx, color)  # 按索引累加颜色值

        # 计算每个位置的平均颜色
        valid = count > 0  # 防止除零
        envMap[valid] /= count[valid, None]  # 归一化，避免无效索引的影响

        # envidx, uniqueIdx  = np.unique(envidx, return_index=True) # m2
        # color = color[uniqueIdx, :].astype(envMap.dtype)# m2 x 3
        # envMap[envidx] = color

        #保存处理结果和生成环境贴图的图片
        if showStepByStep == True:
            # check correct range on ball
            checkIm = plt.imread(imgFolder+'/'+paramDict['imgName'])
            maskedIm = checkIm*np.logical_not(np.expand_dims(maskOri, axis=2))
            plt.imsave(os.path.join(checkFolder, 'ballMask_{}.png'.format(imgId)), maskedIm)
            # show current env map
            envMap0 = np.reshape(envMap, (envH, envW, 3))
            plt.imsave(os.path.join(checkFolder, 'env_{}.png'.format(imgId)), envMap0.astype(np.uint8))
            
            cv2.imwrite(os.path.join(checkFolder, 'normal_{}.png'.format(imgId)), ((normal_img[:,:,::-1]/2+0.5)*255).astype(np.uint8))

        envMap = np.reshape(envMap, (envH, envW, 3))
        import torch
        from utils.env_utils import mask, mask_hemisphere, get_pattern
        envMap = mask_hemisphere(torch.from_numpy(envMap).permute(2,0,1).contiguous())

        cfg = {
            "polygon": [
                [-3.375, -2, 3.3919],
                [0.6250, -2, 3.3919],
                [0.6250, 2, 3.3919],
                [-3.375, 2, 3.3919],
            ],
            "ball_center": [1.375, 0.0, 0.0],
            "ball_radius": 0.225
        }
        os.makedirs("tmp/pattern_detect", exist_ok=True)
        
        
        while True:
            pattern = get_pattern(envMap, cfg)
            envMap_masked, inlight = mask(envMap.clone(), cfg, ret_mask=True) #* Mask the active lighting area in the env map
            envMap_masked = envMap_masked.permute(1,2,0).contiguous().numpy()
            inlight = inlight.permute(1,2,0).contiguous().numpy()
            
            plt.imsave("tmp/pattern_detect/env.png", envMap_masked.clip(0.0, 255.0).astype(np.uint8))
            plt.imsave("tmp/pattern_detect/pattern.png", pattern.clip(0.0, 255.0).astype(np.uint8))

            # 用户输入
            user_input = input("Enter 'w', 'a', 's', 'd', 'q', 'e', 'z', 'x', 'c', 'v', or press 'enter' to quit: ").strip().lower()
            
            if user_input == 'ok':
                break
            elif user_input == 's':
                cfg['polygon'] = adjust_polygon_center(cfg['polygon'], 0, 0.1, 0)
            elif user_input == 'w':
                cfg['polygon'] = adjust_polygon_center(cfg['polygon'], 0, -0.1, 0)
            elif user_input == 'd':
                cfg['polygon'] = adjust_polygon_center(cfg['polygon'], -0.1, 0, 0)
            elif user_input == 'a':
                cfg['polygon'] = adjust_polygon_center(cfg['polygon'], 0.1, 0, 0)
            elif user_input == 'q':
                cfg['polygon'] = adjust_polygon_center(cfg['polygon'], 0, 0, 0.1)
            elif user_input == 'e':
                cfg['polygon'] = adjust_polygon_center(cfg['polygon'], 0, 0, -0.1)
            elif user_input == 'z':
                cfg['polygon'] = adjust_polygon_dir(cfg['polygon'], 0.1, 0)
            elif user_input == 'x':
                cfg['polygon'] = adjust_polygon_dir(cfg['polygon'], -0.1, 0)
            elif user_input == 'c':
                cfg['polygon'] = adjust_polygon_dir(cfg['polygon'], 0, 0.1)
            elif user_input == 'v':
                cfg['polygon'] = adjust_polygon_dir(cfg['polygon'], 0, -0.1)
            else:
                print("Invalid input. Please enter 'w', 'a', 's', 'd', 'q', 'e', 'z', 'x', 'c', 'v', or press 'ok' to quit.")

        # envMap_masked = mask_env_map_with_color(envMap_masked, 120)
        
        plt.imsave(envSavePath, envMap_masked.clip(0.0, 255.0).astype(np.uint8))
        plt.imsave(patternSavePath, pattern.clip(0.0, 255.0).astype(np.uint8))

        print('Environment map has been created!!!')
