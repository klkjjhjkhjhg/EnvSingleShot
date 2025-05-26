import numpy as np
import cv2
import os
from tqdm import tqdm
import mitsuba as mi
import argparse
mi.set_variant('cuda_ad_rgb')
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
np.random.seed(10)
class MiBatch:
    def __init__(self, env_root, save_root, gamma=False) -> None:
        files = sorted(os.listdir(env_root))
        names = [name for name in files if name.endswith('.png') and '.' != name[0]]
        xmls = [name for name in files if name.endswith('.xml') and '.' != name[0]]
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok=True)
        # env_paths = sorted([os.path.join(env_folder, f) for f in os.listdir(env_folder) if f.endswith('.png') and os.path.isfile(os.path.join(env_folder, f))])     
        self.env_datas = {}
        for name in names:
            env_data = self._load_env_data(os.path.join(env_root, name))
            self.env_datas[name] = env_data
        self.env_names = names
        self.scene = mi.load_file(os.path.join(env_root, xmls[0]))
        self.params = mi.traverse(self.scene)
        self.params['lighting.emitter.radiance.data'] = self.params['lighting.emitter.radiance.data'] * 8
        self.light = self.params['lighting.emitter.radiance.data'].numpy().copy()
        self.mirror_scene = mi.load_file("resources/misc/MirrorPlane.xml")
        self.mirror_params = mi.traverse(self.mirror_scene)
        self.gamma = gamma
        self.albedo = 0.6
        self.render_log = {} 

    def _load_env_data(self, env_path):
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

        return env_data
   
    def renderconstlcmaterial(self,env_path):
        all_images = []
        for d, a in [(0.0, 0.04), (0.2, 0.04), (0.6, 0.04), (0.0, 0.36), (0.2, 0.36), (0.6, 0.36)]:
            s = round(self.albedo - d, 1)
            self.params['plane.bsdf.bsdf_0.reflectance.value'] = d
            self.params['plane.bsdf.bsdf_1.alpha.value'] = a
            self.params['plane.bsdf.bsdf_1.specular_reflectance.value'] = s
            self.params.update()
            img = mi.render(self.scene,params=self.params).numpy()
            img1 = img[:,:256]
            all_images.append(img1)
        first_row = np.concatenate((all_images[0], all_images[1], all_images[2]), axis=1)  # 第一行 a=0.04
        second_row = np.concatenate((all_images[3], all_images[4], all_images[5]), axis=1)  # 第二行 a=0.36
        concatenated_image = np.concatenate((first_row, second_row), axis=0)
        return concatenated_image

    def scale_light(self, x, k, m):
        scale = np.mean(x, axis=-1, keepdims=True) ** k * m
        return x * scale

    def renderconstlc(self,env_path):
        env_name = os.path.basename(env_path)
        self.params['EnvironmentMapEmitter.data'] = self.scale_light(self.env_datas[env_name], 2, 50) #环境光
        self.params['lighting.emitter.radiance.data']  = self.params['lighting.emitter.radiance.data'] * 0
        Lc = self.renderconstlcmaterial(env_path)

        if self.gamma:
            Lc = Lc ** (1 / 2.2)
        Lc = np.clip(Lc, 0, 1) * 255
        name = os.path.splitext(os.path.basename(env_path))[0]
        if ".png" not in name:
            name = name + ".png"
        self.render_log[name] = env_name
        cv2.imwrite(os.path.join(self.save_root, name), Lc[:, :, ::-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expDir', default='expDir')
    opt = parser.parse_args()
    expDir = opt.expDir
    env_folder = os.path.join(expDir, "envmap")
    save_folder = os.path.join(expDir, "plcs")
    
    render = MiBatch(env_folder,save_folder, gamma=True)  
    env_paths = sorted([os.path.join(env_folder, f) for f in os.listdir(env_folder) if f.endswith('.png') and os.path.isfile(os.path.join(env_folder, f)) and '.' != f[0]])
    env_data_dict = {}
    for env_path in tqdm(env_paths, desc='pLCs'):
        render.renderconstlc(env_path)
        
            
