import os
import numpy as np
from torch.utils import data as data
import torch,json, pickle
from utils.utils import imfrombytes, img2tensor, paths_from_folder
from utils.fiel_client import FileClient

class matBaseDataset(data.Dataset):
    """
    Read SVBRDF data from dataset.
    the image is (n*Img), N, D, R, S
    """
    def __init__(self, opt):
        super(matBaseDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        data_folder = opt['data_path']
        json_path = opt.get('json_path', None)
        if json_path is not None:
            with open(json_path, 'r') as f:
                paths = json.load(f)
                self.data_paths = [os.path.join(data_folder, path) for path in paths]
        else:
            self.data_paths = sorted(paths_from_folder(data_folder))

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
    
    def __getitem__(self, index):
        img_path = self.data_paths[index]
        img_bytes = self.file_client.get(img_path, 'brdf')
        img = imfrombytes(img_bytes, float32=True, bgr2rgb=True)
        n = img.shape[1] // img.shape[0]
        svbrdf = np.split(img, n, axis=1)[-4:] # n, d ,r, s
        r = svbrdf[2].mean(-1, keepdims=True)
        svbrdf = np.concatenate([svbrdf[0], svbrdf[1], r, svbrdf[3]], axis=-1) * 2 - 1
        svbrdf = img2tensor(svbrdf, bgr2rgb=False, float32=True, normalization=False)
        batch = {
            'svbrdfs': svbrdf,
            'path': img_path
        }
        return batch

    def __len__(self):
        return len(self.data_paths)

class MatSynth3Dataset(matBaseDataset):#svbrdf,渲染图,lc
    def __init__(self,opt):
        super(MatSynth3Dataset,self).__init__(opt)
        #渲染的图片路径
        render_json_path = opt.get('renderjson_path', None)
        if render_json_path is not None:
            with open(render_json_path, 'r') as f:
                self.render_mapping = json.load(f)
        else:
            raise ValueError("JSON file with render mappings is required but not provided.")
        # 提取所有PNG路径和环境光路径
        self.renderdata_paths = list(self.render_mapping.keys())
        renderfolder_path = opt['renderdata_path']
        self.render_paths = [os.path.join(renderfolder_path, path) for path in self.renderdata_paths]
        self.envlcdata_paths = [os.path.splitext(path)[0] for path in self.render_mapping.values()]

        #* Load lighting clues of environment light
        envlcfolder_path = opt.get('envlc_path', None)
        self.use_lc = (envlcfolder_path is not None)
        if self.use_lc:
            all_files = os.listdir(envlcfolder_path)
            self.envlcfolder_names = [os.path.splitext(file)[0] for file in all_files if os.path.isfile(os.path.join(envlcfolder_path, file))]   
            self.envlc_paths = [ # 将渲染的图片对应的key环境光名字和lc文件夹中lc名字保持一致并提取出来
                    os.path.join(envlcfolder_path, os.path.basename(path)+".png")
                    for path in self.envlcdata_paths if path in self.envlcfolder_names
                ]
        #* Load SG coefficients of enviroment lights
        envsgfolder_path = opt.get('envsg_path', None)
        self.use_sg = (envsgfolder_path is not None)
        if self.use_sg:
            env_paths = [os.path.splitext(path)[0] for path in self.render_mapping.values()]
            self.envsg_paths = [
                    os.path.join(envsgfolder_path, os.path.basename(path)+".pth")
                    for path in env_paths
                ]
        #* Load all enviroment mip maps
        self.envmip_path = opt.get('envmip_path', None)
        self.use_envmip = (self.envmip_path is not None)
        if self.use_envmip:
            self.mip_lod = opt.get('mip_lod', 8)
            self.envmips = {}
            for path in set(self.render_mapping.values()):
                filename, ext = os.path.splitext(path)
                self.envmips[filename] = []
                envmip_path = os.path.join(self.envmip_path, filename+".bin")                
                with open(envmip_path, 'rb') as f:
                    mips = pickle.load(f)
                self.envmips[filename].extend(mips)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        basename = os.path.basename(batch['path'])
        batch["name"] = basename

        render_img_path = self.render_paths[index]
        render_img_bytes = self.file_client.get(render_img_path, 'render_args')
        render_img = imfrombytes(render_img_bytes, float32=True)        
        first_image, active_image, env_img, fourth_image = np.split(render_img, 4, axis=1)
        first_image = img2tensor(first_image, bgr2rgb=True, float32=True) # captured image in sRGB color space
        active_image = img2tensor(active_image, bgr2rgb=True, float32=True) # captured image in sRGB color space
        env_img = img2tensor(env_img, bgr2rgb=True, float32=True) # captured image in sRGB color space
        fourth_image = img2tensor(fourth_image, bgr2rgb=True, float32=True) # sphere lighting in sRGB color space
        first_image = first_image ** 2.2 # captured image in linear color space
        # first_image = ((active_image ** 2.2) + (env_img ** 2.2))

        inputs = torch.cat((first_image,fourth_image), dim=0) if self.opt.get('cat_ball',True) else first_image       

        if self.use_lc:
            envlc_path = self.envlc_paths[index]
            envlc_bytes = self.file_client.get(envlc_path, 'envlc_args')
            envlc = imfrombytes(envlc_bytes, float32=True)
            rows, cols = 2, 3
            cell_size = 256
            split_images = []
            for row in range(rows):# 分割图片
                for col in range(cols):
                    start_y, end_y = row * cell_size, (row + 1) * cell_size
                    start_x, end_x = col * cell_size, (col + 1) * cell_size
                    split_img = envlc[start_y:end_y, start_x:end_x]
                    split_img = img2tensor(split_img, bgr2rgb=True, float32=True)
                    split_img = split_img ** 2.2
                    split_images.append(split_img)
            envlc_imgs = torch.cat(split_images, dim=0) # lighting clues in linear color space 
            inputs = torch.cat((inputs,envlc_imgs),dim=0) if self.opt.get('cat_envlc',True) else inputs
            batch['envlc'] = envlc_imgs * 2 - 1 # processed to (-1, 1)
            batch['envlc_img'] = envlc_imgs
            batch['envlc_path'] = envlc_path
        if self.use_envmip:
            key = self.render_mapping[os.path.basename(render_img_path)]
            for i in range(self.mip_lod+1):
                # read env lighting mipmaps (linear color space)
                batch['envmips_'+str(i)] = self.envmips[os.path.splitext(key)[0]][i]    
            batch['mip_lod'] = self.mip_lod

        batch['inputs'] = inputs * 2 - 1 # processed to (-1, 1)
        batch['render_img'] = inputs
        batch['render_path'] = render_img_path
        batch['env_img'] = env_img
        batch['act_img'] = active_image
        return batch

    def __len__(self):
        return len(self.data_paths)
       
class MatSynthRealImage(matBaseDataset):
    def __init__(self,opt):
        super(MatSynthRealImage).__init__()
        self.opt = opt
        self.dataFolder = opt['data_path']
        self.data_paths = sorted(paths_from_folder(self.dataFolder, getDirs=True))
        envlcfolder_path = opt.get('envlc_path', None)
        self.use_lc = (envlcfolder_path is not None)

        #* Load lighting clues of environment light
        self.envlcfolder_path = opt.get('envlc_path', None)
        self.use_lc = (self.envlcfolder_path is not None)
        self.pattern_path = opt.get('pattern_path', False)

        #* Load enviroment mip maps
        self.envmip_path = opt.get('envmip_path', None)
        self.use_envmip = (self.envmip_path is not None)
        self.envmips = {}
        self.envnames = []
        if self.use_envmip:
            self.mip_lod = opt.get('mip_lod', 8)
            all_files = sorted(os.listdir(self.envmip_path))
            self.envmips_names = [os.path.splitext(file)[0].split('_')[0] for file in all_files if file.endswith('.bin')] 
            self.envmips_paths = [ 
                    os.path.join(self.envmip_path, file_name +"_env.bin")
                    for file_name in [os.path.splitext(os.path.basename(path))[0] for path in self.data_paths] 
                    if file_name in self.envmips_names
                ]
            # for path in self.envmips_paths:   
            #     filename, ext = os.path.splitext(path)
            #     self.envnames.append(filename)
            #     self.envmips[filename] = []
            #     self.envmip_path = os.path.join(self.envmip_path, filename +".bin")
            #     with open(self.envmip_path, 'rb') as f:
            #         mips = pickle.load(f)
            #     self.envmips[filename].extend(mips)

        self.methodIndex = opt.get('methodIndex', 0)
        self.io_backend_opt = opt['io_backend']
        self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        self.pdfResizeTo = opt.get('pdfResizeTo', None)
        self.n_img = opt.get('n_img', 5)
        self.n_skip = opt.get('n_skip', 6)
        self.oriFolder = opt.get('ori_path', None)
        if self.oriFolder is not None: self.oriFolder = sorted(paths_from_folder(self.dataFolder, getDirs=True))

    def __getitem__(self,index):
        batch = {}

        render_img_path = self.data_paths[index]
        batch['name'] = os.path.basename(render_img_path)
        name = os.path.splitext(batch['name'])[0]
        render_img_bytes = self.file_client.get(render_img_path, 'render_args')
        render_img = imfrombytes(render_img_bytes, float32=True)        
        render_img = img2tensor(render_img, bgr2rgb=True, float32=True) # env lighting contribution in captured image in sRGB color space
        render_img = render_img ** 2.2 # captured image in linear color space

        inputs = render_img
        
        if self.pattern_path:
            pattern_path = os.path.join(self.pattern_path, name+'.png')
            envlc_bytes = self.file_client.get(pattern_path, 'patterns')
            pattern = imfrombytes(envlc_bytes, float32=True)
            pattern = img2tensor(pattern, bgr2rgb=True, float32=True)
            batch['pattern'] = pattern

        if self.use_lc:
            envlc_path = os.path.join(self.envlcfolder_path, name + '_env.png')
            envlc_bytes = self.file_client.get(envlc_path, 'envlc_args')
            envlc = imfrombytes(envlc_bytes, float32=True)
            rows, cols = 2, 3
            cell_size = 256
            split_images = []
            for row in range(rows):# 分割图片
                for col in range(cols):
                    start_y, end_y = row * cell_size, (row + 1) * cell_size
                    start_x, end_x = col * cell_size, (col + 1) * cell_size
                    split_img = envlc[start_y:end_y, start_x:end_x]
                    split_img = img2tensor(split_img, bgr2rgb=True, float32=True)
                    split_img = split_img ** 2.2
                    split_images.append(split_img)
            envlc_imgs = torch.cat(split_images, dim=0) # lighting clues in linear color space 
            inputs = torch.cat((inputs,envlc_imgs),dim=0) if self.opt.get('cat_envlc',True) else inputs
            batch['envlc'] = envlc_imgs * 2 - 1 # processed to (-1, 1)
            batch['envlc_img'] = envlc_imgs
            batch['envlc_path'] = envlc_path

        if self.use_envmip:
            mipmap_path = os.path.join(self.envmip_path, name+'_env.bin')                
            with open(mipmap_path, 'rb') as f:
                mips = pickle.load(f)
            for i in range(self.mip_lod+1):
                batch['envmips_'+str(i)] = mips[i]    
            batch['mip_lod'] = self.mip_lod
        
        if self.oriFolder is not None:
            ori_img_path = self.oriFolder[index]
            ori_img_bytes = self.file_client.get(ori_img_path, 'ori_args')
            ori_img = imfrombytes(ori_img_bytes, float32=True)
            ori_img = img2tensor(ori_img, bgr2rgb=True, float32=True)
            batch['ori_img'] =ori_img

        batch['inputs'] = inputs * 2 - 1 # processed to (-1, 1)

        return batch

    def __len__(self):
        return len(self.data_paths)
