import shutil
import numpy as np
import os
import random
import time
import torch
from os import path as osp
import yaml
from collections import OrderedDict
import cv2

from torchvision.utils import make_grid
import math


def torch_cross(a, b, dim=-3):
    if dim == -3 or dim == 0:
        x = a[...,1,:,:] * b[...,2,:,:] - a[...,2,:,:]*b[...,1,:,:]
        y = a[...,2,:,:] * b[...,0,:,:] - a[...,0,:,:]*b[...,2,:,:]
        z = a[...,0,:,:] * b[...,1,:,:] - a[...,1,:,:]*b[...,0,:,:]
    elif dim == -1:
        x = a[...,1] * b[...,2] - a[...,2]*b[...,1]
        y = a[...,2] * b[...,0] - a[...,0]*b[...,2]
        z = a[...,0] * b[...,1] - a[...,1]*b[...,0]
    return torch.stack([x,y,z], dim=dim)
def scandir(dir_path, suffix=None, recursive=False, full_path=False, getDirs=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file() or getDirs:
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def torch_norm(arr, dim=1, norm=True, square=False, keepdims=True, eps=1e-12):
    length = torch.norm(arr, dim = dim, keepdim=keepdims)
    if square:
        length = torch.square(length)
    if norm:
        return arr / (length + eps)
    else:
        return length

def paths_from_folder(folder, suffix=None, getDirs=False):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder, suffix=suffix, getDirs=getDirs))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def img2tensor(imgs, bgr2rgb=True, float32=True, normalization=False, singleChannel = False):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if (not singleChannel):
            if (img.shape[2] == 3 and bgr2rgb ):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
        else:
            img = torch.from_numpy(img)
        if float32:
            img = img.float()
        if normalization:
            img = img/255.0
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def imfrombytes(content, flag='color', float32=False, bgr2rgb=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if bgr2rgb:
        img=img[:,:,::-1]
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def build_dataloader(dataset, dataset_opt):
    batch = 1 if dataset_opt.get('len', None) is None else dataset_opt['len']
    dataloader_args = dict(dataset=dataset, batch_size=batch, shuffle=False, num_workers=1)

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    return torch.utils.data.DataLoader(**dataloader_args)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        if 'DEBUG' in path:
            shutil.rmtree(path)
            print(f'Path already exists. In DEBUG mode, delete it', flush=True)
        else:
            new_name = path + '_archived_' + get_time_str()
            print(f'Path already exists. Rename it to {new_name}', flush=True)
            os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    
def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        if opt['val'].get('savePath', None) is None:
            mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if ('strict_load' not in key) and ('pretrain_network' not in key) and ('resume' not in key) and ('Pattern' not in key):
            os.makedirs(path, exist_ok=True)
    if opt['val'].get('savePath', None) is not None:
        mkdir_and_rename(opt['val'].get('savePath'))
    if opt['val'].get('inputSavePath', None) is not None:
        mkdir_and_rename(opt['val'].get('inputSavePath'))

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, root_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # if 'name' in opt and 'name' not in opt['network_g']:
    #     opt['network_g']['name'] = opt['name']
    # datasets
    if opt.get('datasets', None) is not None:
        for phase, dataset in opt['datasets'].items():
            # for several datasets, e.g., test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt and 'scale' not in dataset:
                dataset['scale'] = opt['scale']
            if 'brdf_args' in opt and 'brdf_args' not in dataset:
                dataset['brdf_args'] = opt['brdf_args']
            if 'fix_hr' in opt and 'fix_hr' not in dataset:
                dataset['fix_hr'] = opt['fix_hr']
            if dataset.get('dataroot_gt') is not None:
                dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
            if dataset.get('dataroot_lq') is not None:
                dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    if root_path is not None:
        opt['path']['root'] = root_path
    else:
        opt['path']['root'] = osp.abspath(
            osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')
        opt['path']['options'] = osp.join(experiments_root,'options')

        # change some options for debug mode
        if 'debug' in opt['name'].lower():
            if 'val' in opt:
                opt['val']['val_freq'] = 50 if 'noval' not in opt['name'].lower() else 100000
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 50
            opt['logger']['use_tb_logger'] = False
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
        opt['path']['options'] = osp.join(results_root, 'options')

    return opt


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 固定哈希算法种子
    # os.environ['PYTHONHASHSEED'] = str(seed)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1), gamma=False):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if gamma:
            img_np = img_np**0.4545
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def imwrite(img, file_path, params=None, auto_mkdir=True, float2int=False):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    if float2int:
        img = (img*255).astype(np.uint8)
    return cv2.imwrite(file_path, img, params)


def torch_dot(a,b, dim=-3, keepdims=True):
    return torch.sum(a*b,dim=dim,keepdims=keepdims)

def toLDR_torch(img, gamma=True, sturated=True):
    if gamma:
        img = img**0.4545
    if sturated:
        img = torch.clip(img,0,1)
    img = (img*255).int()
    return img

def toHDR_torch(img, gamma=True):
    img = img.float()/255
    if gamma:
        img = img ** 2.2
    return img
