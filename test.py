from os import path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.utils import make_exp_dirs, parse, set_random_seed, build_dataloader

import argparse
from models.mat_model import MAT_model
from models.denoise_model import Denoise_model
from data.brdf_dataset import MatSynth3Dataset, MatSynthRealImage
from copy import deepcopy

def build_model(opt):
    opt = deepcopy(opt)
    model = MAT_model(opt) if opt['network_g']['type'] != 'Denoise' else Denoise_model(opt)
    return model

def build_dataset(dataset_opt):
    dataset_opt = deepcopy(dataset_opt)
    if not dataset_opt.get('is_real', False):
        dataset = MatSynth3Dataset(dataset_opt)
    else: 
        dataset = MatSynthRealImage(dataset_opt)
    return dataset

def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=is_train)
    
    opt['dist'] = False

    opt['rank'], opt['world_size'] = 0, 1

    # random seed
    seed = opt.get('manual_seed', 10)
    set_random_seed(seed + opt['rank'])

    return opt, args.opt

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, opt_path = parse_options(root_path, is_train=False)
    test_model(opt)

def test_model(opt):
    # mkdir and initialize loggers
    make_exp_dirs(opt)
    
    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)
