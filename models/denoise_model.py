from models.base_model import BaseModel
from utils.utils import imwrite,tensor2img
import torch

import os.path as osp
from copy import deepcopy
from tqdm import tqdm
from archs.naf_arch import NAFNet
from collections import OrderedDict

def build_network(opt):
    opt = deepcopy(opt)
    net = NAFNet(**opt)
    return net

class Denoise_model(BaseModel):
    """Mat_model."""
    def __init__(self, opt):
        self.use_normal_lc = opt['network_g'].get('use_normal_lc', False) # [False, 'sg', 'splitsum']
        self.vis_lightclues = opt['network_g'].get('vis_lightclues', False)
        self.visSavePath = opt['val'].get('savePath', None)
        super(Denoise_model, self).__init__(opt)
        self.input_pattern = opt['network_g'].get('input_pattern', False)
        self.real_input = opt['network_g'].get('real_input', False)
        self.initNetworks(opt.get('print_net',True))

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        # logger.info(f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
            if k.startswith('step_counter'):
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

    def buildNet(self, net_opt, path_key):
        net = build_network(net_opt)
        net = self.model_to_device(net)

        load_path = self.opt['path'].get('pretrain_network_' + path_key, None)
        # load pretrained models
        if load_path is not None:
            params = 'params'
            self.load_network(net, load_path, self.opt['path'].get('strict_load_'+path_key, True), params)
        return net
    def initNetworks(self, printNet=True):
        self.net_g = self.buildNet(self.opt.get('network_g'), 'g')
        
    def feed_data(self, data):
        self.inputs = data.get('inputs', None)
        if self.inputs is not None:
            self.inputs = self.inputs.to(self.device)
        self.lightclues = data.get('envlc_img', None)
        if self.lightclues is not None: self.lightclues = self.lightclues.to(self.device)
        self.pattern = data.get('pattern', None)
        if self.pattern is not None:
            self.pattern = self.pattern.to(self.device)

    def get_inputs(self):
        self.gt = None
        tex = self.tex.clone().unsqueeze(0).broadcast_to(self.inputs.shape) if self.pattern is None else self.pattern
        pattern = tex * 2 - 1
        inputs = self.inputs if not self.input_pattern else torch.cat([self.inputs, pattern], dim=1)
        return inputs

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            b, c, h, w = self.inputs.shape
            inputs = self.get_inputs()
            if self.use_normal_lc:
                output, self.mid_normal, self.lightclues = self.net_g(inputs, self.env, self.env_render, self.view_pos, self.lightclues)
            else:
                output = self.net_g(inputs)
            self.output = output
        self.net_g.train()

    def save_pre_visuals(self, path, pred):
        output = (pred / 2 + 0.5) ** 0.4545

        output_img = tensor2img(output,rgb2bgr=True)
        imwrite(output_img, path, float2int=False)

    def get_current_visuals(self, pred, gt):
        out_dict = OrderedDict()
        out_dict['predsvbrdf'] = pred.detach()
        if gt is not None:
            out_dict['gtsvbrdf'] = gt.detach()
        else:
            out_dict['gtsvbrdf'] = pred.detach()
        if hasattr(self, 'brdf'):
            out_dict['brdf'] = self.brdf.detach()
        return out_dict

    
    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        if self.opt.get('pbar',True):
            pbar = tqdm(total=len(dataloader), unit='image')
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            if self.opt['val'].get('save_img', False):
                results = self.get_current_visuals(self.output, self.gt)
                if self.visSavePath is not None:
                    save_path = self.visSavePath
                else:
                    save_path = osp.join(self.opt['path']['visualization'], dataset_name)
                brdf_path=osp.join(save_path, val_data['name'][0])
                self.save_pre_visuals(brdf_path, results['predsvbrdf'])

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    error = torch.abs((self.output / 2 + 0.5)-self.gt).mean()*opt_.pop('weight')
                    self.metric_results[name] += error 
            torch.cuda.empty_cache()
            if self.opt.get('pbar',True):
                pbar.update(1)
                pbar.set_description(f'Testing')
                # break
        if self.opt.get('pbar',True):
            pbar.close()