from models.base_model import BaseModel
from utils.utils import imwrite,tensor2img, torch_norm
import torch, os
from utils.render_util import EnvRender
from copy import deepcopy
from tqdm import tqdm
import os.path as osp
from archs.matnaf_arch import LCNet
from collections import OrderedDict

def build_network(opt):
    opt = deepcopy(opt)
    net = LCNet(**opt)
    return net

class MAT_model(BaseModel):
    """Mat_model."""
    def __init__(self, opt):
        self.use_normal_lc = opt['network_g'].get('use_normal_lc', 'splitsum') # [False, 'sg', 'splitsum']
        self.vis_lightclues = opt['network_g'].get('vis_lightclues', False)
        self.vis_mirrorball = opt['network_g'].get('vis_mirrorball', False)
        self.input_origin = opt['network_g'].get('input_origin', False)
        self.input_pattern = opt['network_g'].get('input_pattern', False)
        self.real_input = opt['network_g'].get('real_input', False)
        self.brdf_args = opt['network_g'].pop('brdf_args')
        self.visSavePath = opt['val'].get('savePath', None)
        self.visInputSavePath = opt['val'].get('inputSavePath', None)
        super(MAT_model, self).__init__(opt)
        self.initRender()
        self.initNetworks(opt.get('print_net',True))
        if self.is_train:
            self.init_training_settings()

    def initNetworks(self, printNet=True):
        super().initNetworks(printNet)
        if self.jitter_input: self.net_d = self.buildNet(self.opt.get('network_d'), 'd', printNet)
    
    def feed_data(self, data):
        self.svbrdf = data.get('svbrdfs', None)
        if self.svbrdf is not None:
            self.svbrdf = self.svbrdf.to(self.device)
        self.inputs = data.get('inputs', None)
        if self.inputs is not None:
            self.inputs = self.inputs.to(self.device)

        self.lightclues = data.get('envlc_img', None)
        if self.lightclues is not None: self.lightclues = self.lightclues.to(self.device)
        
        if self.use_normal_lc == 'sg':
            env_sg = data.get('envsg', None)
            if env_sg is not None:
                self.env = env_sg.to(self.device)
        if self.use_normal_lc == 'splitsum':
            self.mip_lod = data.get('mip_lod', None)
            assert self.mip_lod.float().mean() == self.brdf_args.get('nLod', 8)
            if self.mip_lod is not None:
                self.env = []
                for i in range(self.mip_lod[0]+1):
                    self.env.append(data['envmips_'+str(i)].permute(0, 2, 3, 1).contiguous())
        
        if self.svbrdf is not None:
            self.env_contrib = data.get("env_img", None).to(self.device)
            self.act_contrib = data.get("act_img", None).to(self.device)
        
        self.pattern = data.get('pattern', None)
        if self.pattern is not None:
            self.pattern = self.pattern.to(self.device)
        
        if self.real_input and self.input_origin:
            self.origin = data.get('ori_img', None).to(self.device)

    def initRender(self):
        self.env_render = EnvRender(self.brdf_args, device=self.device) #! all tensor are calculated on gpu in default
        self.view_pos = torch.tensor([1.375,0,3.3919]).to(self.device)

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
        if self.input_origin and 'LCNet' in str(net.__class__) and not self.real_input:
            intro_weight = load_net.pop('intro.weight')
            intro_bias = load_net.pop('intro.bias')
            zeros_kernel = torch.zeros_like(intro_weight)
            intro_weight = torch.cat([intro_weight, zeros_kernel, zeros_kernel], dim=1)
            intro_dict = {'intro.weight': intro_weight, 'intro.bias': intro_bias}
            load_net.update(intro_dict)
        net.load_state_dict(load_net, strict=strict)

    def get_inputs(self):
        if self.real_input:
            tex = self.tex.clone().unsqueeze(0).broadcast_to(self.inputs.shape) if self.pattern is None else self.pattern
            pattern = tex * 2 - 1
            
            inputs = self.inputs if not self.input_origin else torch.cat([self.inputs, self.origin * 2 - 1], dim=1)
            inputs = inputs if not self.input_pattern else torch.cat([inputs, pattern], dim=1)
            return inputs
        else:
            return self.inputs
    
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            b, c, h, w = self.inputs.shape
            inputs = self.get_inputs()
            if self.use_normal_lc:
                output, self.mid_normal, self.lightclues = self.net_g(inputs, self.env, self.env_render, self.view_pos, self.lightclues)
            else:
                output = self.net_g(inputs)
            normal, diffuse,roughness,specular = torch.split(output,[3,3,1,3],dim=1)
            normal = torch_norm(normal,dim=1)
            self.output = torch.cat([normal, diffuse, roughness, specular],dim=1)
        self.net_g.train()

    def initNetworks(self, printNet=True):
        self.net_g = self.buildNet(self.opt.get('network_g'), 'g')
        
    def buildNet(self, net_opt, path_key):
        net = build_network(net_opt)
        net = self.model_to_device(net)

        load_path = self.opt['path'].get('pretrain_network_' + path_key, None)
        # load pretrained models
        if load_path is not None:
            params = 'params'
            self.load_network(net, load_path, self.opt['path'].get('strict_load_'+path_key, True), params)
        return net
        
    def computeLoss(self, output, loss_dict, mid_normal=None, isDesc=False):
        normal, diffuse,roughness,specular = torch.split(output,[3,3,1,3],dim=1)
        normal = torch_norm(normal,dim=1)
        l_total = 0
        if self.cri_nlc is not None:
            l_midn = self.cri_nlc(mid_normal, self.svbrdf[:,:3])
            loss_dict['l_midn'] = l_midn
            l_total += l_midn
        l_total += self.brdfLoss(normal, diffuse, roughness, specular, loss_dict, isDesc)
        return l_total
    
    def save_visuals(self, path, pred, gt):
        if gt is not None:
            output = torch.cat([gt,pred],dim=2)*0.5+0.5
        else:
            output = torch.cat([torch.ones_like(pred)*(-1.0),pred],dim=2)*0.5+0.5
        normal, diffuse, roughness, specular = torch.split(output,[3,3,1,3],dim=1)
        roughness = torch.tile(roughness,[1,3,1,1])
        if self.opt['val'].get('gammCorr', False):
            diffuse, specular = diffuse**0.4545, specular**0.4545
        output = torch.cat([normal,diffuse,roughness,specular],dim=-1)
        
        renderer, gtrender = self.eval_render(pred, gt)
        render = torch.split(torch.cat([gtrender,renderer],dim=-2),[1]*gtrender.shape[-4], dim=-4)
        render = torch.cat(render, dim=-1).squeeze(1)

        if self.real_input:
            self.lighting.initTexture(torch.flip(self.pattern, dims=[-1]), 'Torch')
        inputs = (self.inputs*0.5+0.5)**0.4545
        renderimg = [inputs[:, :3, :, :]]
        poly_render = self.polyRenderer.render(pred, light_dir=self.lDir, view_dir=self.vDir, n_xy=False)
        renderimg.append(poly_render ** 0.4545)
        env_contrib = (self.inputs / 2 + 0.5 - poly_render).clip(0.0, 1.0) ** 0.4545
        
        if self.use_normal_lc and self.svbrdf is None:
            renderimg[0] = torch.cat([renderimg[0], env_contrib],dim=-1)
            renderimg[1] = torch.cat([renderimg[1], self.mid_normal / 2 + 0.5],dim=-1)
        elif self.svbrdf is None:
            renderimg[0] = torch.cat([renderimg[0], torch.zeros_like(renderimg[0])],dim=-1)
            renderimg[1] = torch.cat([renderimg[1], env_contrib],dim=-1)
        else:
            renderimg[0] = torch.cat([renderimg[0], self.act_contrib, self.env_contrib],dim=-1)
            renderimg[1] = torch.cat([self.mid_normal / 2 + 0.5, renderimg[1], env_contrib],dim=-1)

        renderimg = torch.cat(renderimg,dim=2)
        output = torch.cat([renderimg,render**0.4545, output], dim=-1)
        output_img = tensor2img(output,rgb2bgr=True)
        imwrite(output_img, path, float2int=False)
        if self.vis_lightclues:
            lightclues = self.lightclues.chunk(6, dim=1)
            lc_path = path.replace('.png', '-lc.png')
            output_lc = tensor2img((torch.cat(lightclues, dim=-1)/2+0.5)**0.4545, rgb2bgr=True)
            imwrite(output_lc, lc_path, float2int=False)
    
    def save_pre_visuals(self, path, pred):
        output = pred*0.5+0.5
        normal, diffuse, roughness, specular = torch.split(output,[3,3,1,3],dim=1)
        roughness = torch.tile(roughness,[1,3,1,1])
        if self.opt['val'].get('gammCorr', False):
            diffuse, specular = diffuse**0.4545, specular**0.4545
        output = torch.cat([normal,diffuse,roughness,specular],dim=-1)
        output_img = tensor2img(output,rgb2bgr=True)
        imwrite(output_img, path, float2int=False)
        if self.vis_lightclues and self.opt['val'].get('save_gt', False):
            save_path = self.opt['val'].get('lcsSavePath')
            lightclues = self.lightclues.chunk(6, dim=1)
            basename = os.path.basename(path)
            lc_path = 'lc_'+basename
            output_lc = tensor2img((torch.cat(lightclues, dim=-1)/2+0.5)**0.4545, rgb2bgr=True)
            imwrite(output_lc, os.path.join(save_path,lc_path), float2int=False)
        if self.vis_mirrorball:
            save_path = self.opt['val'].get('ballSavePath')
            basename = os.path.basename(path)
            ball_path = 'ball_'+basename
            output = tensor2img(self.inputs[:,3:]/2+0.5, rgb2bgr=True)
            imwrite(output, os.path.join(save_path,ball_path), float2int=False)

    def save_rendering(self,path,inputs, gamma=False):
        if gamma:
            out = (inputs[:,:3]*0.5+0.5)**0.4545
        else:
            out = inputs[:,:3]*0.5+0.5
        output_img = tensor2img(out,rgb2bgr=True)
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
                results = self.get_current_visuals(self.output, self.svbrdf)
                if self.visSavePath is not None:
                    save_path = self.visSavePath
                else:
                    save_path = osp.join(self.opt['path']['visualization'], dataset_name)

                brdf_path=osp.join(save_path, val_data['name'][0])
                self.save_pre_visuals(brdf_path,results['predsvbrdf'])

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                # print('pred:',self.output)
                # print('gt:',self.brdf)
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    if metric_type == 'pix':
                        error = torch.abs(self.output-self.svbrdf).mean()*opt_.pop('weight')
                    self.metric_results[name] += error 
            torch.cuda.empty_cache()
            if self.opt.get('pbar',True):
                pbar.update(1)
                pbar.set_description(f'Testing')
                # break
        if self.opt.get('pbar',True):
            pbar.close()
            
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name,tb_logger)
    
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name};\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\t'
        print(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)