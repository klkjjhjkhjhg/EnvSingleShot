import numpy as np
import torch
from utils.utils import torch_norm, torch_dot, toLDR_torch, toHDR_torch, torch_cross
import nvdiffrast.torch as dr

class BRDF():
    '''
    The base class of brdf, containing several brdf term implement (ggx, ...).
    Every subclass should implement the abstractmethod 'eval(), sample()'.
    '''
    def __init__(self, opt, device='cpu'):
        '''
        Construct method

        Args:
            opt (dict): The configuration of current brdf instance, containing none properties.
            The class inherited from this class could pass the parameters through 'opt'.
        '''
        self.opt = opt if opt is not None else {'size': 256, 'order':'pndrs'}
        self.brdf = None
        self.device=device

    def GGX(self, NoH, roughness, eps=1e-12):
        '''
        Isotropic GGX distribution based on
        "Microfacet Models for Refraction through Rough Surfaces"
        https://www.cs.cornell.edu//~srm/publications/EGSR07-btdf.pdf

        Args:
            NoH (Tensor): the dot product (cosine) of middle vector and surface normal
            roughness (Tensor): roughness of the surface

        Returns:
            Tensor : The evaluation result of given roughness and NoH
        '''
        alpha = roughness * roughness
        
        tmp = alpha / torch.clip((NoH * NoH * (alpha * alpha - 1.0) + 1.0),eps)
        return tmp * tmp * (1 / torch.pi)

class Sampler():
    def __init__(self, opt, device='cpu'):
        self.opt = opt
        self.device = device
    
    @staticmethod
    def hammersley_2d(n, b):

        def radical_inverse(index, base):
            result = 0.0
            f = 1.0 / base
            while index > 0:
                result += (index % base) * f
                index //= base
                f /= base
            return result
        
        points = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            points[i, 0] = i / n  # Uniformly distributed along the x-axis
            points[i, 1] = radical_inverse(i, b)  # Low-discrepancy values along the y-axis
        return points

class Directions():
    def __init__(self, opt, device='cpu'):
        self.opt = opt
        self.device = device

    def random_pos(self, batch=0, n=1, r=[-1, 1]):
        if batch == 0:
            shape = (2, n)
        else:
            shape = (batch, 2, n)
        xy = torch.empty(shape, dtype=torch.float32, device=self.device).uniform_(r[0], r[1])
        if batch == 0:
            shape = (1, n)
        else:
            shape = (batch, 1, n)
        z = torch.zeros(shape, dtype=torch.float32, device=self.device)
        return torch.cat([xy, z], dim=-2)
    
    @staticmethod
    def buildTBNStatic(n):
        judgement = ~torch.logical_or((n[...,2:] == 1), (n[...,2:] == -1))
        tZ = torch.stack([torch.zeros_like(n[...,2]),torch.zeros_like(n[...,2]),torch.ones_like(n[...,2])],dim=-1)
        tX = torch.stack([torch.ones_like(n[...,2]),torch.zeros_like(n[...,2]),torch.zeros_like(n[...,2])],dim=-1)
        t = torch.where(judgement, tZ, tX)
        t = torch_norm(t - n * torch_dot(n, t, dim=-1), dim=-1)
        b = torch_cross(n, t, dim=-1)
        TBN = torch.stack([t, b, n], dim=-2)
        return TBN

    @staticmethod
    def worldTBNTransfrom(vec, n, v=None, mode='view', permute=False, world2tbn=True, tbn2world=False):
        # if reverseTBN is True
        # n: (b, h, w, 3) or (h, w, 3) or (b, 1, h, w, 3)
        expand=False
        if n.ndim == 3:
            n = n.unsqueeze(0)
        elif n.ndim == 5:
            n = n.squeeze(1)
        if mode == 'static':
            TBN = Directions.buildTBNStatic(n)
        elif mode == 'view':
            TBN = Directions.buildTBNView(n,v)
        if isinstance(vec, list):
            result = []
            for vv in vec:
                if tbn2world:
                    TBN = TBN.permute(0,1,2,4,3).contiguous()
                tmpV = torch.matmul(TBN, vv.permute(
                    2,3,1,0).contiguous()).squeeze(0)
                result.append(tmpV.permute(3, 2, 0, 1).contiguous())
        else:
            # TBN n, 3, 3, h, w
            # vec ln, c, h, w
            vec = vec.permute(0, 3, 4, 2, 1).contiguous()
            if tbn2world:
                TBN = TBN.permute(0,1,2,4,3).contiguous()
            result = torch.matmul(TBN, vec)
            if permute:
                result = result.permute(0, 4, 3, 1, 2).contiguous()
        return result

    @staticmethod
    def buildTBNView(n, v):
        NoV = torch_dot(n,v, dim=-1)
        tv = torch_norm(v-n*NoV, dim=-1)
        tx = torch.stack([torch.ones_like(n[:, :, :, 2]),torch.zeros_like(n[:, :, :, 2]),torch.zeros_like(n[:, :, :, 2])],dim=-1)

        t = torch.where(~(NoV == 0.0), tv, tx)
        b = torch_cross(n, t, dim=-1)
        TBN = torch.stack([t, b, n], dim=-2)
        return TBN

class PlanarSVBRDF(BRDF):
    '''
    A subclass of BRDF which is a planar svBRDF represented by an multi-channel matrix with the same size of surface matrix.
    '''
    def __init__(self, opt=None, svbrdf=None, device='cpu'):
        '''
        Construction function

        Args:
            opt (dict): Configuration of svbrdf, which contains following properties:
                size: the size of surface and svBRDF matrix
            svbrdf (Tensor, optional): Pre-computed svbrdf. Defaults to None.
        '''
        super().__init__(opt, device)
        self.size = self.opt['size']
        self.brdf = svbrdf

    def sample(self, u, v):
        return self.sampleGGX(u, v)

    def importanceSampling(self, n, r, view, sN=1, h=None):

        
        dim = -3
        if isinstance(r, torch.Tensor):
            r = r.unsqueeze(dim-1).unsqueeze(dim-1)
        view = view.unsqueeze(dim-1).unsqueeze(dim-1)
        r = r / 2 + 0.5
        if h is None:
            #* the shape of self.u, self.v is (b, sampleN, sampleN)
            b, c, x, y = n.shape
            # Generate Hammersley sequence
            points = Sampler.hammersley_2d(sN**2, 2)  # Generate sampleN * sampleN points
            u_vals = points[:, 0].reshape(1, sN, sN, 1, 1, 1)  # Reshape into 6D tensor
            v_vals = points[:, 1].reshape(1, sN, sN, 1, 1, 1)  # Reshape into 6D tensor
            # Convert to torch tensors
            u = torch.tensor(u_vals, dtype=torch.float32, device=self.device)
            v = torch.tensor(v_vals, dtype=torch.float32, device=self.device)

            # Broadcast to match the target shape
            u = u.broadcast_to((b, sN, sN, 1, x, y))
            v = v.broadcast_to((b, sN, sN, 1, x, y))
            #* Importance sampling half-vectors
            phi = 2 * torch.pi * v
            cosTheta = torch.sqrt((1 - u) / (1 + (r**4 - 1) * u))
            sinTheta = torch.sqrt(1 - cosTheta * cosTheta)
            h = torch.cat([sinTheta * torch.cos(phi), sinTheta * torch.sin(phi), cosTheta], dim=dim).view(b, -1, 3, *sinTheta.shape[dim+1:])
            h = Directions.worldTBNTransfrom(h, n.permute(0, 2, 3, 1), mode='static', permute=True, tbn2world=True).view(b, sN, sN, 3, *sinTheta.shape[dim+1:])
            #* h = torch_norm(h, dim=3)
            VoH = torch_dot(view, h, dim=dim)
            l = 2 * VoH * h - view
        else:
            h = h.permute(0,2,3,1).contiguous().unsqueeze(-1).unsqueeze(-1)
            VoH = torch_dot(view, h, dim=dim)
            l = 2 * VoH * h - view

        #* Calculate pdf of the importance sampled vectors
        n = n.unsqueeze(dim-1).unsqueeze(dim-1)
        NoH = torch_dot(n, h, dim=dim).clip(1e-12)
        VoH = VoH.clip(1e-12)
        
        D = self.GGX(NoH, r)
        pdfL = D * NoH / 4.0 / VoH
        pdfH = D * NoH
        
        return l, h, D, pdfL, pdfH

    def _separate_brdf(self, svbrdf=None, n_xy=False, r_single=True, lightdim=True):

        if svbrdf is None:
            svbrdf = self.brdf
        if svbrdf.dim() == 4:
            b, c, h, w = svbrdf.shape
        else:
            b = 0
            svbrdf = svbrdf.unsqueeze(0)

        if self.opt['order'] == 'pndrs' or self.opt['order'] == 'ndrs':
            if not n_xy:
                n = svbrdf[:, 0:3]
                d = svbrdf[:, 3:6]
                if r_single:
                    r = svbrdf[:, 6:7]
                    s = svbrdf[:, 7:10]
                else:
                    r = svbrdf[:, 6:7]
                    s = svbrdf[:, 9:12]
            else:
                n = svbrdf[:, 0:2]
                n = self.unsqueeze_normal(n)
                d = svbrdf[:, 2:5]
                if r_single:
                    r = svbrdf[:, 5:6]
                    s = svbrdf[:, 6:9]
                else:
                    r = svbrdf[:, 5:6]
                    s = svbrdf[:, 8:11]
        elif self.opt['order'] == 'dnrs':
            d = svbrdf[:, 0:3]
            n = svbrdf[:, 3:6]
            r = svbrdf[:, 6:7]
            s = svbrdf[:, 7:10]
        if b != 0 and lightdim:
            n = n.unsqueeze(1)
            d = d.unsqueeze(1)
            r = r.unsqueeze(1)
            s = s.unsqueeze(1)
        elif b==0 and not lightdim:
            n, d, r, s = [torch.squeeze(x, dim=-4) for x in [n, d, r, s]]
        return torch_norm(n,dim=-3), d, r, s

class Render():
    def __init__(self, opt=None, device='cpu'):
        if opt is not None:
            self.opt = opt
        else:
            opt = {
                'nbRendering': 1,
                'size': 256,
                'order' : 'pndrs',
                'toLDR' : False,
                'lampIntensity' : np.pi
            }
            self.opt = opt
        self.nbRendering = opt['nbRendering']
        self.size = opt.get('size', 256)
        self.device = device
        self.toLDR = self.opt.get('toLDR', False)
    def torch_generate(self, camera_pos_world, light_pos_world, surface=None, pos=None, normLight=True):
        # permute = self.opt['permute_channel']
        if pos is None and surface is None:
            pos = self.generate_surface(self.size)
        elif surface is not None:
            pos = surface.to(self.device)
        else:
            pos.unsqueeze_(-1).unsqueeze_(-1)

        light_pos_world = light_pos_world.view(*light_pos_world.shape, 1, 1)
        camera_pos_world = camera_pos_world.view(*camera_pos_world.shape, 1, 1)

        view_dir_world = torch_norm(camera_pos_world - pos, dim=-3)

        # pos = torch.tile(pos,[n,1,1,1])
        light_dis_square = torch.sum(torch.square(light_pos_world - pos), -3, keepdims=True)

        if normLight:
            light_dir_world = torch_norm(light_pos_world - pos, dim=-3)
        else:
            light_dir_world = light_pos_world-pos

        return light_dir_world, view_dir_world, light_dis_square, pos

    def generate_surface(self, size):
        '''generate a plane surface with the size of $size

        Args:
            size (int): the size of edge length of surface

        Returns:
            pos: the position array of surface
        '''
        x_range = torch.linspace(-1, 1, size, device=self.device)
        y_range = torch.linspace(-1, 1, size, device=self.device)
        y_mat, x_mat = torch.meshgrid(x_range, y_range, indexing='ij')
        pos = torch.stack([x_mat, -y_mat, torch.zeros(x_mat.shape, device=self.device)], axis=0)
        pos = torch.unsqueeze(pos, 0)
        return pos

    def to(self, device):
        self.device = device

class EnvRender(Render):
    def __init__(self, opt=None, device='cpu'):
        super().__init__(opt, device)
        
        self.lampIntensity = 0.5 #! The training and testing lighting intensity is 0.5.
        lutPath = self.opt.get('lutPath', 'resources/misc/bsdf_256_256.bin')
        self.lut = torch.from_numpy(np.fromfile(lutPath, dtype=np.float32).reshape(1, 256, 256, 2)).to(self.device)
        self.nLod = opt.get('nLod', 8)
        
    def render(self, svbrdf, env_map_stack, view_pos, obj_pos=None, n_xy=False, r_single=True, toLDR=None, useDiff=True, clip=True):
        if not isinstance(svbrdf, PlanarSVBRDF):
            svbrdf = PlanarSVBRDF(self.opt, svbrdf, device=self.device)
        if toLDR is None:
            toLDR = self.toLDR
            
        n, d, r, s = svbrdf._separate_brdf(n_xy=n_xy, r_single=r_single, lightdim=False)
        _, view_dir, _, surface = self.torch_generate(view_pos, view_pos, pos=obj_pos)

        lampIntensity = self.lampIntensity
        
        render_result = self.__render(n, d, r, s, view_dir, env_map_stack, useDiff, lampIntensity)
        
        if self.nbRendering == 1:
            render_result.squeeze_(0)
        render_result = render_result.clip(0.0, 1.0) if clip else render_result
        
        if toLDR:
            render_result = toLDR_torch(render_result)
            render_result = toHDR_torch(render_result)
        return render_result
    
    def __render(self, n, d, r, s, v, env_map_stack, useDiff=True, iten=1.0, eps=1e-6):
        if isinstance(r, torch.Tensor):
            r = torch.clip(r * 0.5 + 0.5, 0.01)
        else:
            r = max(r * 0.5 + 0.5, 0.01)
        dim = -3
            
        n = torch_norm(n, dim=dim)
        #* reflect v by n
        reflection = 2 * torch_dot(v, n, dim=dim) * n - v

        NoV = torch_dot(n, v, dim=dim)
        mask = NoV > 0.0
        NoV = torch.clip(NoV, eps)

        # diffuse term
        d = d * 0.5 + 0.5
        diffuse_light = self.fetchEnvMap(n, env_map_stack, torch.ones_like(r))
        diffuse_color = d * diffuse_light

        # specular term
        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(r,min=0.0,max=1.0)],dim)
        uv = fg_uv.permute(0, 2, 3, 1).contiguous()
        fg_lookup = dr.texture(self.lut, uv, filter_mode='linear', boundary_mode='clamp')
        fg_lookup = fg_lookup.permute(0, 3, 1, 2).contiguous()

        s = s * 0.5 + 0.5
        specular_ref = (s * fg_lookup[:,0:1] + fg_lookup[:,1:2])
        specular_light = self.fetchEnvMap(reflection, env_map_stack, r)
        specular_color = specular_ref * specular_light

        # integrated together
        color = specular_color * iten if not useDiff else (specular_color + diffuse_color) * iten
        return color * mask
     
    def fetchEnvMap(self, reflect, env_map_stack, roughness):
        #* parametrize the reflection direction to polar coordinates
        reflect = reflect.permute(0, 2, 3, 1).contiguous()
        x, y, z = reflect[:,:,:,0], reflect[:,:,:,1], reflect[:,:,:,2]
        theta = torch.acos(z.clip(-1+1e-6, 1-1e-6))
        phi = torch.atan2(y, x)
        u = 1 - phi / (2 * torch.pi) + 0.25
        flag = u > 1.0
        u[flag] = u[flag] - 1.0
        v = theta / torch.pi
        
        uv = torch.stack([u, v], dim=-1)
        lod = (roughness.squeeze(-3)) * self.nLod
        
        env_map_stack = [e.to(self.device) for e in env_map_stack]
        tex = env_map_stack[0]
        light = dr.texture(tex, uv, mip_level_bias=lod, mip=env_map_stack[1:], boundary_mode='clamp', max_mip_level=self.nLod)
        
        light = light.permute(0, 3, 1, 2).contiguous()
        return light