import os
import numpy as np
import cv2
import torch

class Metrics:
    def __init__(self,RMSE=True, device='cpu'):
        self.RMSE = RMSE
        self.device = device
            
    def __get_svbrdf_parametes(self,svbrdf):
        n, d, r, s = np.split(svbrdf,svbrdf.shape[1] // 256,1)[-4:]
        n = np.expand_dims(n,0)
        d = np.expand_dims(d,0)
        r = np.expand_dims(r,0)
        s = np.expand_dims(s,0)
        svs = np.concatenate([n,d,r,s],0)
        return svs

    def svbrdfs_from_dir(self,gt_dir,pre_dir, exp_name='exp'):
        gt_path = sorted(os.listdir(gt_dir))
        gt_svbrdfs = np.ones([len(gt_path),4,256,256,3])
        for i,name in enumerate(gt_path):
            path = os.path.join(gt_dir,name)
            svbrdf = cv2.imread(path)[:,:,::-1]/255
            gt_svbrdfs[i] = self.__get_svbrdf_parametes(svbrdf)
        pre_path = gt_path
        pre_path.sort()
        pre_svbrdfs = np.ones([len(pre_path),4,256,256,3])
        for i,name in enumerate(pre_path):
            path = os.path.join(pre_dir,name)
            svbrdf = cv2.imread(path)[:,:,::-1]/255
            pre_svbrdfs[i] = self.__get_svbrdf_parametes(svbrdf)

        self.__do_metric(gt_svbrdfs, pre_svbrdfs, exp_name)

    def __log_metric(self, results, exp_name):
        label = ["Norm.", "Diff.", "Rogh.", "Spec.", "Mean."]
        if len(results) == 6:
            label.append("Rend.")
        log = "%s: " % exp_name
        for i, metrics in enumerate(results):
            log += "[%s] " % (label[i])
            for metric_name, metric_value in metrics.items():
                log += "%s: %.4f " % (metric_name[0], metric_value)
        log += "\n"
        return log
    
    def __mean_metric(self, results):
        keys = results[0].keys()
        mean_dict = {}
        for key in keys:
            mean_dict[key] = 0
        
        for metric_dict in results[:4]:
            for metric_name, metric_value in metric_dict.items():
                mean_dict[metric_name] += metric_value
                
        for i, metric_name in enumerate(keys):
            mean_dict[metric_name] /= 4
        results.insert(4, mean_dict)
        return results

    def __do_metric(self, gt_svbrdfs, pre_svbrdfs, exp_name):
        gt_svbrdfs = torch.from_numpy(gt_svbrdfs).permute(0, 1, 4, 2, 3).contiguous().float()
        pre_svbrdfs = torch.from_numpy(pre_svbrdfs).permute(0, 1, 4, 2, 3).contiguous().float()
        metrics = {
            "RMSE": self.__RMSE_torch
        }
        ngt, dgt, rgt, sgt = gt_svbrdfs.chunk(4, 1)
        n, d, r, s = pre_svbrdfs.chunk(4, 1)
        img_pairs = [[ngt.squeeze(1), n.squeeze(1)], [dgt.squeeze(1), d.squeeze(1)], [rgt.squeeze(1), r.squeeze(1)], [sgt.squeeze(1), s.squeeze(1)]]
        res = []
        for i, imgs in enumerate(img_pairs):
            res.append({})
            for metric in metrics:
                imgs1, imgs2 = imgs
                flag = getattr(self, metric)
                if flag:
                    res[i][metric] = metrics[metric](imgs1, imgs2).cpu().item()
        res = self.__mean_metric(res)
        log = self.__log_metric(res, exp_name)
        print(log)

    def __RMSE_torch(self,gt,pre):
        mse = torch.mean((gt-pre)**2)
        rmse = torch.sqrt(mse)
        return rmse