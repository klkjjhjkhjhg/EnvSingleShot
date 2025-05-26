
import os
from utils.metric import Metrics
import time
from utils.utils import set_random_seed

if __name__ =='__main__':
    set_random_seed(10)
    starttime= time.time()
    result_root='results/Results_Syn/visualization'

    metrics = Metrics(RMSE=True, device='cuda')
    explist = ['MatSynth3Dataset']

    for exp in explist:
        metrics.svbrdfs_from_dir("resources/synthetic_data/SVBRDFs",os.path.join(result_root, exp), exp_name=exp)
    print("cost time: %.2f seconds"%(time.time()-starttime))
    