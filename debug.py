import cv2
import numpy as np
img =  cv2.imread('/home/sda/klkjjhjkhjhg/code/DeepMaterial/results/data/EnvSingleShotFinalExp/Real/HasRef/SVBRDFs/Ours/IMG_4420.png')

d = img[:,256:256*2]/255.
s = img[:,256*3:256*4]/255.
svbrdf = np.concatenate([img[:,:256]/255.,d**(1/2.2),img[:,256*2:256*3]/255.,s**(1/2.2)],axis=1)

cv2.imwrite('tmp/svbrdf.png',svbrdf*255)