import numpy as np
import os
import os.path as osp
import cv2
import argparse

mode = 'normal' #* normal or debug
def clipInput(root="Inputs/Real", fovReal = 40, delta=0.75, size=256, outFolder="cropped"):
    fov = 35
    delta = delta
    materialSize=2
    p = 4
    h = (materialSize + delta) / np.tan((fov)/180*np.pi)
    fovh = np.array([np.arctan(delta/2/h), np.arctan((2*delta+p)/2/h)])
    hrange = (fovh/np.pi*180/fovReal).clip(0.0, 1.0)
    os.makedirs(outFolder, exist_ok=True)
    imgs = sorted(os.listdir(root))
    for n in imgs:
        img = cv2.imread(osp.join(root, n))
        (h, w) = img.shape[:2]  
        rotated = img
        cv2.imwrite(osp.join(root, 'left', osp.splitext(n)[0]+'.png'), rotated)
    imgs = os.listdir(root)
    imgs = [n for n in imgs if n[0] != '.']
    for n in imgs:
        img = cv2.imread(osp.join(root, n))
        h, w = img.shape[:2]
        pixRange = w/2*hrange
        wUvRange = (w/2-pixRange).astype(np.int32)
        wUvRange = wUvRange[::-1]
        height = pixRange[1] - pixRange[0]
        hUvRange = np.array([(h-height)/2,h-(h-height)/2]).astype(np.int32)
        crop = img[hUvRange[0]:hUvRange[1], wUvRange[0]:wUvRange[1]]
        crop = cv2.resize(crop, (size,size), interpolation=cv2.INTER_AREA)
        filename, ext = osp.splitext(n)
        cv2.imwrite(osp.join(outFolder, filename+'.png'),crop)
        if osp.exists(osp.join(outFolder, filename+'.jpg')):
            os.remove(osp.join(outFolder, filename+'.jpg'))
        if osp.exists(osp.join(outFolder, filename+'.JPG')):
            os.remove(osp.join(outFolder, filename+'.JPG'))

if __name__ == '__main__':
    #* Change the system parameters for every test function.
    #镜球检测
    parser = argparse.ArgumentParser()
    parser.add_argument('--expDir', default='expDir')
    opt = parser.parse_args()
    expDir = opt.expDir
    
    inDir = osp.join(expDir, "raw_images")
    outDir = osp.join(expDir, "images")
    
    sample_size_real = 9
    rect_size_real = 18
    pattern_scale = sample_size_real/(rect_size_real/2)
    fov = 35
    light_size_real = 18*pattern_scale
    fovReal = np.arctan(36/2/24)/np.pi * 180 # calculated using equivalent focal length. 24mm is the focal length of the G7X MarkII.
    # fovReal = 40 # Using the value of the website: https://www.zhihu.com/tardis/bd/art/677281026?source_id=1001
    delta = 0.5
    delta_real = (delta * sample_size_real) / 2
    height = (sample_size_real+delta_real) / np.tan(fov/180*np.pi)
    
    #* Crop the captured image automatically
    print("Clipping the captured image.")
    clipInput(inDir, fovReal, delta=delta, outFolder=outDir)
