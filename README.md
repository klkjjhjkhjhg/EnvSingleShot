# On-Site Single Image SVBRDF Reconstruction with Active Planar Lighting
Lianghao Zhang, Ruya Sun, Li Wang, Fangzhou Gao, Zixuan Wang, Jiawan Zhang*

(* indicates the corresponding author)

<!-- ### [Paper]() | [Data]() -->

## Preparation
- Set up the python environment

```sh
conda create -n EnvSingleShot python=3.10.0
conda activate EnvSingleShot

pip install -r requirement.txt
```

- Our project has been tested on Ubuntu 20.04 using CUDA==12.8 and PyTorch==2.7.0+cu128. You can install PyTorch by the following command.

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

- The [nvidiffrast library]() is required for rendering. You can install it using `cd {nvdiffrast_root} | pip install .` after download the code from their repository. We use the version 0.3.1.

- Download our pre-trained model and testing data from [Google Drive](https://drive.google.com/drive/folders/17r54W9M6Z6_jlgxk4JvfLmqkXyCVjpl0?usp=sharing). We provide three models: EnvSingleShot, EnvSingleShot_Denoise, and EnvSingleShot_Real, designed for reconstruction on both synthetic and real scenes. For detailed descriptions of each model’s functionality, please refer to our paper. Additionally, we have included configuration files in the repository that specify which model to use for different types of scenes.

- After downloading the model and data, you can put the downloaded files into the corresponding folders in the repository:

```
- {project_root}
    - ...
    - resources
        - misc
        - real_data
            - raw_images
        - real_data_ready
            - ...
        - synthetic_data
            - SVBRDFs
            - Inputs
            - EnvLCs
            - EnvMips
    - ckpts
        - EnvSingleShot.pth
        - EnvSingleShot_Denoise.pth
        - EnvSingleShot_Real.pth
```

## Run the code

#### Testing (Synthetic Data)

1. Directly run the following command to test the model on synthetic data. The results will be saved in the `results/Results_Syn` folder.

   ```
   python test.py -opt test_syn.yml
   ```

2. Run the following command to calculate the RMSE error of estimated SVBRDFs. The printed results should be same as the results in Table 1 of our paper.

   ```
   python scripts/cal_metrics.py
   ```
   
#### Testing (Real Data)

1. Before reconstructing SVBRDFs, we should first get the environment map and real lighting patterns from the raw captured images. 

  ```
  python scripts/1_cropimage.py --expDir resources/real_data # crop input images from the original images.
  python scripts/2_singleball.py --expDir resources/real_data # manually select the mirror ball.
  python scripts/3_singlenv.py --expDir resources/real_data # split the environment map and real lighting patterns.
  python scripts/4_prefilter_envmaps.py --expDir resources/real_data #  pre-filter the environment map.
  python scripts/5_LightClue_real.py --expDir resources/real_data # render the plane lighting clues.
  ```

- In these steps, step 2 and 3 needs users to manually adjust the parameters of edge detection or lighting separation. Instead, we provided a set of pre-processed data of these steps in the `resources/real_data_ready` folder. You can skip these two manual steps by copying the data from the `resources/real_data_ready` folder to the `resources/real_data` folder. The folder that should be copied contains: `sparse`, `resource`, `mipmaps`, and `patterns`.

2. After running above commands, we can run the following command to reconstruct SVBRDFs. The results will be saved in the `results/real_data/svbrdfs_ours` folder, which shoud be the same as the results in Figure 11 of our paper.

  ```
  python test.py -opt test_denoise.yml
  python test.py -opt test_real.yml
  ```

## Citation

Acknowledgements: part of our code is inherited from  [BasicSR](https://github.com/XPixelGroup/BasicSR) and [NeRO](https://github.com/liuyuan-pal/NeRO). We are grateful to the authors for releasing their code. Besides, we would like to thank the authors of [RADN](https://team.inria.fr/graphdeco/projects/deep-materials/) and [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth) for providing the datasets.
