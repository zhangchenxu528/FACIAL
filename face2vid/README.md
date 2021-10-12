# Everybody Dance Now (pytorch)
A PyTorch implementation of ["Everybody Dance Now"](https://arxiv.org/abs/1808.07371) from Berkeley AI lab. 
Including all functionality except pose normalization. 

Other implementations:

[yanx27](https://github.com/CUHKSZ-TQL/EverybodyDanceNow_reproduce_pytorch) EverybodyDanceNow reproduced in pytorch

[nyoki-pytorch](https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow) pytorch-EverybodyDanceNow

Also check out [densebody_pytorch](https://github.com/Lotayou/densebody_pytorch) for 3D human mesh estimation from monocular images.

## Environment
- Ubuntu     18.04 (But 16.04 should be fine too)
- Python     3.6
- CUDA       9.0.176
- PyTorch    0.4.1post2

For other necessary packages, Use `pip install -r requirements` for a quick install.  
1. The project requires tensorflow>1.9.0 since the pose estimator is implemented in Keras. If you are using an independent Keras package, change the corresponding import command in `./pose_estimator/compute_coordinates_for_video.py`. However, you won't be able to use tensorboard this way.
2. The project uses imageio for video processing, which requires the ffmpeg core to be downloaded and installed after the pip command. Just follow the error message and you'll get there.

## Dataset Preparation
1. To reproduce our results, download the [Afrobeat workout Sequence](https://www.youtube.com/watch?v=kyKNPPQW3bM) from YouTube. ([clipconverter](https://www.clipconverter.cc/) is a great downloading tool.) 

2. Open the mp4 file with _imageio_ , remove the first 25 seconds (625 frames for our video), and resize the rest frames into 288\*512(for 16:9 HD video). Save all the frames in a single folder named `train_B`. It is highly recommended that the frames are named with their index numbers, like `00001.png`, `00002.png`.

3. Download a pre-trained pose-estimator from [Yandex Disk](https://yadi.sk/d/blgmGpDi3PjXvK) and put it under subfolder `pose-estimator`, then run the following script to estimate the pose for each frame and render the poses into RGB images. 

    `python ./pose_estimator/compute_coordinates.py`

    The script is supposed to generate a folder named `train_A` containing corresponding pose stickfigure images (also named as `00001.png`, `00002.png`, etc.), and a numpy file `poses.npy` that contains estimated poses of size N\*18\*2, where N is the number of frames. 

    The numpy file is not necessary for training global generator, but we need it for training face-enhancer since we need to estimate and crop the head region from synthesized frames.

    __Note__: You can also use Openpose or any other pose-estimation networks for this step. Just make sure you organize your pose data as suggested above.

4. Wrap `train_A`, `train_B` and `poses.npy` into the same folder and put it under `./datasets/`.

## Use your own dataset
The model is not fully trained/tested on other dancing videos. You are encouraged to play with your own dataset as well, but the performance is not guaranteed.

Empirically, to increase the change of success in training/testing, it is important that your video:
- has a fixed, clean background
- is more than 5 minutes
- contains a single person performing "basic" actions(which will be elaborated later)
- (Optional, only if you use optical flow loss) contains mimimal change in lighting conditions

On the contrary, your training would possibly fail if your video contains
- Intensive movements such as turning around, kneeling down
- Heavy limb occlusions 
- Large scale variations (e.g. dancer running towards/away from the camera)

If you encounter any failcase, do not hesitate to leave an issue to let us know!

## Testing
1. Download pretrained checkpoints:
- [Pose2vid generator](https://yadi.sk/d/gpKvisk8uLuUyA): put in `./checkpoints/everybody_dance_now_temporal/`
#### Update on 20181204
- [Face enhancer for Afrobeat sequence](https://yadi.sk/d/U_sRn9dZiV-G0w): put in `./face-enhancer/checkpoints/dance_test_new_down2_res6/`
- [Pretrained VGG 16 weights](https://yadi.sk/d/uKcv5uxzD40WjA): put in `./face-enhancer/utils/`

2. Prepare the testing sequence: Save the skeleton figures in a folder named `test_A`, slice the corresponding pose coordinates from previously cached `poses.npy`, and wrap them in a single folder (for example `cardio_dance_test`) and put it under `./datasets/`.

   In addition, the program supports using first ground-truth frame as a reference, so create a new folder `test_B` and put inside the ground truth frame corresponding to the first item in `test_A` (with identical file name of course).

3. Run the following command for global synthesis 

    `sh ./scripts/test_full_512.sh`
    
   This will generates a coarse video stored in `./results/$NAME$/$WHICH_EPOCH$/test_clip.avi` and cache all synthesized frames for face_enhancer evaluation.
 
4. Run the face-enhancer to get the final result.

    `python ./face-enhancer/enhance.py`

## Training
#### Step I: Training global pose2vid network.
1. Prepare the dataset following the instructions above. 

2. For pose2vid baseline, run the script 

   `sh ./scripts/train_full_512.sh` 
    
   If you wish to incorporate optical flow loss, run the script

   `sh ./scripts/train_flow_512.sh`
    
   __Warning__: this module will increase memory cost and slows down the training speed by 40% to 50%. Also it's very sensitive to background flow, so use it at your discretion. However, if you can accurately estimate the dancer's body mask, using masked flow could help with temporal smoothing. Please send a PR if you find masked Flowloss effective.

#### Step II: Training local face-enhancing network.
1. Rename your `train_B` folder into `test_real` (Or you can save a copy and rename it)

2. Test the global pose2vid network (either trained from Step I or initialized with downloaded pretrained model) with your `train_A` dataset, save the results into a folder named `test_sync` with matching names.

3. Open the face-enhancement training script at `./face_enhancement/main.py`, modify the `dataset_dir, pose_dir, checkpoint dir, log_dir` variables, and run the script.

4. The default network structure is 2 downsample layers, 6 Resblocks, and 2 upsample layers. You can modify it for best enhancing effect, just change the corresponding parameters at line 22. Also the crop size is adjustable at line 23(default is 96).

## Citation
Should you find this implementation useful, please add the following citation in your paper/open-sourced project:
```
@article{chan2018everybody,
  title={Everybody dance now},
  author={Chan, Caroline and Ginosar, Shiry and Zhou, Tinghui and Efros, Alexei A},
  journal={arXiv preprint arXiv:1808.07371},
  year={2018}
}
```

## Acknowledgement
This repo borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
