############## To train full model #############
##### Using GPUs with 12G memory (not tested)
# Using labels only (0.56s per batch)
export CUDA_VISIBLE_DEVICES=0
python train.py --name everybody_dance_now_pwcflow --model pose2vid --dataroot ./datasets/cardio_dance_512/ --continue_train --netG local --ngf 32 --num_D 3 --tf_log --niter_fix_global 10 --label_nc 0 --no_instance --save_epoch_freq 2 --lr=0.0001 --resize_or_crop none # > train_512_pwc_log.txt & 
# python train.py --debug --name everybody_dance_now_debug --model pose2vid --dataroot ./datasets/cardio_dance_512/ --netG local --ngf 32 --num_D 3 --tf_log --niter_fix_global 10 --label_nc 0 --no_instance --save_epoch_freq 2 --lr=0.0001 --resize_or_crop none

