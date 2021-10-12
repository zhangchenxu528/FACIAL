############## To train images at 2048 x 1024 resolution after training 1024 x 512 resolution models #############
##### Using GPUs with 12G memory (not tested)
# Using labels only
export CUDA_VISIBLE_DEVICES=1
python train.py --name everybody_dance_now_temporal --load_pretrain checkpoints/everybody_dance_now_temporal --dataroot ./datasets/dance/ --netG local --ngf 32 --num_D 3 --tf_log --niter_fix_global 20 --label_nc 0 --no_instance --save_epoch_freq 2 --resize_or_crop none --fineSize 1024
