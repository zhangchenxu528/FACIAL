train3new
python train.py --name train3 --model pose2vid --dataroot ./datasets/train3/ --netG local --ngf 32 --num_D 3 --tf_log --niter_fix_global 0 --label_nc 0 --no_instance --save_epoch_freq 2 --lr=0.0001 --resize_or_crop resize --no_flip --verbose --n_local_enhancers 1 --continue_train

python test_video.py --name train3 --model pose2vid --dataroot ./datasets/train3/ --which_epoch latest --netG local --ngf 32 --label_nc 0 --n_local_enhancers 1 --no_instance --resize_or_crop resize