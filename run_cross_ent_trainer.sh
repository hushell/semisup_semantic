python train.py --name city256_ce --output_nc 20 --loss cross_ent --dataset cityscapesAB --which_model_netG resnet_softmax_9blocks --batchSize 1 --widthSize 256 --heightSize 256 --targetSize 286 --phase train --niter 100 --niter_decay 160 --optim_method adam --lr 0.0002 --lr_scheme lut --port 8097 --gpu_ids 1