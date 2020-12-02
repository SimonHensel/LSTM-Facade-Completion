nvidia-docker run --user $(id -u):$(id -g) -it --rm -v $PWD:$PWD -w $PWD tensorflow/tensorflow:1.12.0-gpu-py3 python ./source/trainer.py \
--model_type MDMD_LSTM --loss_type DEFAULT --checkpoint_path checkpoints/checkpoint_mdmdlstm/model.ckpt
#nvidia-docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:devel-gpu ./resnet_cifar.py