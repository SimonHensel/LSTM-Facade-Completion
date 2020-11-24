nvidia-docker run --user $(id -u):$(id -g) -it --rm -v $PWD:$PWD -w $PWD tensorflow/tensorflow:1.12.0-gpu-py3 python ./source/evaluate.py \
--model_type MDMD_LSTM  --checkpoint_path ./checkpoints/checkpoint_mdmdlstm/model.ckpt --hidden_size 2500 \
| tee evaluation/evaled_mdmdlstm25.txt

#1250

#nvidia-docker run --user $(id -u):$(id -g) -it --rm -v $PWD:$PWD -w $PWD tensorflow/tensorflow:1.12.0-gpu-py3 python ./source/evaluate.py \
#--model_type MD_LSTM  --checkpoint_path ./checkpoints/checkpoint_mdlstm/model.ckpt --hidden_size 2500 \
#| tee evaluation/evaled_mdlstm25.txt
