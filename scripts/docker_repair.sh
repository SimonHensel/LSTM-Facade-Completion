reset
nvidia-docker run --user $(id -u):$(id -g) -it --rm -v $PWD:$PWD -w $PWD \
tensorflow/tensorflow:1.12.0-gpu-py3 python3 source/repair_detections.py --model_type MD_LSTM \
--threshold 0.5 \
--max_size 25 \
--detections_dir data/repair_test/ \
--image_dir data/repair_test/ \
--output_dir outputs/repair_test/ 