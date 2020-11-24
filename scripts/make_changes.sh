#sudo nvidia-docker run tensorflow/tensorflow:1.12.0-gpu-py3 pip install --upgrade pip
#sudo nvidia-docker commit ef115cb39b89 tensorflow/tensorflow:1.12.0-gpu-py3
#sudo nvidia-docker run tensorflow/tensorflow:1.12.0-gpu-py3 apt-get update
#sudo nvidia-docker commit ef115cb39b89 tensorflow/tensorflow:1.12.0-gpu-py3
#sudo nvidia-docker run tensorflow/tensorflow:1.12.0-gpu-py3 apt-get install -y libsm6 libxext6 libxrender-dev
#sudo nvidia-docker commit ef115cb39b89 tensorflow/tensorflow:1.12.0-gpu-py3
#sudo nvidia-docker run tensorflow/tensorflow:1.12.0-gpu-py3 pip install numpy opencv-python hilbertcurve
#sudo nvidia-docker commit ef115cb39b89 tensorflow/tensorflow:1.12.0-gpu-py3

#nvidia-docker pull tensorflow/tensorflow:1.12.0-gpu-py3
#docker save -o <path for generated tar file> <image name>
#docker load -i <path to image tar file>
#docker save -o c:/myfile.tar centos:16