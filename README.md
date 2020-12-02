LSTM Facade Completion
======================

Implementation of the LSTM networks for facade completion described in the paper PLACEHOLDER

This repository is still under cleanup. View with care.

Usage
============
Used with the following libraries and versions
* Python 3.5.2 
* h5py                2.8.0
* hilbertcurve        1.0.3
* kiwisolver          1.0.1
* matplotlib          3.0.1
* numpy               1.15.4
* opencv-python       4.2.0.34
* pandas              0.23.4
* scikit-learn        0.20.0
* scipy               1.1.0
* sklearn             0.0
* tensorboard         1.12.0
* tensorflow-gpu      1.12.0

It is recommendet to use the openly available docker image *tensorflow/tensorflow:1.12.0-gpu-py3*. The scripts for training and evaluation are coded with this docker image in mind. If some libraries are missing, they can be installed additionally using the *docker_shell.sh* script and commiting the changes after install.

Scripts and their functions:

* *docker_eval.sh* - Evaluation using docker.
* *docker_shell.sh* - Get shell acces on docker image. Can be used to install missing libraries or software. 
* *docker_train.sh* - Starts training using docker image.
* *train.sh* - Sarts training in shell.
* *eval_all.sh* - Shell for multiple evaluations (uncomment wanted evaluations). 

For training a GPU with 24GB memory is recommendet. We used a Nvidia P6000 for the results presented in the paper.

Execute scripts from the main directory.
For example:
```
./scripts/docker_train.sh
```

Model Options
=======

Argument `--model_type` sets the network architecture to use.
Main options are:
 * MD_LSTM
 * MDMD_LSTM
 * QRNN

Checkpoints
===========

Checkpoints can be downloaded at [google drive](https://drive.google.com/file/d/1Hbg1I2JHbfP0HlLyXMHSZYnl_xpwwOl1/view?usp=sharing)

Data Used
=========

The project is implemented, to be compatible with the following data sets:

* [CMP facade database](http://cmp.felk.cvut.cz/~tylecr1/facade/)
* [ICG Graz50](https://people.ee.ethz.ch/~rhayko/paper/cvpr2012_riemenschneider_lattice/)


References
==========
Code from [philipperemy](https://github.com/philipperemy/tensorflow-multi-dimensional-lstm) served as the basis for this project.