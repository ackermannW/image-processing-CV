# Digital image processing in OpenCV using Python
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23#ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=white)

This repository shows usage of OpenCV Python package for 
digital image processing. 

## Linux Setup

1. Clone the repository:

``` sh 
git clone https://github.com/ackermannW/image-processing-CV
cd image-processing-CV
```

2. Run the bash setup script `setup.sh`. This script downloads `miniconda`, installs it and creates a python environment from the `environment.yml` file.
When prompted accept the T&C by pressing `a`.

``` sh
chmoa a+x ./setup.sh
./setup.sh
```

3. Activate the environment by using

``` bash
conda activate image-processing-cv
```

## Windows Setup

1. Make sure that GIT is installed.

2. Clone the repository:

``` sh 
git clone https://github.com/ackermannW/image-processing-CV
cd image-processing-CV
```

3. Run the powershell setup script `setup.ps1`. This script downloads `miniconda`, installs it and creates a python environment from the `environment.yml` file.
When prompted accept the T&C by pressing `a`.

``` sh
.\setup.ps1
```
Note that it is also recommended to utilize Windows subsystem for Linux on Windows OS.

# Contents

1. Basic image transformations
2. Image space filtering 
3. Image filtering in frequency domain
4. Edge detection
5. Histogram and thresholding 
6. Image segmentation
7. Classification and object detection

# GPU setup 
It is recommended to utilize GPU acceleration to speed up deep learning tasks.
Instructions for CUDA cores can be [here](https://www.tensorflow.org/install/pip) 
in the official Tensorflow documentation.
Note that it is recommended to use WSL on Widnows OS.
