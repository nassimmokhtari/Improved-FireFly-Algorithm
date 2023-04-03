# Improved FireFly Algorithm


This repository contains the implementation of the improved Firefly algorithm, using a training-free evaluation of the model quality, in order to perform Neural Architecture Search on NAS-BENCH-101 and NAS-BENCH-201. This method can be used to build automatically a  convolutional neural network for any image classification problem. 

Our paper can be found at:

[Improving Neural Architecture Search by Mixing a FireFly algorithm with a Training Free Evaluation](https://www.researchgate.net/publication/362293584_Improving_Neural_Architecture_Search_by_Mixing_a_FireFly_algorithm_with_a_Training_Free_Evaluation)

If you use or build on our work, please consider citing us:

```
@INPROCEEDINGS{9892861,
  author={Mokhtari, Nassim and Nédélec, Alexis and Gilles, Marlène and De Loor, Pierre},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Improving Neural Architecture Search by Mixing a FireFly algorithm with a Training Free Evaluation}, 
  year={2022},
  pages={1-8},
  doi={10.1109/IJCNN55064.2022.9892861}}

```


## Setup
In order to be able to use our implementation, please follow these instructions :

### I) Nas-Bench-101 setup
**1** download "nasbench_only108.tfrecord" from https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord and place it into "api" folder

**2** Install **nasbench** :

    git clone https://github.com/google-research/nasbench
    cd nasbench
    pip install -e .

* **note:** If you are using **tensorflow 2**,  you may have a problem when running nasbench, please refer to this solution  https://github.com/google-research/nasbench/issues/27#issuecomment-805730342

**3** - Install **nasbench_keras** :

    git clone https://github.com/lienching/nasbench_keras
    cd nasbench_keras
    pip install -e .

**4** - Install **xautodl** : 

	pip install xautodl

**5** - Update **numpy** if necessery.

### II) Nas-Bench-201 setup

**1** - download "NAS-Bench-201-v1_1-096897.pth" from https://drive.google.com/open?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_ and place it into "api" folder.

**2** - Install **nasbench201** :

	pip install nas-bench-201 


## Usage
You can start the Neural Architecture Search using the default parameters by running the main.py from the command line :

	python ./main.py
* Samples from **CIFAR-10** are provided in data/samples.pt

You can run on your own dataset **(saved as pytorch tensor)** by using:

	python ./main.py --data_path <PATH_TO_YOUR_DATASET> --input_shape <H> <W> <C> --data_format channels_last --num_labels <#Classes>
* The input shape must be provided in the form of a set of integers separated by a blank, default is 3 32 32.
* the data format must matches the input shape, default is channels_first.


Several parameters can be used to refine the search for a neural architecture. You can find more details about these parameters using :

	python ./main --help
