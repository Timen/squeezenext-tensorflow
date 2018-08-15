## _SqueezeNext Tensorflow:_ A tensorflow Implementation of SqueezeNext 
This repository contains a tensorflow implementation of SqueezeNext, a hardware-aware neural network design.

    @article{DBLP:journals/corr/abs-1803-10615,
      author    = {Amir Gholami and
                   Kiseok Kwon and
                   Bichen Wu and
                   Zizheng Tai and
                   Xiangyu Yue and
                   Peter H. Jin and
                   Sicheng Zhao and
                   Kurt Keutzer},
      title     = {SqueezeNext: Hardware-Aware Neural Network Design},
      journal   = {CoRR},
      volume    = {abs/1803.10615},
      year      = {2018},
      url       = {http://arxiv.org/abs/1803.10615},
      archivePrefix = {arXiv},
      eprint    = {1803.10615},
      timestamp = {Wed, 11 Apr 2018 17:54:17 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1803-10615},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
## Pretrained model:
Using the data from the paper, original caffe version on github and other sources I tried to recreate the 1.0-SqueezeNext-23 model as closely as possible. The model
achieved a 55% top 1 accuracy on validation set and a 79% top 5 accuracy on the validation set. This is about 3% under the reported results. Causes for this
could be that the network was trained with a batch size of 256 instead of 1024, and because of the the number of steps required for 120 epochs increased 4 fold.
The learning rate schedule was modified to account for the lower batch size and the increased number of steps. It could have also been caused by an error in the implementation of the combination of the residual and the network. In the caffe models both the network and residual have the ReLU activation function applied before being added together, in this pretrained model the ReLu is applied after adding them together similar to the original ResNet structure.

These changes (reflected in the v_1_0_SqNxt_23_mod config) might be the reason for the decreased the final validation accuracy, but due to the fact that one training session takes 4 days on a gtx1080ti only this modified version
was trained and can be downloaded from here [v_1_0_SqNxt_23_mod](https://drive.google.com/open?id=1zOjSQR5KLHZyd7Y-VuJBNVw7K0_8UyOk).


    
## Installation:
This implementation was made using version 1.8 of the tensorflow api. Earlier versions are untested, and may not work due to the
use of some recently added functions for data loading and processing. 

- Make sure tensorflow 1.8 or higher is by running:
    ```Shell
    python -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 2
    python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3
     ```
  And verifying the output is 1.8.0 or above.
  
- Clone this repository:

  ```Shell
  git clone https://github.com/Timen/squeezenext-tensorflow.git
  ```
- Install requirements:
    ```Shell
    pip install -r requirements.txt
    ```

## Preparing the Dataset:
SqueezeNext like most other classifiers is trained with the ImageNet dataset (http://www.image-net.org/). One can download the 
data from the afromentioned website, however this can be rather slow so I recommend downloading the dataset using torrents
available on (http://academictorrents.com/) namely:

[Training Images](http://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2/tech),
[Validation Images](http://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5/tech&dllist=1)
and [Bounding Box Annotations](http://academictorrents.com/details/28202f4f8dde5c9b26d406f5522f8763713e605b/tech&dllist=1)

Please note one should still abide by the original License agreement of the Imagenet dataset. After downloading these files please perform the following steps to prepare the dataset.

- Create a directory used for processing and storing the dataset.
    Please note you should have at least around 500 GB of free space available on the drive you are processing the dataset. 
    Once this directory is created copy the 3 files downloaded earlier to the root of this directory from that directory execute
    the following command:
  ```Shell
     export DATA_DIR=$(pwd)
     
     ```
    

- Execute the following command from this projects root folder:
  ```Shell
  bash datasets/process_downloaded_imagenet.sh $DATA_DIR
  
  ```
  Where $DATA_DIR is the root of the directory created to hold the 3 downloaded files.
  
- Wait for processing to finish.
    The script process_downloaded_imagenet.sh will automatically extract the tarballs and process al the data into tf-records.
    The whole process can take between 2 and 5 hours depending on how fast the hard drive and cpu are.

## Training:
After installation and dataset preparation one only needs to execute the run_train.sh script to start training. By executing
the following command from the projects root folder:

```Shell
bash run_train.sh
```
This will start training the 1.0 v1 version of squeezenext for 120 epochs with batch size 256. With a GTX1080Ti this training
will take up to 4 days. If your gpu has a smaller memory capacity then a gtx1080ti you probably need to lower the batch size
to be able to run the training. 


## Prediction:
Prediction is done using the predict.py script, to run it you give it a path to a jpeg image and pass the directory containing
a trained model in the model_dir argument.

```Shell
python predict.py ./tabby_cat.jpg --model_dir ?TRAIN_DIR from the run_train.sh or pretrained model directory?
```

This script will load the image and run the classifier on it, the output is the top 5 human readable class labels.

## Modifying the hyper parameters:
The batch size number of epochs and some other settings regarding epoch size, file location etc. can be passed as command 
line arguments to the train.py script. 

Switching between specific configurations such as the grouped convolution and the non grouped
convolution versions of squeezenext should be done by selecting which config file from the configs folder to use. This can be done
by passing the file name without the .py as the command line argument --configuration. It is easy to add your own configuration just
copy one of the other configs and rename the file to something new (keep in mind it will be imported in python so stick to numbers letters
and under scores). You can then change the parameters in the file to customize your own config and pass the new file name as --configuration parameter.(the python scripts in configs are automatically imported)



