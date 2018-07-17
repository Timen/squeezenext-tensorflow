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

Please note one should still abide by the original License agreement of the Imagenet dataset. After downloading these files please the following steps to prepare the dataset.

- Create a directory used for processing and storing the dataset.
    Please note you should have at least around 500 GB of free space available on the drive you are processing the dataset. 
    Once this directory is created copy the 3 files downloaded earlier to the root of this directory.

- Execute the following command from this projects root folder:
  ```Shell
  bash datasets/process_downloaded_imagenet.sh $DATA_DIR
  
  ```
  Where $DATA_DIR is the root of the directory created to hold the 3 downloaded files.
  
- Wait for processing to finish.
    The script process_downloaded_imagenet.sh will automatically extract the tarballs and process al the data into tf-records.
    The whole process can take between 2 and 5 hours depending on how fast the hard drive and cpu are.
    