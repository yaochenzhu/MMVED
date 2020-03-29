# MMVED: Multimodal Variational Encoder Decoder Framework for Micro Video Popularity Prediction Tasks

This is our implementation of MultiModal Variational Encoder-Decoder Network (MMVED) for micro-video popularity prediction, which contains two parts:

- Micro-video popularity regression on NUS dataset
- Micro-video temporal popularity prediction on Xigua dataset

Each part contains everything required to train or test the corresponding MMVED model. For the Xigua datset we collect, we release the data as well to faciliate future research.

## Architecture
![](https://github.com/yaochenzhu/MMVED/blob/master/framework.png)

## Environment

- python == 3.6.5
- numpy == 1.16.1
- tensorflow == 1.13.1
- tensorflow-probability == 0.6.0

## Datasets

### The Xigua dataset

The Xigua micro-video temporal popularity prediction dataset we collect is available [here](https://drive.google.com/open?id=1-q46LeBvi1-z7riJB28tDqk-hM5eu8g_). Download the whole data folder and put them in the xigua directory. Descriptions of the files are as follows:

- **`resnet50.npy`**:
   (N×128). Visual features extracted by ResNet50 pre-trained on ImageNet (PCA dimension-reduced).
 
- **`audiovgg.npy`**:
   (N×128). Aural features extracted by AudioVGG pre-trained on AudioSet.
 
- **`fudannlp.npy`**:
   (N×20). Textual features extracted by the FudanNLP toolkit.

- **`social.npy`**:
   (N×3). Social features crawled from the user attributes.

- **`len_9/target.npy`**: (N×9×2). Popularity groundtruth (0-axis) and absolute time (1-axis) at each timestep.

- **`split/0-4/{train, val, test}.txt`**: Train, val and test samples for five splits of datasets used in our paper.

### The NUS dataset

The original NUS dataset can be found [here](https://acmmm2016.wixsite.com/micro-videos) which was released together with the TMALL model in this [paper](http://www.nextcenter.org/wp-content/uploads/2017/06/MicroTellsMacro.JournalNExT.pdf). The descriptions of files in the dataset directory in the NUS folder are as follows:

- **`vid.txt`**:  The ids of the micro-videos that we were able to download successfully at the time of our experiment.

- **`split/0-4/{train, val, test}.txt`**: five splits of datasets we used in our paper.

## Examples to run the Codes

The basic usage of the codes for training and testing MMVED model on both Xigua and NUS dataset is as follows:

- **For training**: 

	```python train.py --lambd [LAMBDA] --split [SPLIT]```
- **For testing**:

	```python predict.py --model_path [PATH_TO_MODEL] --test_mods [VATS]```

For more advanced arguments, run the code with --help argument.

##

### **If you find our codes and dataset helpful, please kindly cite the following papers. Thanks!**

Fullfledged version: [Here]() ; www 2020 short paper: [Here](https://drive.google.com/open?id=15sltJ8-JU9eePJtY6bvt5ycevOlgy8QT)

	@article{mmved-fullfledged,
	  title={Predicting the Popularity of Micro-videos with Multimodal Variational Encoder-Decoder Framework},
	  author={Zhu, Yaochen and Xie, Jiayi and Chen, Zhenzhong},
	  booktitle={arXiv preprint arXiv:},
	  year={2020},
	}	

	@inproceedings{mmved-www2020-preliminary,
	  title={A Multimodal Variational Encoder-Decoder Framework for Micro-video Popularity Prediction},
	  author={Xie, Jiayi and Zhu, Yaochen and Zhang, Zhibin and Peng, Jian and Yi, Jing and Hu, Yaosi and Liu, Hongyi and Chen, Zhenzhong},
	  booktitle={The World Wide Web Conference},
	  year={2020},
	}

	
