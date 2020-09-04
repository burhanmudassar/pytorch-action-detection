# Spatio-Temporal Action Detection : Pytorch

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#models'>Models</a>
- <a href='#train'>Train</a>
- <a href='#test'>Test</a>
- <a href='#visualizing'>Visualizing</a>
- <a href='#performance'>Performance</a>
- <a href='#todo'>TODO</a>
- <a href='#citation'>Citation</a>
- <a href='#references'>Reference</a>

## Installation
- Dependencies
  * Python 3.6
  * Pytorch 1.4.0
  * OpenCV Python
  
## Datasets
- UCF101-24
- JHMDB-21
- UCFSports

To download one of these datasets run the following command with `DATASET_NAME` set to one of the options `ucf24, ucfsports, jhmdb`
```Shell
./data/get_dataset.sh <DATASET_NAME>
```

## Models
- The project is structured so that adding a new backbone and/or model into the framework is 
easy.
- Right now, ACT with VGG16, ResNet50 and Mobilenet V2 backbones are supported. PRs are welcome for more models and/or backbones
- Coming soon, I3D backbone

## Train

To train the model run train.py and customize the arguments. A configuration file needs to be specified. The config file
is a nested YAML file. 

- To train on flow-inputs just set input_type to "brox"
- To train on fusion models, set input_type to "fusion". Right now only decision fusion of scores is supported

```Shell
python3 train.py 
        --cfg=cfgs/exp_cfgs/act_vgg16.yml
        --dataset=ucf24
        --split=1
        --dataset_dir=data/ 
        --input_type=rgb 
        --max_iter=120000 
```

### Logging

Supports experiment logging through both tensorboard as well as Weights and Biases.

## Test

To test the model run test.py with the specified arguments. You can specify the checkpoint iterations to test in the config file or select a single checkpoint by specfying `test_scope`.

```Shell
python3 test.py 
        --cfg=cfgs/exp_cfgs/act_vgg16.yml
        --dataset=ucf24
        --split=1
        --dataset_dir=data/ 
        --input_type=rgb 
        --phase test
```

## Visualizing

To visualize the outputs of the model run visualize.py

## Evaluation of Action Detection Metrics

The evaluation of video AP requires building action-tubes in an offline manner. Much of the code is adopted from
the ACT Caffe repo with modifications made to support multi-threading (as it involves opening a lot of raw result files).
The evaluation will return the following metrics

- Frame AP
- Classification Accuracy
- Mean Average Best Overlap (MABO)
- Video AP


## Performance
##### UCFSports Frame AP
<table style="width:100% th">
  <tr>
    <td>Model </td>
    <td>ACT RGB</td> 
    <td>ACT Brox FLO</td> 
    <td>ACT Mean Fusion</td> 
  </tr>
  <tr>
    <td align="left">VGG-16</td> 
    <td>76.91</td>
  </tr>
  <tr>
    <td align="left">Mobilenet V2</td> 
    <td>76.91</td>

  </tr>
  <tr>
    <td align="left"> ResNet-50 </td>
    <td>76.50</td>
  </tr>
</table>
 
## TODO



## Citation
If this work has been helpful in your research please consider

## Acknowledgements
The code in this branch is heavily adopted and repurposed for
our work from the following codebases. In addition the core structure for the repo is adapted from ssds.pytorch
Big thanks to the following.
- Real Time Online Activity Detection (Gurkirt Singh)
- Action Tubelet Detector (Vicky Kalogeiton)

## References
- [1] Wei Liu, et al. SSD: Single Shot MultiBox Detector. [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [2] S. Saha, G. Singh, M. Sapienza, P. H. S. Torr, and F. Cuzzolin, Deep learning for detecting multiple space-time action tubes in videos. BMVC 2016 
- [3] X. Peng and C. Schmid. Multi-region two-stream R-CNN for action detection. ECCV 2016
- [4] G. Singh, S Saha, M. Sapienza, P. H. S. Torr and F Cuzzolin. Online Real time Multiple Spatiotemporal Action Localisation and Prediction. ICCV, 2017.
- [5] Kalogeiton, V., Weinzaepfel, P., Ferrari, V. and Schmid, C., 2017. Action Tubelet Detector for Spatio-Temporal Action Localization. ICCV, 2017.
- [Original SSD Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thanks to Max deGroot, Ellis Brown for Pytorch implementation of [SSD](https://github.com/amdegroot/ssd.pytorch)
 
