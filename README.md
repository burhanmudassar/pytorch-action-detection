# Spatio-Temporal Action Detection : Pytorch

[](assets/009.gif)

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

```
./setup.sh
```
  
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

To test the model run test.py with the specified arguments. You can specify the checkpoint iterations to test in the config file or select a single checkpoint by specfying `test_scope`. By default, it will save the un-processed tubelets for each frame alongwith 2D NMSed frame
detections. 

```Shell
python3 test.py 
        --cfg=cfgs/exp_cfgs/act_vgg16.yml
        --dataset=ucf24
        --split=1
        --dataset_dir=data/ 
        --input_type=rgb 
        --phase test
```

Once the tubelets are generated, run eval_utils.py with the specified arguments. It will evaluate frame level AP, MABO, CLASSIF
followed by building action tubes and then evaluating video mAP. Multithreading is implemented which considerably reduces the tube
building time.

```Shell
python3 eval_utils.py 
--c results/checkpoints/new_models/act_mobilenetv2/ 
--eval_iter 30000 
--K 2 
--dataset ucfsports 
--split 0 
--path=data/ucfsports/ 
--eval_mode rgb
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
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td align="left">Mobilenet V2</td> 
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td align="left"> ResNet-50 </td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>
 
## TODO

Pull Requests Welcome!

- [ ] Add a switch to disable/enable wandb logging
- [x] Call Evaluator directly from test
- [ ] Remove redundant methods from datasets (better way to init, )
- [x] Remove redundant NMS step in eval_utils
- [ ] Train and upload models
- [ ] Add K to the output directory name
- [ ] Combine command overrides from train, test and visualize into a single file
- [ ] Add support for AVA


## Acknowledgements
The code in this branch is heavily adopted and repurposed for
our work from the following codebases. In addition the core structure for the repo is adapted from [ssds.pytorch].(https://github.com/ShuangXieIrene/ssds.pytorch)

Big thanks to the following.
- [Real Time Online Activity Detection](https://github.com/gurkirt/realtime-action-detection) [1]
- [Action Tubelet Detector](https://github.com/vkalogeiton/caffe/tree/act-detector) [2]
- [ssds.pytorch](https://github.com/ShuangXieIrene/ssds.pytorch)

## References
- [1] G. Singh, S Saha, M. Sapienza, P. H. S. Torr and F Cuzzolin. Online Real time Multiple Spatiotemporal Action Localisation and Prediction. ICCV, 2017.
- [2] Kalogeiton, V., Weinzaepfel, P., Ferrari, V. and Schmid, C., 2017. Action Tubelet Detector for Spatio-Temporal Action Localization. ICCV, 2017.
- [3] A huge thanks to Max deGroot, Ellis Brown for Pytorch implementation of [SSD](https://github.com/amdegroot/ssd.pytorch)
 
