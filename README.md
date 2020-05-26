# EBSNetMEFNet

This repo includes the source code of the paper: "[Learning a Reinforced Agent for Flexible Exposure Bracketing Selection](CVPR 2020) by Zhouxia Wang, Jiawei Zhang, Mude Lin, Jiong Wang, Ping Luo, Jimmy Ren.

## Quick Test

The code is tested on 64 bit Linux (Ubuntu 14.04 LTS), and besed on Pytorch 0.4.1 with Python 2.7.

1. Clone this github repo

        git clone https://github.com/wzhouxiff/EBSNetMEFNet.git
        cd EBSNetMEFNet
    
2. Download models and testset from [Baidu Drive](https://pan.baidu.com/s/1o39r3Mmj523IJT6e7YcFFQ) (extraction code: jqfp). Models are in folder *checkpoints* which testset is in folder *testset*.

3. Update scripts/test.sh with your path.
                
                usage: test.py [-h] [--data-type] [--results PATH] [--score-path PATH]
                DIR DIR

                PyTorch EBSNetMEFNet

                positional arguments:
                DIR                       path to dataset
                DIR                       path to models

                optional arguments:
                -h, --help                show this help message and exit
                --data-type               'night' or 'day'
                --results                 path to save results
                --score-path              path to save psnr and ssim
                
4. Run scripts/test.sh.

                sh scripts/test.sh

## EBSNet v.s. MEFNet

**EBSNet** - **E**xposure **B**racketing **S**election **N**etwork: Used for exposure bracketing selection by analyzing both the illumination and semantic information of an auto-exposure preview image and Learnt via RL which rewarded by **MEFNet**.

**MEFNet** - **M**ulti-***E*xpusre **F**usion **N**etwork: Used for fusing the selected exposure bracketing predicted by **EBSNet**.

This two networks can be trained jointly.

<img src="images/framework.jpg">

## Dataset

* x: AE image
* z<sub>0</sub> ~ z<sub>9</sub>: exposure sequence
* zz<sub>H</sub>: generated HDR image
* [testset](https://pan.baidu.com/s/1o39r3Mmj523IJT6e7YcFFQ) -  extraction code: jqfp

<img src="images/samples.jpg">

<!-- ## Dataset
## Models && object boxes && adjacency matrices
Models, object boxes and ajacency matrices are in [HERE](https://pan.baidu.com/s/13tvWT5FmfvIFaBRE9nq1WQ).

## Usage
    usage: test.py [-h] [-j N] [-b N] [--print-freq N] [--weights PATH]
               [--scale-size SCALE_SIZE] [--world-size WORLD_SIZE] [-n N]
               [--write-out] [--adjacency-matrix PATH] [--crop-size CROP_SIZE]
               [--result-path PATH]
               DIR DIR DIR

    PyTorch Relationship

    positional arguments:
      DIR                       path to dataset
      DIR                       path to feature (bbox of contextural)
      DIR                       path to test list

    optional arguments:
      -h, --help                show this help message and exit
      -j N, --workers N         number of data loading workers (defult: 4)
      -b N, --batch-size N      mini-batch size (default: 1)
      --print-freq N, -p N      print frequency (default: 10)
      --weights PATH            path to weights (default: none)
      --scale-size SCALE_SIZE   input size
      --world-size WORLD_SIZE   number of distributed processes
      -n N, --num-classes N     number of classes / categories
      --write-out               write scores
      --adjacency-matrix PATH   path to adjacency-matrix of graph
      --crop-size CROP_SIZE     crop size
      --result-path PATH        path for saving result (default: none)

## Test
Modify the path of data before running the script.

    sh test.sh
    
## Result

PISC: Coarse-level

Methods|Intimate|Non-Intimate|No Relation|mAP
-|-|-|-|-
Union CNN  | 72.1 | 81.8 | 19.2| 58.4
Pair CNN  | 70.3 | 80.5 | 38.8 | 65.1
Pair CNN + BBox + Union  | 71.1 | 81.2 | 57.9 | 72.2
Pair CNN + BBox + Global | 70.5 | 80.0 | 53.7 | 70.5
Dual-glance | 73.1 | **84.2** | 59.6 | 79.7 | 35.4 | 79.7
Ours | **81.7** | 73.4 | **65.5** | **82.8**

PISC: Fine-level

Methods|Friends|Family|Couple|Professional|Commercial|No Relation|mAP
-|-|-|-|-|-|-|-
Union CNN | 29.9 | 58.5 | 70.7 | 55.4 | 43.0 | 19.6 | 43.5
Pair CNN  | 30.2 | 59.1 | 69.4 | 57.5 | 41.9 | 34.2 | 48.2
Pair CNN + BBox + Union  | 32.5 | 62.1 | 73.9 | 61.4 | 46.0 | 52.1 | 56.9
Pair CNN + BBox + Global | 32.2 | 61.7 | 72.6 | 60.8 | 44.3 | 51.0 | 54.6
Dual-glance | 35.4 | **68.1** | **76.3** | 70.3 | **57.6** | 60.9 | 63.2
Ours | **59.6** | 64.4 | 58.6 | **76.6** | 39.5 | **67.7** | **68.7**

PIPA-relation: 

Methods   | accuracy 
-|-
Two stream CNN | 57.2
Dual-Glance | 59.6 
Ours  | **62.3**

## Citation
    @inproceedings{Wang2018Deep,
        title={Deep Reasoning with Knowledge Graph for Social Relationship Understanding},
        author={Zhouxia Wang, Tianshui Chen, Jimmy Ren, Weihao Yu, Hui Cheng, Liang Lin},
        booktitle={International Joint Conference on Artificial Intelligence},
        year={2018},
    }

## Contributing
For any questions, feel free to open an issue or contact us (zhouzi1212@gmail.com)
 -->
