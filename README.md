# SSAST: Self-Supervised Audio Spectrogram Transformer
 - [Introduction](#Introduction)
 - [Citing](#Citing)  
 - [Getting Started](#Getting-Started)
 - [SSAST Model](#SSAST-Model) 
 - [Data Preparation](#Data-Preparation)
 - [Self-Supervised Pretraining](#Self-Supervised-Pretraining)  
 - [Fine-tuning On Downstream Tasks](#Fine-tuning-On-Downstream-Tasks)
 - [Pretrained Models](#Pretrained-Models)
 - [Contact](#Contact)

## Introduction  
Here the SSAST has been adapted for analysis in the CIAB project. Please find the main repo [here](https://github.com/YuanGongND/ssast)

<p align="center"><img src="https://github.com/YuanGongND/ssast/blob/main/figure/10854_ssast.png?raw=true" alt="Illustration of AST." width="800"/></p>

The repository features an adapted copy of the **Self-Supervised Audio Spectrogram Transformer (SSAST)** proposed in the AAAI 2022 paper [SSAST: Self-Supervised Audio Spectrogram Transformer](https://arxiv.org/abs/2110.09784) ([Yuan Gong](https://yuangongnd.github.io/), [Cheng-I Jeff Lai](http://people.csail.mit.edu/clai24/), [Yu-An Chung](http://people.csail.mit.edu/andyyuan/), [James Glass](https://www.csail.mit.edu/person/jim-glass); MIT CSAIL).  

SSAST is the first **patch-based** joint discriminative and generative self-supervised learning framework, and also the first self-supervised learning framework for AST. SSAST significantly boosts AST performance on all downstream tasks we evaluated with an average improvement of 60.9%, leading to similar or even better results than a supervised pretrained AST. SSAST can be used as a drop-in replacement of previous ImageNet (supervised) pretrained AST, and has the advantage of 1) no labeled data is used; 2) flexible patch size and shape, ImagenNet pretraining only supports square patches; and 3) better performance on many tasks, in particular speech tasks.

## Citing  
Please cite the paper if you find this repository useful. 
```  
@article{gong2021ssast,
  title={SSAST: Self-Supervised Audio Spectrogram Transformer},
  author={Gong, Yuan and Lai, Cheng-I Jeff and Chung, Yu-An and Glass, James},
  journal={arXiv preprint arXiv:2110.09784},
  year={2021}
}
```  

  
## Getting Started  

### Prepare the Environment
Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd ast/ 
python3 -m venv venvssast
source venvssast/bin/activate
pip install -r requirements.txt 
```
### Replicate the CIAB analysis
First cd to /src/finetune/ciab/ then extract the CIAB features:
```python
python prep_ciab.py
```
When this has run, to finetune a model on the task of COVID-19 detection from audio run:
```bash
sh run_ciab.sh
```


## Pretrained-Models

We provide the following self-supervised pretrained models. All models are trained with full AudioSet + Librispeech. Click the model name to download. Tiny model should be able to pretrain and fine-tune on an 8GB GPU with a reasonable batch size.

For best performance, you should use either of the following models, patch-based AST is better for audio tasks, frame-based AST is better for speech tasks.

| Model Name                                                                                        | Data  | Pretrain fshape | Pretrain tshape | #Masked   Patches | Model Size  | Avg Audio  Performance | Avg Speech  Performance |
|---------------------------------------------------------------------------------------------------|-------|-----------------|-----------------|-------------------|-------------|------------------------|-------------------------|
| [SSAST-Base-Patch-400](https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1)   | AudioSet + Librispeech | 16              | 16              | 400               | Base (89M)  | 59.9                   | 79.5                    |
| [SSAST-Base-Frame-400](https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1)   | AudioSet + Librispeech | 128             | 2               | 400               | Base (89M)  | 57.6                   | 84.0                    |

Following models does not have best performance, we release them for analysis purpose and low-resource devices.

| Model Name                                                                                        | Data  | Pretrain fshape | Pretrain tshape | #Masked   Patches | Model Size  | Avg Audio  Performance | Avg Speech  Performance |
|---------------------------------------------------------------------------------------------------|-------|-----------------|-----------------|-------------------|-------------|------------------------|-------------------------|
| [SSAST-Base-Patch-250](https://www.dropbox.com/s/mxrm9qog6aj8hif/SSAST-Base-Patch-250.pth?dl=1)   | AudioSet + Librispeech | 16              | 16              | 250               | Base (89M)  | 58.6                   | 79.5                    |
| [SSAST-Base-Frame-250](https://www.dropbox.com/s/4e6l7ulhwrfoana/SSAST-Base-Frame-250.pth?dl=1)   | AudioSet + Librispeech | 128             | 2               | 250               | Base (89M)  | 55.6                   | 81.6                    |
| [SSAST-Small-Patch-400](https://www.dropbox.com/s/i24w446rl9pkf05/SSAST-Small-Patch-400.pth?dl=1) | AudioSet + Librispeech | 16              | 16              | 400               | Small (23M) | 58.1                   | 78.2                    |
| [SSAST-Tiny-Patch-400](https://www.dropbox.com/s/fkbtf78y94113wz/SSAST-Tiny-Patch-400.pth?dl=1)   | AudioSet + Librispeech | 16              | 16              | 400               | Tiny (6M)   | 53.3                   | 75.7                    |
| [SSAST-Tiny-Frame-400](https://www.dropbox.com/s/rx7g60ruzawffzv/SSAST-Tiny-Frame-400.pth?dl=1)   | AudioSet + Librispeech | 128             | 2               | 400               | Tiny (6M)   | 47.8                   | untested                |

Above links are dropbox direct download links (i.e., wget works). For those don't have access to Dropbox, use a VPN or use the [OneDrive Links](https://mitprod-my.sharepoint.com/:f:/g/personal/yuangong_mit_edu/EuAuTEZNYPhOmlLFFjRFvGUBcgnIXBqFgFE33GDK69h-Zw?e=d3MEgT).

 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email harry.coppock@imperial.ac.uk
