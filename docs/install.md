## Getting Started

Start by cloning this repository:
```
git clone git@github.com:huangyangyi/ELICIT.git
cd ELICIT
```

## Environment

We use the following environment in our experiments: 
- Ubuntu 18.04
- CUDA 11.3
- 1 GPU with >12G memory for rendering
- 4 NVIDIA V100 GPUs(32G) for training
- Python=3.7
- PyTorch=1.12.0
- torchvision=0.13.0


### Installation

1. Setting the `CUDA_HOME` environment variable.
2. Install `PyTorch` and `torchvision` following [the official tutorial](https://pytorch.org/get-started/).
3. Install dependencies by 
```bash
pip install -r requirements.txt
```

## Download SMPL model

Please register [SMPLify](https://smplify.is.tue.mpg.de/) at first, and then download mpips_smplify_public_v2.zip from the webpage, unzip and copy the neural SMPL model:
```bash
SMPL_DIR=/path/to/smpl
MODEL_DIR=$SMPL_DIR/smplify_public/code/models
cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models
```
You can follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.