## ZJU-Mocap dataset

1. Follow the instructions in [NeuralBody](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) to request for the dataset.

2. Modify the yaml files in `tools/prepare_zju_mocap`, take `387.yaml` as an example, update `zju_mocap_path` to the root path of the downloaded ZJU-Mocap data.

```yaml
dataset:
  zju_mocap_path: /path/to/ZJUMocap
  subject: '387'
  sex: 'neutral'
```

3. Run the scripts of `prepare_data.py` and `render_smpl` to preprocess the dataset and render SMPL meshes for training. For example:

```bash
export PYTHONPATH=/path/to/elicit:$PYTHONPATH
cd tools/prepare_zju_mocap
python prepare_data.py --cfg 387.yaml
python render_smpl.py --cfg 387.yaml
```

## Human3.6M dataset

1. Since the license of the Human3.6M dataset does not allow us to release its data, we cannot publicly release the Human3.6M dataset. Please request the dataset [here](http://vision.imar.ro/human3.6m/description.php) and follow [this instruction](https://github.com/zju3dv/neuralbody/blob/master/tools/custom) to prepare the data in ZJU-Mocap format. You can also refer to this [page](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md) for the Human3.6M dataset preprocessed by [animabtable_nerf](https://github.com/zju3dv/animatable_nerf).

2. Modify the yaml files in `tools/prepare_h36m`, take `s1.yaml` as an example, update `h36m_path` to the root path of the Human3.6M data.

```yaml
dataset:
  h36m_path: /path/to/H36M/
  subject: 'S1/Posing'
  sex: 'neutral'
```

3. Run the scripts of `prepare_data.py` and `render_smpl` to preprocess the dataset and render SMPL meshes for training. For example:

```bash
export PYTHONPATH=/path/to/elicit:$PYTHONPATH
cd tools/prepare_h36m
python prepare_data.py --cfg s1.yaml
python render_smpl.py --cfg s1.yaml
```

## Customized data

For customized data with single image input, we provided examples from DeepFashion datasets in `dataset/fashion`. Please follow step 1-2 to add new data, and follow step 3 for preprocessing.

1. Estimate SMPL parameters of the subject. Here we use the pose estimation model of [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF) and [PyMAF-X](https://github.com/HongwenZhang/PyMAF/tree/smplx). Please convert the estimated parameters into the same format of CLIFF (refer to `data/fashion/fs2818_m394/gt_params.npz` as an example).

2. Prepare the input image and the subject mask (rescale to 1024*1024), and put the data with the folowing structure.
```
dataset
    └── fashion
        └── ${subject_name}
            ├── gt_image.png # input image
            ├── gt_mask.png # input mask
            └── gt_params.npz # SMPL params
```

3. Edit the yaml file in `tools/prepare_fashion`, here we use the motion sequence of subject 394 in ZJU-Mocap dataset.
```yaml
dataset: # motion data!
  zju_mocap_path: /path/to/ZJUMoCap/ # root of the motion dataset
  subject: 'CoreView_394' # MOTION_ID
  sex: 'neutral'
```
If you want to use a customized motion sequence, please prepare it in the same struction of ZJUMocap:
```
${CUSTOM_ROOT}
    └── ${MOTION_ID}
        ├── annots.npy # annotation of the cameras, it's recommanded to use a similar camera setting of ZJUMocap with 20~30 cameras around the subject
        └── new_params # SMPL annotation of the motion sequence
            ├── 0.npy
            ├── 1.npy
            └── ...
```
Then run the scripts to prepare SMPL meshes renderings.
```bash
python render_smpl.py --cfg ${subject_name}.yaml
```