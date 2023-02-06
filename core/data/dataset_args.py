from configs import cfg
import os

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394']
    subjects = subjects + [sub + '_smpl' for sub in subjects]

    if cfg.category in ['human_nerf', 'elicit'] and (cfg.task in ['zju_mocap', 'mydemo', 'fashion']):
        for sub in subjects:
            dataset_attrs.update({
                f"zju_{sub}_train": {
                    "dataset_path": f"dataset/zju_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"zju_{sub}_test": {
                    "dataset_path": f"dataset/zju_mocap/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
            })
    h36m_subjects = ['s1', 's5', 's6', 's7', 's8', 's9', 's11']
    h36m_subjects = h36m_subjects + [sub + '_smpl' for sub in h36m_subjects]
    if cfg.category in ['human_nerf', 'elicit'] and cfg.task == 'h36m':
        for sub in h36m_subjects:
            dataset_attrs.update({
                f"h36m_{sub}_train": {
                    "dataset_path": f"dataset/h36m/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"h36m_{sub}_test": {
                    "dataset_path": f"dataset/h36m/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
            })

    thuman_subjects = ['hyd_1_M', 'lw_2_F', 'sty_1_M', 'xsx_2_M', 'xyz_1_F']
    thuman_subjects = thuman_subjects + [sub + '_smpl' for sub in thuman_subjects]
    if cfg.category in ['human_nerf', 'elicit'] and cfg.task == 'thuman':
        for sub in thuman_subjects:
            dataset_attrs.update({
                f"thuman_{sub}_train": {
                    "dataset_path": f"dataset/thuman/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"thuman_{sub}_test": {
                    "dataset_path": f"dataset/thuman/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
            })
    if cfg.category in ['human_nerf', 'elicit'] and (cfg.task in ['mydemo', 'fashion']):
        subjects = os.listdir('dataset/{}/'.format(cfg.task))
        for sub in subjects:
            dataset_attrs.update({
                f"{cfg.task}_{sub}_train": {
                    "dataset_path": f"dataset/{cfg.task}/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"{cfg.task}_{sub}_test": {
                    "dataset_path": f"dataset/{cfg.task}/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
            })

    if cfg.category in ['human_nerf', 'elicit'] and cfg.task == 'wild':
        dataset_attrs.update({
            "monocular_train": {
                "dataset_path": 'dataset/wild/monocular',
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "monocular_test": {
                "dataset_path": 'dataset/wild/monocular',  
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'wild'
            },
        })


    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
