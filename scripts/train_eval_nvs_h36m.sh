# Novel view synthesis on Human 3.6M
INTV=5

SID=s1
for i in `seq 0 100 999`;
do
python train_partial.py --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i train.start_frame $(expr $i / $INTV) train.end_frame $(expr $i / $INTV + 1) train.single_frame_id `printf "frame_%06d" $(expr $i)` freeview.frame_idx $i| tee logs/train_h36m_${SID}_nvs_$i.log
python run.py --type freeview --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i  bgcolor "[0.,0.,0.]" freeview.frame_idx $(expr $i / $INTV) show_truth True
done;

SID=s5
for i in `seq 0 180 1799`;
do
python train_partial.py --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i train.start_frame $(expr $i / $INTV) train.end_frame $(expr $i / $INTV + 1) train.single_frame_id `printf "frame_%06d" $(expr $i)` freeview.frame_idx $i| tee logs/train_h36m_${SID}_nvs_$i.log
python run.py --type freeview --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i  bgcolor "[0.,0.,0.]" freeview.frame_idx $(expr $i / $INTV) show_truth True
done;

SID=s6
for i in `seq 0 100 999`;
do
python train_partial.py --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i train.start_frame $(expr $i / $INTV) train.end_frame $(expr $i / $INTV + 1) train.single_frame_id `printf "frame_%06d" $(expr $i)` freeview.frame_idx $i| tee logs/train_h36m_${SID}_nvs_$i.log
python run.py --type freeview --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i  bgcolor "[0.,0.,0.]" freeview.frame_idx $(expr $i / $INTV) show_truth True
done;
SID=s7
for i in `seq 0 290 2899`;
do
python train_partial.py --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i train.start_frame $(expr $i / $INTV) train.end_frame $(expr $i / $INTV + 1) train.single_frame_id `printf "frame_%06d" $(expr $i)` freeview.frame_idx $i| tee logs/train_h36m_${SID}_nvs_$i.log
python run.py --type freeview --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i  bgcolor "[0.,0.,0.]" freeview.frame_idx $(expr $i / $INTV) show_truth True
done;

SID=s8
for i in `seq 0 160 1599`;
do
python train_partial.py --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i train.start_frame $(expr $i / $INTV) train.end_frame $(expr $i / $INTV + 1) train.single_frame_id `printf "frame_%06d" $(expr $i)` freeview.frame_idx $i| tee logs/train_h36m_${SID}_nvs_$i.log
python run.py --type freeview --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i  bgcolor "[0.,0.,0.]" freeview.frame_idx $(expr $i / $INTV) show_truth True
done;

SID=s9
for i in `seq 0 190 1899`;
do
python train_partial.py --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i train.start_frame $(expr $i / $INTV) train.end_frame $(expr $i / $INTV + 1) train.single_frame_id `printf "frame_%06d" $(expr $i)` freeview.frame_idx $i| tee logs/train_h36m_${SID}_nvs_$i.log
python run.py --type freeview --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i  bgcolor "[0.,0.,0.]" freeview.frame_idx $(expr $i / $INTV) show_truth True
done;

SID=s11
for i in `seq 0 140 1399`;
do
python train_partial.py --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i train.start_frame $(expr $i / $INTV) train.end_frame $(expr $i / $INTV + 1) train.single_frame_id `printf "frame_%06d" $(expr $i)` freeview.frame_idx $i| tee logs/train_h36m_${SID}_nvs_$i.log
python run.py --type freeview --cfg configs/human_nerf/h36m/$SID/finetune_nvs.yaml experiment finetune_nvs_$i  bgcolor "[0.,0.,0.]" freeview.frame_idx $(expr $i / $INTV) show_truth True
done;