# Novel view synthesis on ZJU MoCap
SID=313
for i in 99 198 297 396 495 594 693 792 891 990;
do
python train.py --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_0 train.single_frame_id `printf "frame_%06d" $(expr $i)`;
python run.py --type freeview --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_$i  bgcolor [0.,0.,0.] freeview.frame_idx $(expr $i) bbox_offset 0.05
done;
SID=315
for i in 100 300 500 700 900 1100 1300 1500 1700 1900;
do
python train.py --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_0 train.single_frame_id `printf "frame_%06d" $(expr $i)`;
python run.py --type freeview --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_$i  bgcolor [0.,0.,0.] freeview.frame_idx $(expr $i) bbox_offset 0.05
done;
SID=377
for i in 23 80 137 194 251 308 365 422 479 536;
do
python train.py --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_0 train.single_frame_id `printf "frame_%06d" $(expr $i)`;
python run.py --type freeview --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_$i  bgcolor [0.,0.,0.] freeview.frame_idx $(expr $i) bbox_offset 0.05
done;
SID=386
for i in 0 54 108 162 216 270 324 378 432 486;
do
python train.py --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_0 train.single_frame_id `printf "frame_%06d" $(expr $i)`;
python run.py --type freeview --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_$i  bgcolor [0.,0.,0.] freeview.frame_idx $(expr $i) bbox_offset 0.05
done;
SID=387
for i in 1 55 109 163 217 271 325 379 433 487;
do
python train.py --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_0 train.single_frame_id `printf "frame_%06d" $(expr $i)`;
python run.py --type freeview --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_$i  bgcolor [0.,0.,0.] freeview.frame_idx $(expr $i) bbox_offset 0.05
done;
SID=390
for i in 100 200 300 400 500 600 700 800 900 1000;
do
python train.py --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_0 train.single_frame_id `printf "frame_%06d" $(expr $i)`;
python run.py --type freeview --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_$i  bgcolor [0.,0.,0.] freeview.frame_idx $(expr $i) bbox_offset 0.05
done;
for i in 28 83 138 193 248 303 358 413 468 523;
do
python train_partial.py --cfg configs/human_nerf/zju_mocap/392/final_v1_nvs.yaml experiment final_v1_nvs_$i train.start_frame $i train.end_frame $i train.single_frame_id `printf "frame_%06d" $(expr $i + 0)` freeview.frame_idx $i;
python run.py --type freeview --cfg configs/human_nerf/zju_mocap/392/final_v1_nvs.yaml experiment final_v1_nvs_$i train.start_frame $i train.end_frame $i train.single_frame_id `printf "frame_%06d" $(expr $i + 0)` freeview.frame_idx $(expr $i + 0) bgcolor "[0.,0.,0.]" show_truth True;
done;
SID=393
for i in 100 199 298 397 496 595 694 793 892 991;
do
python train.py --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_0 train.single_frame_id `printf "frame_%06d" $(expr $i)`;
python run.py --type freeview --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_$i  bgcolor [0.,0.,0.] freeview.frame_idx $(expr $i) bbox_offset 0.05
done;
SID=394
for i in 0 48 96 144 192 240 288 336 384 432;
do
python train.py --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_0 train.single_frame_id `printf "frame_%06d" $(expr $i)`;
python run.py --type freeview --cfg configs/human_nerf/zju_mocap/${SID}/final_v1_nvs.yaml experiment final_v1_nvs_$i  bgcolor [0.,0.,0.] freeview.frame_idx $(expr $i) bbox_offset 0.05
done;