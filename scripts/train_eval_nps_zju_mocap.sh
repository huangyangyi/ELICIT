# Novel pose synthesis on ZJU MoCap
for SID in s1 s5 s6 s7 s8 s9 s11;
do
python train.py  --cfg configs/human_nerf/h36m/$SID/finetune.yaml 
python run.py --type movement --cfg configs/human_nerf/h36m/$SID/finetune.yaml bgcolor "[0.,0.,0.]" show_truth True 
done