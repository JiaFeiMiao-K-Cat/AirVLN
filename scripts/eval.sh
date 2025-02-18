cd ./AirVLN
echo $PWD


nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,2,3 &

python -u ./src/vlnce_src/train.py \
--run_type eval \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize 1 \
--EVAL_CKPT_PATH_DIR ../DATA/output/AirVLN-seq2seq/train/checkpoint \
--EVAL_DATASET val_unseen \
--EVAL_NUM -1


