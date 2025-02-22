cd ./AirVLN
echo $PWD

# python -u ./airsim_plugin/AirVLNSimulatorServerTool_LLM.py --onscreen --gpus 0

python -u ./src/vlnce_src/eval_vlm.py \
--EVAL_GENERATE_VIDEO \
--run_type eval \
--name VLM_4o \
--batchSize 1 \
--EVAL_LLM gpt-4o \
--EVAL_DATASET val_unseen \
--EVAL_NUM -1 \
--Image_Width_RGB 640 \
--Image_Height_RGB 360 \
--Image_Width_DEPTH 640 \
--Image_Height_DEPTH 360 \
--maxAction 500


# python -u ./src/vlnce_src/train.py \
# --run_type eval \
# --policy_type seq2seq \
# --collect_type TF \
# --name AirVLN-seq2seq \
# --batchSize 1 \
# --EVAL_CKPT_PATH_DIR ../DATA/output/AirVLN-seq2seq/train/checkpoint \
# --EVAL_DATASET val_unseen \
# --EVAL_NUM -1
