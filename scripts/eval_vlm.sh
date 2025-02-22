cd ./AirVLN
echo $PWD

nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool_LLM.py --gpus 2,3 &

python -u ./src/vlnce_src/eval_vlm.py \
--EVAL_GENERATE_VIDEO \
--run_type eval \
--name VLM \
--batchSize 1 \
--EVAL_LLM llama3.2-vision:11b-instruct-q8_0 \
--EVAL_DATASET val_unseen \
--EVAL_NUM -1 \
--Image_Width_RGB 1280 \
--Image_Height_RGB 720 \
--Image_Width_DEPTH 1280 \
--Image_Height_DEPTH 720 \
--maxAction 10
