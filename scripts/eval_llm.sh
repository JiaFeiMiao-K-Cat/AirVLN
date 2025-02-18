cd ./AirVLN
echo $PWD

nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool_LLM.py --gpus 3 &

python -u ./src/vlnce_src/eval_llm.py \
--run_type eval \
--name LLM \
--batchSize 1 \
--EVAL_LLM qwen2.5:32b-instruct-fp16 \
--EVAL_DATASET val_seen \
--EVAL_NUM -1 \
--Image_Width_RGB 640 \
--Image_Height_RGB 360 \
--Image_Width_DEPTH 640 \
--Image_Height_DEPTH 360 \
--maxAction 500