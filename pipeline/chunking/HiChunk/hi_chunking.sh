
export RT=2
export WINDOW_SIZE=16384

dataset_paths=(
  "./dataset/docs/2wikimqa"
  "./dataset/docs/dureader"
  "./dataset/docs/hotpotqa"
  "./dataset/docs/multifieldqa_en"
  "./dataset/docs/multifieldqa_zh"
  "./dataset/docs/musique"
  "./dataset/docs/narrativeqa"
  "./dataset/docs/qasper"
  "./dataset/docs/HiCBench"
  "./dataset/docs/qasper_all"
  "./dataset/docs/gov_report_5w"
  "./dataset/docs/OHRBench_sub"
  "./dataset/docs/GutenQA"
)

for dp in "${dataset_paths[@]}";
do
  echo "python ./pipeline/chunking/HiChunk/HiChunk.py \
    --input_root ${dp} \
    --model_path ${MODEL_PATH} \
    --window_size ${WINDOW_SIZE} \
    --limit 100 \
    --recurrent_type ${RT} \
    --model_deploy vllm"

  python ./pipeline/chunking/HiChunk/HiChunk.py \
    --input_root ${dp} \
    --model_path ${MODEL_PATH} \
    --window_size ${WINDOW_SIZE} \
    --limit 100 \
    --recurrent_type ${RT} \
    --model_deploy vllm
done
