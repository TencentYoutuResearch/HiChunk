

MODEL_PATH="BAAI/bge-large-en-v1.5"
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
  echo "python ./pipeline/chunking/SemanticChunk/semantic_chunk.py \
    --books_dir ${dp} \
    --model_path ${MODEL_PATH}"

  python ./pipeline/chunking/SemanticChunk/semantic_chunk.py \
    --books_dir ${dp} \
    --model_path ${MODEL_PATH}
done
