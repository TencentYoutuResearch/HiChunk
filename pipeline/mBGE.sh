
chunk_type=$1   # HC for hi_chunking. C for normal chunking
chunk_size=$2
merge_over_chunk=0
max_level=10
embedding_model_path="BAAI/bge-m3"
root_path=$(pwd)
work_dir="${root_path}/dataset" # dir for storing data

source_dir="${work_dir}/qas" # source dataset dir

if [ "$chunk_type" == "HC" ]; then
  tree_chunk_model="HiChunk/{dataset}/end2end_predict_limit100_window16384_rt2.json"
  chunk_dir="BgeM3/${chunk_type}${chunk_size}_L${max_level}"
  merge_over_chunk=1
elif [ "$chunk_type" == "SC" ]; then
  tree_chunk_model="SemanticChunker/{dataset}/bge-large-en-v1.5.json"
  chunk_dir="BgeM3/${chunk_type}${chunk_size}"
elif [ "$chunk_type" == "LC" ]; then
  tree_chunk_model="LumberChunker_wot_dt/{dataset}/Deepseek.json"
  chunk_dir="BgeM3/LC${chunk_size}"
elif [ "$chunk_type" == "LCQW" ]; then
  tree_chunk_model="LumberChunker_wot_dt/{dataset}/Qwen3-4B.json"
  chunk_dir="BgeM3/LC${chunk_size}_QW"
else
  tree_chunk_model="None"
  chunk_dir="BgeM3/${chunk_type}${chunk_size}"
fi

mkdir -p "${work_dir}/${chunk_dir}/"

index_dir="${work_dir}/${chunk_dir}/index"
embed_dir="${work_dir}/${chunk_dir}/embed"
retrieved_dir="${work_dir}/${chunk_dir}/retrieval"

folder_names=(
  "multifieldqa_en" "qasper" "2wikimqa" "dureader" "hotpotqa" "narrativeqa" "musique" "multifieldqa_zh"
  "HiCBench"
  "qasper_all"
  "OHRBench_sub"
  "GutenQA"
  )
str=$(IFS=, ; echo "${folder_names[*]}")
echo "$str"

echo "python ./pipeline/indexing/indexing.py \
        --model_name_or_path ${embedding_model_path} \
        --chunk_size ${chunk_size} \
        --output_folder  ${index_dir}\
        --input_folder ${source_dir} \
        --tree_chunk_model ${tree_chunk_model} \
        --allowed_files ${str} \
        --merge_over_chunk ${merge_over_chunk} \
        --max_level ${max_level}"

python ./pipeline/indexing/indexing.py \
        --model_name_or_path ${embedding_model_path} \
        --chunk_size ${chunk_size} \
        --output_folder  ${index_dir}\
        --input_folder ${source_dir} \
        --tree_chunk_model ${tree_chunk_model} \
        --allowed_files ${str} \
        --merge_over_chunk ${merge_over_chunk} \
        --max_level ${max_level}

for folder in "${folder_names[@]}"; do
    # generate embeddings
    echo "python ./pipeline/embedding/generate_passage_embeddings.py \
            --model_name_or_path ${embedding_model_path} \
            --output_dir ${embed_dir}/${folder}  \
            --psgs_dir  "${index_dir}/${folder}/" \
            --shard_id 0 --num_shards 1 \
            --lowercase"
    python ./pipeline/embedding/generate_passage_embeddings.py \
            --model_name_or_path ${embedding_model_path} \
            --output_dir ${embed_dir}/${folder}  \
            --psgs_dir  "${index_dir}/${folder}/" \
            --shard_id 0 --num_shards 1 \
            --lowercase

    echo "./pipeline/retrieval/passage_retrieval.py \
            --model_name_or_path ${embedding_model_path} \
            --passages_embeddings_dir ${embed_dir}/${folder} \
            --data_dir ${index_dir}/${folder} \
            --output_dir ${retrieved_dir}/${folder} \
            --lowercase \
            --projection_size 1024 \
            --device cuda"
    python ./pipeline/retrieval/passage_retrieval.py \
            --model_name_or_path ${embedding_model_path} \
            --passages_embeddings_dir ${embed_dir}/${folder} \
            --data_dir ${index_dir}/${folder} \
            --output_dir ${retrieved_dir}/${folder} \
            --lowercase \
            --projection_size 1024 \
            --device "cuda"

    echo "python ./pipeline/merge_output.py \
            --input_folder  ${retrieved_dir}/${folder} \
            --input_dataFile ${source_dir}/${folder}.jsonl \
            --output_dataFile ${work_dir}/${chunk_dir}/data/${folder}.jsonl"
    python ./pipeline/merge_output.py \
            --input_folder  "${retrieved_dir}/${folder}" \
            --input_dataFile "${source_dir}/${folder}.jsonl" \
            --output_dataFile "${work_dir}/${chunk_dir}/data/${folder}.jsonl"
done

