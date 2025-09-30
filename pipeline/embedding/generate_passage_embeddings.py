import argparse
import os
import pickle

import torch


def load_passages(path):
    """加载段落数据文件
    
    参数:
        path (str): 段落数据文件路径
        
    返回:
        list: 加载的段落数据列表
    """
    if not os.path.exists(path):
        return
    passages = []
    # 支持加载pickle格式的段落数据
    if path.endswith(".pkl"):
        with open(path, 'rb') as fin:
            passages = pickle.load(fin)
    return passages


@torch.no_grad()
def embed_passages(args, passages, model):
    """生成段落嵌入向量
    
    参数:
        args (Namespace): 命令行参数对象
        passages (list): 段落数据列表
        model (BGEM3FlagModel): 嵌入模型实例
        
    返回:
        tuple: (段落ID列表, 嵌入向量数组)
    """
    total = 0
    all_ids, all_embeddings = [], []
    batch_ids, batch_text = [], []
    for k, p in enumerate(passages):
        batch_ids.append(p["id"])
        # 处理段落文本（可选是否包含标题）
        text = p["text"]
        if not args.no_title:
            text = (p.get("title", "") + " " + p["text"]).lstrip()
        # 可选转换为小写
        if args.lowercase:
            text = text.lower()
        batch_text.append(text)

        # 当批次达到指定大小或处理最后一段时进行编码
        if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:
            print(f'encoding passages[{k-len(batch_text)}: {k+1}]')
            # 使用模型生成嵌入向量
            output = model.encode(batch_text)['dense_vecs']
            embeddings = torch.tensor(output).cpu()
            total += len(batch_ids)
            all_ids.extend(batch_ids)
            all_embeddings.append(embeddings)

            # 重置批次
            batch_text = []
            batch_ids = []

    # 合并所有嵌入向量
    if len(all_embeddings) > 0:
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_ids, all_embeddings


def main(args):
    """主处理函数"""
    from FlagEmbedding import BGEM3FlagModel
    # 初始化嵌入模型（使用FP16加速）
    model = BGEM3FlagModel(args.model_name_or_path, use_fp16=not args.no_fp16, devices=['cuda:0'])

    print(f"Model loaded from {args.model_name_or_path}.", flush=True)
    # 获取所有段落数据文件
    psgs_list = filter(lambda x: x.endswith('.pkl'), os.listdir(args.psgs_dir))
    psgs_list = list(map(lambda x: os.path.join(args.psgs_dir, x), psgs_list))
    print('passage count: ', len(psgs_list))
    
    # 处理每个段落文件
    for psg in psgs_list:
        passages = load_passages(psg)

        # 分片处理（分布式支持）
        shard_size = len(passages) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size
        if args.shard_id == args.num_shards - 1:
            end_idx = len(passages)

        passages = passages[start_idx:end_idx]

        # 生成嵌入向量
        ids, embeddings = embed_passages(args, passages, model)

        # 保存嵌入结果
        doc_name = os.path.splitext(os.path.basename(psg))[0]
        save_file = os.path.join(args.output_dir, doc_name)
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving {len(ids)} passage embeddings to {save_file}.")
        with open(save_file, mode="wb") as f:
            pickle.dump((ids, embeddings), f)

        print(f"Total passages processed {len(ids)}. Written to {save_file}.")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="生成段落嵌入向量工具")
    parser.add_argument("--psgs_dir", type=str, required=True, help="段落数据文件目录")
    parser.add_argument("--output_dir", type=str, default="wikipedia_embeddings", help="嵌入向量输出目录")
    parser.add_argument("--prefix", type=str, default="passages", help="嵌入文件前缀")
    parser.add_argument("--shard_id", type=int, default=0, help="当前处理分片ID")
    parser.add_argument("--num_shards", type=int, default=1, help="总分片数量")
    parser.add_argument("--per_gpu_batch_size", type=int, default=512, help="GPU批次大小")
    parser.add_argument("--passage_maxlength", type=int, default=512, help="段落最大长度")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="模型路径")
    parser.add_argument("--no_fp16", action="store_true", help="禁用FP16加速")
    parser.add_argument("--no_title", action="store_true", help="忽略标题信息")
    parser.add_argument("--lowercase", action="store_true", help="转换为小写")

    args = parser.parse_args()
    main(args)
