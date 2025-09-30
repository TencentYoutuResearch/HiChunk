import argparse
import glob
import json
import os
import pickle
import time
import numpy as np
import torch
import faiss
from tqdm import tqdm


class Indexer(object):
    """Faiss 索引器，用于高效相似性搜索

    该类封装了 Faiss 索引的创建、搜索和序列化功能，支持两种索引类型：
    1. 精确搜索 (IndexFlatIP)
    2. 量化近似搜索 (IndexPQ)
    """

    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8):
        """初始化索引器

        参数:
            vector_sz (int): 向量维度大小
            n_subquantizers (int): 子量化器数量（0 表示使用精确搜索）
            n_bits (int): 每个子量化器的比特数
        """
        if n_subquantizers > 0:
            # 使用乘积量化（PQ）进行近似搜索
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            # 使用精确内积搜索
            self.index = faiss.IndexFlatIP(vector_sz)
        # 映射：Faiss 内部ID -> 外部数据库ID
        self.index_id_to_db_id = []

    def index_data(self, ids, embeddings):
        """索引一批向量数据

        参数:
            ids (list): 外部数据库ID列表
            embeddings (np.array): 向量嵌入数组
        """
        # 更新ID映射关系
        self._update_id_mapping(ids)
        # 转换向量为float32类型（Faiss要求）
        embeddings = embeddings.astype('float32')

        # 如果索引未训练（如PQ索引），则进行训练
        if not self.index.is_trained:
            self.index.train(embeddings)

        # 添加向量到索引
        self.index.add(embeddings)
        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048):
        """K近邻搜索

        参数:
            query_vectors (np.array): 查询向量数组
            top_docs (int): 返回的top K结果数
            index_batch_size (int): 批处理大小（避免内存溢出）

        返回:
            list: 包含 (数据库ID列表, 相似度分数) 元组的列表
        """
        query_vectors = query_vectors.astype('float32')
        result = []
        # 计算批处理数量
        n_batch = (len(query_vectors) - 1) // index_batch_size + 1

        # 使用进度条显示处理进度
        for k in tqdm(range(n_batch)):
            start_idx = k * index_batch_size
            end_idx = min((k + 1) * index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]

            # 执行搜索：返回相似度分数和索引ID
            scores, indexes = self.index.search(q, top_docs)

            # 将内部索引ID转换为外部数据库ID
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs]
                      for query_top_idxs in indexes]

            # 收集本批次结果
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def _update_id_mapping(self, db_ids):
        """更新内部ID到外部数据库ID的映射（内部方法）

        参数:
            db_ids (list): 新增的外部数据库ID列表
        """
        self.index_id_to_db_id.extend(db_ids)


# load passages from pickle file
def load_passages(path):
    if not os.path.exists(path):
        return
    passages = []
    if path.endswith(".pkl"):
        with open(path, 'rb') as fin:
            passages = pickle.load(fin)
    return passages


# compute question embeddings
def embed_queries(args, queries, model, tokenizer):
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                output = model.encode(batch_question)['dense_vecs']
                output = torch.tensor(output)

                embeddings.append(output.cpu())
                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


# index passages
def index_encoded_data(index, embedding_files, indexing_batch_size):
    all_ids = []
    all_embeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        all_embeddings = np.vstack((all_embeddings, embeddings)) if all_embeddings.size else embeddings
        all_ids.extend(ids)
        while all_embeddings.shape[0] > indexing_batch_size:
            all_embeddings, all_ids = add_embeddings(index, all_embeddings, all_ids, indexing_batch_size)

    while all_embeddings.shape[0] > 0:
        all_embeddings, all_ids = add_embeddings(index, all_embeddings, all_ids, indexing_batch_size)

    # print("Data indexing completed.")


# add embeddings to index
def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_add = ids[:end_idx]
    embeddings_add = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_add, embeddings_add)
    return embeddings, ids


# add passages to data
def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def load_data(data_path):
    d = None
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            d = json.load(fin)
    elif data_path.endswith(".jsonl"):
        d = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                d.append(example)
    return d


def main(args):
    start_time_indexing = time.time()
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(args.model_name_or_path,  use_fp16=not args.no_fp16, devices=['cuda:0'])
    tokenizer = model.tokenizer
    print(f"Loading model from: {args.model_name_or_path}. time: {time.time() - start_time_indexing:.1f} s.")
    example_count = 0
    # process each file in the passages_embeddings_dir
    index_global = Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
    passage_id_map_global = {}
    for fileName in os.listdir(args.passages_embeddings_dir):
        passages_embeddings_path = os.path.join(args.passages_embeddings_dir, fileName)
        index = Indexer(args.projection_size, args.n_subquantizers, args.n_bits)

        # index all passages
        input_paths = glob.glob(passages_embeddings_path)
        input_paths = sorted(input_paths)
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        index_encoded_data(index_global, input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time() - start_time_indexing:.1f} s.")

        data_path = os.path.join(args.data_dir, fileName)

        # load passages
        passages = load_passages(data_path+'.pkl')
        passage_id_map = {x["id"]: x for x in passages}
        passage_id_map_global.update(passage_id_map)

        data = load_data(data_path+'.jsonl')
        output_path = os.path.join(args.output_dir, fileName+'.jsonl')

        queries = [ex["question"] for ex in data]
        questions_embedding = embed_queries(args, queries, model, tokenizer)

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, 128)     # only record top 128 passages
        print(f"{len(passages)} psgs: Search time: {time.time() - start_time_retrieval:.1f} s.")

        add_passages(data, passage_id_map, top_ids_and_scores)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
                example_count += 1
        print(f"Saved results to {output_path}")

    print(f"Total: ", example_count)

    for fileName in os.listdir(args.passages_embeddings_dir):
        data_path = os.path.join(args.data_dir, fileName)
        data = load_data(data_path+'.jsonl')
        output_path_global = os.path.join(args.output_dir+'_global', fileName+'.jsonl')

        queries = [ex["question"] for ex in data]
        questions_embedding = embed_queries(args, queries, model, tokenizer)

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index_global.search_knn(questions_embedding, 128)     # only record top 128 passages
        print(f"{len(passage_id_map_global)} psgs: Search time: {time.time() - start_time_retrieval:.1f} s.")

        add_passages(data, passage_id_map_global, top_ids_and_scores)
        os.makedirs(os.path.dirname(output_path_global), exist_ok=True)
        with open(output_path_global, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
                example_count += 1
        print(f"Saved results to {output_path_global}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", type=str,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages_embeddings_dir", type=str, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--model_name_or_path", type=str, default='./../mcontriever',
        help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", default=True, help="lowercase text before encoding")
    parser.add_argument("--device", type=str, default='cuda', help="normalize text")

    args = parser.parse_args()
    main(args)
