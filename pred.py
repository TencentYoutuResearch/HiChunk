import argparse
import json
import multiprocessing
import os
import pickle
import re

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from metrics import cal_evidence_recall
from retrieval_algo import auto_merge_algors


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# call vllm api
def call_vllm(prompt, model, max_gen):
    if args.n > 1:
        # n > 1 means we are doing sampling
        post_data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_gen,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "n": args.n
        }
    else:
        # n = 1 means we are doing greedy search
        post_data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_gen,
            "temperature": 0.0,
            "top_k": 1,
            "n": 1
        }
    response = requests.post(f"http://0.0.0.0:{args.port}/v1/completions", json=post_data)
    return list(map(lambda r: r['text'], response.json()['choices']))


def load_passages(path):
    passages = []
    if not os.path.exists(path):
        return passages
    with open(path, 'rb') as fin:
        passages = pickle.load(fin)

    return passages


# This is the customized building prompt for chat models, here is an example for ChatGLM2
def build_chat(tokenizer, prompt, model_name):
    if 'qwen' in model_name:
        _prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    elif 'llama' in model_name:
        _prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        return prompt
    return _prompt


def process_example(case):
    pred_dict, json_obj = case['pred_dict'], case['json_obj']
    index = json_obj['index']
    print(index)
    # if pred exists, skip this data item
    if 'pred' in pred_dict and not args.redo:
        pred_dict['evidences'] = json_obj.get('evidences', [])
        pred_dict['facts'] = json_obj.get('facts', [])
        pred_dict['retrieved_ids'] = json_obj['retrieved_ids']
        return pred_dict
    else:
        # call vllm to get prediction
        pred = call_vllm(json_obj['prompt'], json_obj['model'], json_obj['max_gen'])

    # print the first prediction
    if index == 0:
        print("prompt: ", json_obj['prompt'])
        print("pred: ", pred)
        print("answers: ", json_obj["answers"])

    return {
        "index": index, "pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
        "context_length": json_obj['context_length'], "context": json_obj['context'], "input": json_obj['input'],
        "evidences": json_obj.get('evidences', []), "id": json_obj["_id"], 'merge_count': json_obj['merge_count'],
        'facts': json_obj.get('facts', []), "retrieved_ids": json_obj['retrieved_ids']
    }


def get_passages_dict(dataset):
    passages_dict = {}
    for fn in os.listdir(f"./dataset/{args.data}/index/{dataset}"):
        if not fn.endswith(".pkl"):
            continue
        temp_passages_file = f"./dataset/{args.data}/index/{dataset}/{fn}"
        passages = load_passages(temp_passages_file)
        passages_dict[fn.replace('.pkl', '')] = passages

    return passages_dict


def build_context(preds, model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name, args):
    pre_passages_file, passages = '', None  # record the last passages file, avoid reload same file
    passages_dict = get_passages_dict(dataset)
    # build context for each data item
    for p, d in tqdm(zip(preds, data)):
        merge_count = 0
        if args.token_num == 0:
            # token num = 0 means no context
            d['context'] = ""
        elif args.token_num == -1:
            # token num = -1 means use the full document as context
            d['context'] = open(f"./dataset/docs/{dataset}/{d['_id']}.txt", 'r').read()
        else:
            temp_passages_file = f"./dataset/{args.data}/index/{dataset}/{d['_id']}.pkl"
            # if passages file is changed, reuse it.
            if temp_passages_file != pre_passages_file:
                passages = load_passages(temp_passages_file)
                pre_passages_file = temp_passages_file
            if 'HC' in args.data:
                chunk_size = int(re.search(r'HC\d+', args.data).group(0)[2:])
                # args.auto_merge is the index of auto_merge_algors
                d['context'], merge_count = auto_merge_algors[args.auto_merge](
                    passages_dict, d, tokenizer, args.token_num, chunk_size
                )
            else:
                d['context'], merge_count = auto_merge_algors[0](
                    passages_dict, d, tokenizer, args.token_num
                )
        # build prompt
        prompt = prompt_format.format(**d)
        prompt = build_chat(tokenizer, prompt, model_name)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if prompt is too long, truncate it
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = (tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                      tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))

        input_data = tokenizer(prompt, truncation=False, return_tensors="pt")
        context_length = input_data.input_ids.shape[-1]
        d['prompt'] = prompt
        d['context_length'] = context_length
        d['merge_count'] = merge_count
        d['model_name'] = model_name
        d['model'] = model
        d['dataset'] = dataset
        d['max_gen'] = max_gen


def load_history(data, save_path, dataset):
    preds = [{}] * len(data)
    # load history pred from file
    if os.path.exists(f"{save_path}/{dataset}.jsonl"):
        with open(f"{save_path}/{dataset}.jsonl", "r", encoding="utf-8") as f:
            for index, lines in enumerate(f.readlines()):
                try:
                    preds[index] = json.loads(lines)
                except:
                    preds[index] = {}
        # preds = sorted(preds, key=lambda _p: _p['index'])
    return preds


def save_prediction(preds, save_path, dataset):
    with open(f"{save_path}/{dataset}.jsonl", "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name, save_path, args):
    preds = load_history(data, save_path, dataset)
    build_context(preds, model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name, args)
    # Calculate evidence recall metric
    evidences_retrieval_metric = []
    for d in data:
        evidences_retrieval_metric.append(sum(
            [
                cal_evidence_recall(e.lstrip('# ').replace('. ', '.'), d['context']) * len(e) for e in d['evidences']
            ]
        ) / (sum([len(e) for e in d['evidences']]) + 1e-20))
    print(f"{dataset} evidences retrieval metric: {np.mean(evidences_retrieval_metric)}")

    assert len(preds) == len(data)
    for i, d in enumerate(data):
        assert d['index'] == i, (d['index'], i)
    # doing prediction
    with multiprocessing.Pool(processes=30) as pool:
        preds = [p for p in pool.imap_unordered(
            process_example, [{'pred_dict': p, 'json_obj': d} for p, d in zip(preds, data)]
        )]
    preds = sorted(preds, key=lambda _p: _p['index'])   # sort preds by index

    save_prediction(preds, save_path, dataset)
    return preds


def load_model_and_tokenizer(model2path, model_name):
    return model_name, AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="chatglm2-6b")
    parser.add_argument("--token_num", type=int, default=4096)
    parser.add_argument("--data", type=str, default="C200")
    parser.add_argument("--auto_merge", type=int, default=0)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model
    print(args)
    # Retrieval is fit for these datasets
    datasets = (
        ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "dureader"] +
        ["HiCBench"] +
        ["qasper_all"] +
        ["OHRBench_sub"] +
        ["GutenQA"]
    )
    # load configs
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # define your model
    model, tokenizer = load_model_and_tokenizer(model2path, model_name)
    max_length = model2maxlen[model_name]
    # define the path for saving the prediction
    save_path = (f"pred/{args.model}/{args.data}_tk{args.token_num}{f'_AM{args.auto_merge}' if args.auto_merge else ''}"
                 f"{f'_run{args.run}' if args.run > 0 else ''}")
    os.makedirs(save_path, exist_ok=True)
    # predict on each dataset
    for dataset in datasets:
        print(dataset)
        data = map(lambda d: json.loads(d), open(f"./dataset/{args.data}/data/{dataset}.jsonl", "r"))
        data = sorted(data, key=lambda _d: _d['index'])
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        get_pred(
            model2path[model_name], tokenizer, data, max_length, max_gen,
            prompt_format, dataset, model_name, save_path, args
        )
