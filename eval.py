import argparse
import json
import os

import numpy as np
from nltk import sent_tokenize
from tqdm import tqdm
from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    cal_evidence_recall
)


args = argparse.ArgumentParser()
args.add_argument("--data", type=str, default="C200_tk4096")
args.add_argument("--model", type=str, default="chatglm2-6b")
args.add_argument("--filter_flag", type=str, default=None)
args = args.parse_args()
print(args)


dataset2metric = {
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "HiCBench": rouge_score,
    "qasper_all": qa_f1_score,
    "OHRBench_sub": rouge_score,
    "GutenQA": rouge_score,
}


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    all_scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            if isinstance(prediction, str):
                prediction = [prediction]
            score = max(score, np.mean(
                list(map(
                    lambda p: dataset2metric[dataset](p.replace('Answer: ', ''), ground_truth, all_classes=all_classes),
                    prediction
                ))
            ))
        total_score += score
        all_scores.append(score)
    return round(100 * total_score / (len(predictions)+1e-20), 2), all_scores


if __name__ == '__main__':
    scores = dict()
    ret_scores = dict()
    merge_count = dict()
    all_files = os.listdir(f"pred/{args.model}/{args.data}")
    for filename in [
        "narrativeqa.jsonl", "qasper.jsonl", "multifieldqa_en.jsonl", "multifieldqa_zh.jsonl",
        "hotpotqa.jsonl", "2wikimqa.jsonl", "musique.jsonl", "dureader.jsonl",
        "HiCBench.jsonl",
        "qasper_all.jsonl",
        "OHRBench_sub.jsonl",
        "GutenQA.jsonl",
    ]:
        if filename not in all_files:
            continue
        predictions, answers, questions, merge_counts = [], [], [], []
        dataset = filename.split('.')[0]
        all_data = []
        with (open(f"pred/{args.model}/{args.data}/{filename}", "r", encoding='utf-8') as f):
            for line in f.readlines():
                try:
                    data = json.loads(line)
                except Exception as e:
                    continue
                if len(data) == 0 or (
                        args.filter_flag is not None and data["all_classes"] is not None and args.filter_flag not in
                        data['all_classes']):
                    # if len(data) == 0:
                    continue
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                questions.append(data["input"])
                all_data.append(data)
                merge_counts.append(data["merge_count"])
        score, all_scores = scorer(dataset, predictions, answers, all_classes)
        for s, d in tqdm(zip(all_scores, all_data)):
            d['dataset_metric'] = s
            if 'evidences' in d and ('OHRBench' in filename or 'GutenQA' in filename or 'HiCBench' in filename):
                d['evidences_retrieval_metric'] = sum(
                    [cal_evidence_recall(e.lstrip('# ').replace('. ', '.'), d['context']) * len(e) for e in
                     d['evidences']]) / (sum([len(e) for e in d['evidences']]) + 1e-20)
            elif 'qasper_all' in filename:
                d['evidences_retrieval_metric'] = sum(
                    [cal_evidence_recall(es.lstrip('# '), d['context']) * len(es) for e in d['evidences'] for es in
                     sent_tokenize(e)]) / (sum([len(es) for e in d['evidences'] for es in sent_tokenize(e)]) + 1e-20)
            else:
                d['answers_retrieval_metric'] = sum(
                    [cal_evidence_recall(e, d['context']) * len(e) for e in d['answers']]) / (
                                                        sum([len(e) for e in d['answers']]) + 1e-20)

        with open(f"pred/{args.model}/{args.data}/eval_{filename}", "w", encoding='utf-8') as f:
            for d in all_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        scores[dataset] = score
        merge_count[dataset] = sum(merge_counts)
        ret_scores[dataset] = round(
            100 * sum([d['evidences_retrieval_metric'] for d in all_data if 'evidences_retrieval_metric' in d]) / (
                    len(all_data) + 1e-20), 2)
        if ret_scores[dataset] == 0:
            ret_scores[dataset] = round(
                100 * sum([d['answers_retrieval_metric'] for d in all_data if 'answers_retrieval_metric' in d]) / (
                        len(all_data) + 1e-20), 2)

        if ret_scores[dataset] == 0.:
            del ret_scores[dataset]
        print("len:", len(all_data))
        print("scores:", json.dumps(scores, indent=0))
        print("merge_count:", json.dumps(merge_count, indent=0))
        print('avg:', f"{sum(scores.values()) / len(scores):.2f}", 'count:', len(all_data), 'dataset_num:', len(scores))
        print("ret_scores:", json.dumps(ret_scores, indent=0))
