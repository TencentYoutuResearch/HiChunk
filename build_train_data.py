import time
import os
import json
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import pandas as pd
from tqdm import tqdm
import random

from transformers import AutoTokenizer

from pipeline.chunking.HiChunk.HiChunk import (text2sentence, index_format, check_length, count_jinhao,
                                               replace_jinhao, PROMPT, level_dict_en)


def split_by_pattern(lines, p='# '):
    segments = []
    last_title_idx = 0
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.startswith(p):
            segments.append([lines[last_title_idx:idx], True])
            last_title_idx = idx

    segments.append([lines[last_title_idx:], True])
    return segments


def apply_answer_struct(answer):
    """
    :param answer: [[[0, True], [4, False]], [],..., []]
    """
    # [[idx, level, True/False],..., [idx, level, True/False]]
    points = [[__[0], level_dict_en[idx], __[1]] for idx, _ in enumerate(answer) for __ in _]
    points = sorted(points, key=lambda p: p[0])
    new_answer = []
    for p in points:
        new_answer.append(f"{p[0]}, {p[1]}, {p[2]}")
    return '\n'.join(new_answer)


def build_residual_lines(annotation, f_idx, s_idx):
    pre_first_seg = deepcopy(annotation[max(0, f_idx - random.randint(0, 1)): f_idx + 1])  # 最多1-2个一级片段
    for pfs in pre_first_seg[:-1]:
        pfs['second'] = [pfs['second'][0]] + random.sample(pfs['second'][1:], k=random.randint(
            min(1, len(pfs['second']) - 1),
            min(4, len(pfs['second']) - 1)))  # 最多2-5个二级片段，保留第一个二级片段。
    pre_first_seg[-1]['second'] = [pre_first_seg[-1]['second'][0]] + random.sample(
        pre_first_seg[-1]['second'][1:s_idx + 1], k=random.randint(min(1, s_idx), min(4, s_idx))
    )
    pre_lines = [
        ___
        for _ in pre_first_seg
        for __ in _['second']
        for ___ in __[0][:random.randint(1, 20)]
    ]
    pre_lines = [replace_jinhao(s, "") for s in pre_lines]

    return pre_lines


class TrainData:
    def __init__(self):
        self.prompt = PROMPT

    def reset(self, residual_lines):
        residual_text = ''.join([index_format(i, _) for i, _ in enumerate(residual_lines)])
        return self.prompt+residual_text, len(residual_lines), [[] for _ in range(MAX_LEVEL)]

    def make_train_data(
            self, results, type, max_length=4096, omit_p=0., shuffle_p=0.
    ):
        sample_len = 0
        train_data = []
        point_level_num = [0 for _ in range(MAX_LEVEL)]
        temp_result = deepcopy(results)
        for fp, annotation in temp_result.items():
            q, start_line_idx, answer = self.reset(residual_lines=[])
            if random.random() < shuffle_p:
                random.shuffle(annotation)

            # 遍历每一个一级片段
            for f_idx, anno in enumerate(annotation):
                first_text, _ = anno['first']
                if check_length('\n'.join(first_text), 50, tokenizer=tokenizer):
                    continue

                if random.random() < omit_p:
                    second_count = random.randint(min(len(first_text), 5), min(len(first_text), 15))
                    anno['second'] = anno['second'][:second_count]
                second_texts = anno['second']

                # 遍历每一个二级片段
                for s_idx, (second_lines, is_title) in enumerate(second_texts):
                    if random.random() < omit_p:
                        # 随机对二级片段进行截断
                        sentence_count = random.randint(min(len(second_lines), 15), min(len(second_lines), 25))
                        second_lines_origin = second_lines[:sentence_count]
                    else:
                        second_lines_origin = second_lines
                    _second_lines = [replace_jinhao(s, "") for i, s in enumerate(second_lines_origin)]
                    _second_lines = [index_format(start_line_idx + i, s) for i, s in enumerate(_second_lines)]

                    line_idx = 0
                    sample_len = len(tokenizer.encode(q))
                    while sample_len < max_length and line_idx < len(_second_lines):
                        line_len = len(tokenizer.encode(_second_lines[line_idx]))
                        sample_len += line_len
                        q += _second_lines[line_idx]

                        level = count_jinhao(second_lines_origin[line_idx])
                        if 0 < level <= MAX_LEVEL:
                            answer[level-1].append([start_line_idx + line_idx, True])
                            point_level_num[level-1] += 1

                        line_idx += 1

                    start_line_idx += line_idx

                    if sample_len >= max_length:
                        answer_str = apply_answer_struct(answer)
                        train_data.append(
                            {
                                'instruction': q,
                                'output': answer_str,
                                'input': '',
                                'fp': fp,
                                'len': len(q),
                                'type': type,
                                "sample_len": sample_len
                            }
                        )

                        pre_lines = build_residual_lines(annotation, f_idx, s_idx)
                        q, start_line_idx, answer = self.reset(pre_lines)
                        break

            answer_str = apply_answer_struct(answer)
            train_data.append(
                {
                    'instruction': q,
                    'output': answer_str,
                    'input': '',
                    'fp': fp,
                    'len': len(q),
                    'type': type,
                    "sample_len": sample_len
                }
            )

        return train_data


def process(task):
    root, fn = task
    res = []
    fp = os.path.join(root, fn)
    with open(fp) as f:
        doc = f.read()
    lines = doc.split('\n')
    lines = text2sentence(lines, -1, replacement=None, head_limit=head_limit, tail_limit=tail_limit)
    lines = list(filter(lambda l: len(l) != 0, lines))
    first_level_segments = split_by_pattern(lines, '# ')
    for seg in first_level_segments:
        second_level_segments = split_by_pattern(seg[0], '## ')
        res.append({'first': seg, 'second': second_level_segments})
    return fn, res


if __name__ == "__main__":
    MAX_LEVEL = 10  # 训练的最大层级
    window = 8192
    head_limit = 100
    tail_limit = 0
    DATE = time.strftime('%m%d', time.localtime())
    splits = ['dev']
    os.makedirs('corpus/combined', exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', trust_remote_code=True)
    for split in splits:
        seeds = [1, 2, 3] if split == 'train' else [1]
        split_data = []
        root_paths = [
            f'corpus/qasper/{split}_file',
            f'corpus/wiki_727_5w/{split}_file',
            f'corpus/gov-report_5w/{split}_file',
        ]

        for root_path in root_paths:
            print(root_path)
            results = {}
            tasks = [(root, f) for root, dirs, files in os.walk(root_path) for f in files]
            with ProcessPoolExecutor(max_workers=64) as executor:
                futures = executor.map(process, tasks)
                for future in tqdm(futures, desc="Processing files"):
                    fn, res = future
                    results[fn] = res

            # 处理成json line的格式
            train_files = os.listdir(root_path)
            data_maker = TrainData()
            tasks = [{k: results[k]} for k in train_files]

            with ProcessPoolExecutor(max_workers=64) as executor:
                for seed in seeds:
                    random.seed(seed)
                    if seed == 1:
                        futures = [
                            executor.submit(data_maker.make_train_data, t, 'original', window, 0.0, 0.0)
                            for t in tasks
                        ]
                        for future in tqdm(futures):
                            split_data += future.result()

                    # data augmentation
                    futures = [
                        executor.submit(data_maker.make_train_data, t, f'internal_shuffle_s{seed}', window, 0.0, 1.0)
                        for t in tasks
                    ]
                    for future in tqdm(futures):
                        split_data += future.result()

                    futures = [
                        executor.submit(data_maker.make_train_data, t, f'augmented_s{seed}', window, 1.0, 1.0)
                        for t in tasks
                    ]
                    for future in tqdm(futures):
                        split_data += future.result()

        split_data_keys = split_data[0].keys()
        pd_split_data = {k: [] for k in split_data_keys}
        for d in split_data:
            for k in split_data_keys:
                pd_split_data[k].append(d[k])
        df = pd.DataFrame(pd_split_data)
        df.to_parquet(f"corpus/combined/for_{split}_data_{DATE}.parquet")

        dataset_info = json.load(open(f'corpus/combined/dataset_info.json', 'r'))
        dataset_info.update(
            {
                f'for_{split}_data_{DATE}': {
                    'file_name': f"for_{split}_data_{DATE}.parquet"
                }
            }
        )
        json.dump(dataset_info, open(f'corpus/combined/dataset_info.json', 'w'), indent=2, ensure_ascii=False)
