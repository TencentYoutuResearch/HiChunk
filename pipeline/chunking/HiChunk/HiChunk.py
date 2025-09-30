import argparse
import asyncio
import itertools
import json
import os
import re
import time

import aiohttp
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer


def replace_jinhao(line, replacement=None):
    if replacement is not None and re.match(r'^( *#*)*', line)[0].strip() != '':
        # 全部替换为 replacement
        return re.sub(r'^( *#*)*', replacement, line, count=1)
    else:
        return line


def count_jinhao(line):
    return re.match(r'^( *#*)*', line)[0].count('#')


def is_english(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def sentence_split_en(line):
    res = filter(lambda l: l.strip() != '', sent_tokenize(line))
    res = list(map(lambda l: l.strip(), res))
    idx = 0
    while idx < len(res) - 1:
        if len(res[idx]) < 10:  # 句子少于10个字符，向后合并。
            res[idx + 1] = res[idx] + ' ' + res[idx + 1]
            res.pop(idx)
        else:
            idx += 1

    return res


def sentence_split_zh(line):
    res = []
    pre_idx = 0
    for i in range(1, len(line)-1):
        if line[i] == '。':  # 按句号切分
            continue
        if line[i - 1] in '0123456789':   # ocr error
            continue
        if line[i + 1] in '0123456789':  # ocr error
            continue
        if len(line[pre_idx: i + 1].strip()) <= 5:
            continue

        res.append(line[pre_idx: i + 1].strip())
        pre_idx = i + 1

    if pre_idx < len(line):
        res.append(line[pre_idx:])

    return res


def sentence_split(line):
    if is_english(line):
        return sentence_split_en(line)
    else:
        return sentence_split_zh(line)


def sentence_truncation(line, head_limit=15, tail_limit=15):
    total_limit = head_limit+tail_limit
    if is_english(line):
        len_factor = 10
    else:
        len_factor = 1

    if 0 < total_limit * len_factor < len(line):
        _head_limit = head_limit * len_factor
        _tail_limit = len(line) - tail_limit * len_factor
        line = line[:_head_limit] + line[_tail_limit:]

    return line


def text2sentence(lines, start_idx=-1, replacement=None, head_limit=15, tail_limit=15):
    """
    :param lines:
    :param start_idx: -1 means no index of each line.
    :param replacement: in [None, '# ', '']. replace the jinhao prefix of one line. None means no replacement.
    :param head_limit:
    :param tail_limit:
    :return:
    """
    res = []
    for idx, line in enumerate(lines):
        res.extend(sentence_split(line))

    for idx, temp in enumerate(res):
        _temp = replace_jinhao(temp, '# ')
        _temp = sentence_truncation(_temp, head_limit, tail_limit)
        _temp = replace_jinhao(_temp, f"{'#'*count_jinhao(temp)} ")
        _temp = replace_jinhao(_temp, replacement)
        if start_idx != -1:
            _temp = index_format(start_idx+idx, _temp)
        res[idx] = _temp+'\n'
    return res


PROMPT = ('You are an assistant good at reading and formatting documents, and you are also skilled at distinguishing '
          'the semantic and logical relationships of sentences between document context. The following is a text that '
          'has already been divided into sentences. Each line is formatted as: "{line number} @ {sentence content}". '
          'You need to segment this text based on semantics and format. There are multiple levels of granularity for '
          'segmentation, the higher level number means the finer granularity of the segmentation. Please ensure that '
          'each Level One segment is semantically complete after segmentation. A Level One segment may contain '
          'multiple Level Two segments, and so on. Please incrementally output the starting line numbers of each level '
          'of segments, and determine the level of the segment, as well as whether the content of the sentence at the '
          'starting line number can be used as the title of the segment. Finally, output a list format result, '
          'where each element is in the format of: "{line number}, {segment level}, {be a title?}".'
          '\n\n>>> Input text:\n')


def index_format(idx, line):
    return f'{idx} @ {line}'


def build_input_instruction(
        prompt, global_start_idx, sentences, window_size, residual_lines=None, tokenizer=None
):
    """
    Build input instruction for once inference
    :param prompt: prompt
    :param global_start_idx: global start index of input sentences
    :param sentences: global sentences
    :param window_size:
    :param residual_lines:
    :param tokenizer:
    :return:
    """
    q = prompt
    # concat residual lines if exists
    residual_index = 0
    while residual_lines is not None and residual_index < len(residual_lines):
        line_text = index_format(residual_index, residual_lines[residual_index])
        temp_text = q+line_text
        q = temp_text
        residual_index += 1
    assert check_length(q, window_size, tokenizer), 'residual lines exceeds window size'

    local_start_idx = 0
    cur_token_num = count_length(q, tokenizer)
    end = False
    # concat sentences until reach window_size
    while global_start_idx < len(sentences):
        line_text = index_format(local_start_idx+residual_index, sentences[global_start_idx])
        temp_text = q+line_text
        line_token_num = count_length(line_text, tokenizer)
        if cur_token_num+line_token_num > window_size:
            break
        cur_token_num += line_token_num
        q = temp_text
        local_start_idx += 1
        global_start_idx += 1
    if global_start_idx == len(sentences):
        end = True

    return q, end, local_start_idx


def points2clip(points, start_idx, end_idx):
    """
    :param points: [a, b, c, d]
    :param start_idx: x
    :param end_idx: y
    assert: x <= a < b < c < d < y
    return [[x, a], [a, b], [b, c], [c, d], [d, y]]
    """
    clips = []
    pre_p = start_idx
    for p in points:
        if p == start_idx or p >= end_idx:
            continue
        clips.append([pre_p, p])
        pre_p = p

    clips.append([pre_p, end_idx])
    return clips


# parse answer string to list of chunking points
def parse_answer_chunking_point(answer_string, max_level):
    local_chunk_points = {level_dict_en[i]: [] for i in range(max_level)}
    for line in answer_string.split('\n'):
        [point, level, _] = line.split(', ')
        if level in local_chunk_points:
            local_chunk_points[level].append(int(point))

    res = list(local_chunk_points.values())
    for idx, _ in enumerate(res):
        if len(_) == 0:
            continue
        keep_idx = list(filter(lambda i: _[i] > _[i-1], range(1, len(_))))
        res[idx] = [_[0]] + list(map(lambda i: _[i], keep_idx))
    return res


# build chunks from chunking points
def build_splits(origin_lines, global_chunk_points):
    total_points = sorted(
        [[__, i + 1] for i, _ in enumerate(global_chunk_points) for __ in _],
        key=lambda p: p[0]
    )
    splits = []
    pre_level, pre_point = 1, 0
    for i, [p, level] in enumerate(total_points):
        if p == 0:
            continue
        splits.append([''.join(origin_lines[pre_point: p]), pre_level])
        pre_level = level
        pre_point = p
    splits.append([''.join(origin_lines[pre_point:]), pre_level])
    return splits


level_dict_en = {
    0: 'Level One',
    1: 'Level Two',
    2: 'Level Three',
    3: 'Level Four',
    4: 'Level Five',
    5: 'Level Six',
    6: 'Level Seven',
    7: 'Level Eight',
    8: 'Level Nine',
    9: 'Level Ten',
}


def check_answer_point(first_level_points, start_idx, end_idx):
    print('parsed_answer:', first_level_points, start_idx, end_idx)
    if len(first_level_points) > 0 and first_level_points[0] < start_idx:
        return False
    for idx in range(1, len(first_level_points)):
        p = first_level_points[idx]
        if p <= first_level_points[idx-1] or p > end_idx:
            return False
    return True


def build_residual_lines(lines, global_chunk_points, start_idx, window_size, recurrent_type):
    if recurrent_type == 1:
        return []
    assert recurrent_type == 2, f'Not implemented for recurrent_type: {recurrent_type}'

    last_first_point = 0
    if len(global_chunk_points[0]) > 0:
        last_first_point = global_chunk_points[0][-1]
    current_second_points = filter(lambda p: p >= last_first_point, global_chunk_points[1])
    temp_second_clips = points2clip(current_second_points, last_first_point, start_idx)
    # 每个一级片段中，最多保留5个二级片段，前2后3，每个二级片段最多20行。
    pre_seg_num, post_seg_num, line_num = 2, 3, 20
    while True:
        residual_second_clips = temp_second_clips
        if len(temp_second_clips) > (pre_seg_num + post_seg_num):
            residual_second_clips = (
                    temp_second_clips[:pre_seg_num] + temp_second_clips[len(temp_second_clips)-post_seg_num:]
            )
        residual_lines = []
        for rsc in residual_second_clips:
            # 每个二级片段最多保留20行
            pre_sent_idx, post_sent_idx = rsc[0], min(rsc[1], rsc[0]+line_num)
            residual_lines.extend(lines[pre_sent_idx: post_sent_idx])
        if len('\n'.join(residual_lines)) < window_size/2:
            print(residual_lines)
            return residual_lines

        # 超出推理窗口一半，则需要减少残余输入。前减1，后减1，行数减5。
        pre_seg_num, post_seg_num, line_num = pre_seg_num-1, post_seg_num-1, line_num-5
        # 最小设置的情况下仍然超出窗口一半，则不添加残余输入。
        if pre_seg_num * post_seg_num * line_num <= 0:
            return []


def check_length(text, max_length, tokenizer):
    return len(tokenizer.encode(text)) <= max_length


def count_length(text, tokenizer):
    return len(tokenizer.encode(text))


class InfModel:
    def __init__(self, model_path, max_new_token, window_size, model_deploy='vllm'):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.tokenizer.pad_token_id = 0
        if model_deploy == 'vllm':
            engine_args = AsyncEngineArgs(model=model_path, tokenizer=model_path, tokenizer_mode="auto",
                                          trust_remote_code=True, max_num_batched_tokens=window_size+1000,
                                          tensor_parallel_size=1, dtype="float16", quantization=None, revision=None,
                                          tokenizer_revision=None, seed=0, gpu_memory_utilization=0.9, swap_space=4,
                                          disable_log_stats=True, max_num_seqs=10, enforce_eager=True)
            self.model = AsyncLLMEngine.from_engine_args(engine_args)
            self.sampling_params = SamplingParams(top_p=0.8, top_k=1, max_tokens=max_new_token,
                                                  stop_token_ids=[self.tokenizer.eos_token_id],
                                                  skip_special_tokens=False)
            self.inf_func = self.inf_vllm
        elif re.fullmatch(r'\d+\.\d+\.\d+\.\d+:\d+', model_deploy):     # model_deploy = ip:port
            self.inf_func = self.inf_api
            self.model = None
        else:
            raise NotImplementedError(f'Not implemented for model_deploy: {model_deploy}')

        self.max_new_token = max_new_token
        self.window_size = window_size
        self.model_deploy = model_deploy

    def apply_chat_template(self, question, known_answer_str=None):
        prefix_ans = known_answer_str
        text = f"""<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n{prefix_ans}"""
        return text

    # 通过VLLM引擎调用推理模型
    async def inf_vllm(self, question, request_id='', known_answer_str=''):
        text = self.apply_chat_template(question, known_answer_str)
        results_generator = self.model.generate(text, self.sampling_params, request_id)
        final_output = None
        async for output in results_generator:
            final_output = output
        pred = final_output.outputs[0].text
        return pred, text

    # 通过API调用推理模型
    async def inf_api(self, question, request_id, known_answer_str=''):
        text = self.apply_chat_template(question, known_answer_str)
        post_data = {
            "model": self.model_path,
            "prompt": text,
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": self.max_new_token,
            "skip_special_tokens": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://{self.model_deploy}/v1/completions", json=post_data) as response:
                pred = await response.json()
                pred = pred['choices'][0]['text']
        return pred, text


def init_chunk_points(multi_level):
    global_chunk_points = [[] for i in range(multi_level)]
    return global_chunk_points


def mapping_idx(local_chunk_points, start_idx, residual_sent_num):
    for idx, points in enumerate(local_chunk_points):
        local_chunk_points[idx] = filter(lambda p: p >= residual_sent_num, points)
        # p-residual_sent_num+start_idx 将局部推理的句子序号转化为全局文档的句子序号
        local_chunk_points[idx] = [p - residual_sent_num + start_idx for p in local_chunk_points[idx]]
    return local_chunk_points


class InferenceEngine:
    def __init__(self, model_path, window_size, model_deploy, max_new_token=4096):
        self.window_size = window_size
        self.model = InfModel(model_path, max_new_token, window_size, model_deploy)
        self.request_ids = ("".join(x) for x in itertools.product("0123456789", repeat=9))

    async def iterative_inf(self, prompt, document, limit=-1, multi_level=10, recurrent_type=1):
        lines = map(lambda l: l.strip(), document.split('\n'))
        lines = list(filter(lambda l: len(l) != 0 and re.match(r'^( *#*)*', l)[0] != l, lines))
        origin_lines = text2sentence(lines, -1, None, -1, 0)
        lines = text2sentence(lines, -1, '', limit, 0)
        error_count, start_idx = 0, 0
        raw_qa, residual_lines = [], []
        global_chunk_points = init_chunk_points(multi_level)
        while start_idx < len(lines):
            residual_sent_num = len(residual_lines)
            question, is_end, question_sent_num = build_input_instruction(
                prompt, start_idx, lines, self.window_size, residual_lines, self.model.tokenizer
            )
            print('len question:', len(question), len(self.model.tokenizer.encode(question)))
            start_time = time.time()
            answer, revised_question = await self.model.inf_func(
                question, next(self.request_ids)
            )
            end_time = time.time()
            print('answer:', answer)
            tmp = {
                'question': revised_question, 'answer': answer, 'start_idx': start_idx,
                'end_idx': start_idx+question_sent_num, 'residual_sent_num': residual_sent_num,
                'time': end_time-start_time, 'question_token_num': len(self.model.tokenizer.encode(question)),
                'answer_token_num': len(self.model.tokenizer.encode(answer))
            }
            # 解析输出结果，将局部句子序号转化为全局的句子序号
            try:
                local_chunk_points = parse_answer_chunking_point(answer, multi_level)
                if not check_answer_point(local_chunk_points[0], 0, question_sent_num+residual_sent_num-1):
                    print('###########check error##############')
                    tmp['status'] = 'check error'
                    local_chunk_points = init_chunk_points(multi_level)
                    local_chunk_points[0].append(start_idx)
                    error_count += 1
                else:
                    tmp['status'] = 'check ok'
                    print('#############check ok################')
                    local_chunk_points = mapping_idx(local_chunk_points, start_idx, residual_sent_num)

            except:
                print('##########parsed error################')
                tmp['status'] = 'parse error'
                local_chunk_points = init_chunk_points(multi_level)
                local_chunk_points[0].append(start_idx)
                error_count += 1
            raw_qa.append(tmp)

            print(local_chunk_points)
            if is_end:  # 全文档推理结束
                start_idx += question_sent_num
                global_chunk_points = merge_result(local_chunk_points, global_chunk_points, start_idx)
                break
            if len(local_chunk_points[0]) > 1:  # 多个一级片段，丢弃掉本次结果的最后一个一级片段
                # 从最后一个一级片段开始构建下次迭代的输入
                start_idx = local_chunk_points[0][-1]
                global_chunk_points = merge_result(local_chunk_points, global_chunk_points, start_idx)
                residual_lines = []
            else:   # 局部推理结果中只有一个一级片段
                # 从上次迭代输入的最后一行开始构建下次迭代的输入，并带上上次迭代的残余行
                start_idx += question_sent_num
                global_chunk_points = merge_result(local_chunk_points, global_chunk_points, start_idx)
                residual_lines = build_residual_lines(
                    lines, global_chunk_points, start_idx, self.window_size, recurrent_type
                )

        print('multi_level_seg_points:', global_chunk_points)
        result = {
            'multi_level_seg_points': global_chunk_points,
            'raw_qa': raw_qa,
            'error_count': error_count,
            'splits': build_splits(origin_lines, global_chunk_points)
        }

        return result


def merge_result(local_chunk_points, global_chunk_points, max_idx):
    for idx, level in enumerate(global_chunk_points):
        global_chunk_points[idx].extend(filter(lambda p: p < max_idx, local_chunk_points[idx]))
    return global_chunk_points


async def main(args):
    inf_engine = InferenceEngine(args.model_path, args.window_size, args.model_deploy)
    root = args.input_root.rstrip('/')
    all_result = {}
    # 遍历所有文件
    for file in os.listdir(root):
        file = os.path.join(root, file)
        print(file)
        document = open(file.strip()).read()
        start_time = time.time()
        result = await inf_engine.iterative_inf(PROMPT, document, args.limit, recurrent_type=args.recurrent_type)
        end_time = time.time()
        result['time_cost'] = end_time-start_time
        result['doc_length'] = len(document)
        result['filepath'] = file.strip()
        all_result[file.strip()] = result
    pretty_path = os.path.join(
        f"{root.replace('/docs/', '/HiChunk/')}",
        f"end2end_predict_limit{args.limit}_window{args.window_size}_rt{args.recurrent_type}.json"
    )
    os.makedirs(os.path.dirname(pretty_path), exist_ok=True)
    with open(pretty_path, 'w') as f:
        json.dump(all_result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument(
        "--model_path", type=str, default="",
    )
    parser.add_argument(
        "--window_size", type=int, default=16384, required=False
    )
    parser.add_argument(
        "--input_root", type=str, default='', required=False
    )
    parser.add_argument(
        "--limit", type=int, default=100, required=False
    )
    parser.add_argument(
        "--recurrent_type", type=int, default=1
    )
    parser.add_argument(
        "--model_deploy", type=str, default='vllm', help='vllm or ip:port'
    )
    args = parser.parse_args()
    print(args)

    asyncio.run(main(args))

