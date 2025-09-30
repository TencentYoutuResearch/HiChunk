import json
import re

import numpy as np


# 计算一行中开头的'#'数量（用于识别Markdown标题级别）
def count_jinhao(line):
    """计算一行开头连续'#'的数量"""
    return re.match(r'^( *#*)*', line)[0].count('#')


# 定义要处理的数据集列表
dataset = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader", "HiCBench",
    "qasper_all", "gov_report_5w", "OHRBench_sub", "GutenQA"
]

# 定义不同分块方法的文件路径模板
chunk_method = [
    f'../../dataset/SemanticChunker/{{dataset}}/bge-large-en-v1.5.json',  # 基于语义的分块方法
    f'../../dataset/LumberChunker_wot_dt/{{dataset}}/Deepseek.json',    # LumberChunker方法
    f'../../dataset/HiChunk/{{dataset}}/end2end_predict_limit100_window16384_rt2.json'  # HiChunk方法
]

# 具有分块点标签的数据集（可以进行分块准确率评估）
datasets_with_chunk_points_label = ["HiCBench", "qasper_all", "gov_report_5w"]

# 遍历所有数据集
for ds in dataset:
    # 为当前数据集生成所有分块方法的结果文件路径
    chunk_result = [cm.format(dataset=ds) for cm in chunk_method]

    # 处理每个分块结果文件
    for cr in chunk_result:
        print('\n\n')
        print('*****************************************')
        print(f'Dataset: {ds}')
        print(f'Chunk Result: {cr}')
        print('*****************************************')
        try:
            # 读取分块结果JSON文件
            with open(cr, 'r') as f:
                data = json.load(f)

            # 计算平均分块时间和平均分块数量
            print(
                "avg_time_per_document:", np.mean([data[fn]['time_cost'] for fn in data]),  # 平均分块时间
            )
            print(
                "avg_chunks_per_document:", np.mean([len(data[fn]['splits']) for fn in data])  # 平均分块数量
            )

            # 如果当前数据集没有分块点标签，跳过准确率计算
            if ds not in datasets_with_chunk_points_label:
                continue

            # 初始化统计变量
            pred_num = [0] * 10  # 预测的分块数量（按标题级别）
            gt_num = [0] * 10    # 实际的分块数量（按标题级别）
            pt_num = [0] * 10    # 正确预测的分块数量（按标题级别）
            pt_no_level = 0      # 正确预测的分块数量（不考虑级别）

            # 遍历每个文件的分块结果
            for fn in data:
                # 读取原始文档内容
                ori_doc = open(f"../../dataset/docs/{ds}/{fn.split('/')[-1]}").read()

                # 统计实际的分块点（Markdown标题）
                for l in ori_doc.split("\n"):
                    if l.startswith('#'):
                        level = count_jinhao(l.strip()) - 1  # 计算标题级别
                        gt_num[level] += 1  # 增加对应级别的计数

                # 获取预测的分块结果
                splits = data[fn]['splits']

                # 遍历每个预测的分块
                for [s, level] in splits:
                    # 更新预测的分块计数
                    pred_num[level - 1] += 1

                    # 检查分块是否正确（标题格式和级别匹配）
                    if s.startswith('#') and count_jinhao(s.strip()) == level:
                        pt_num[level - 1] += 1

                    # 统计不考虑级别的正确分块
                    if s.startswith('#'):
                        pt_no_level += 1

            # 计算各级别的召回率、精确率和F1值
            r = [pt_num[i] / (gt_num[i] + 1e-20) for i in range(10)]  # 召回率
            p = [pt_num[i] / (pred_num[i] + 1e-20) for i in range(10)]  # 精确率
            f1 = [2 * p[i] * r[i] / (p[i] + r[i] + 1e-20) for i in range(10)]  # F1值

            print('Recall by level: ', r)
            print('Precision by level: ', p)
            print('F1 by level: ', f1)

            # 计算不考虑级别的整体指标
            r_wo_l = pt_no_level / (sum(gt_num) + 1e-20)  # 整体召回率
            p_wo_l = pt_no_level / (sum(pred_num) + 1e-20)  # 整体精确率
            f1_wo_l = 2 * p_wo_l * r_wo_l / (p_wo_l + r_wo_l + 1e-20)  # 整体F1值

            print('Overall Recall (without level): ', r_wo_l)
            print('Overall Precision (without level): ', p_wo_l)
            print('Overall F1 (without level): ', f1_wo_l)

        except Exception as e:
            # 处理异常（如文件不存在等）
            pass
