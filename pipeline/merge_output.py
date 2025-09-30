import argparse
import json
import os

from tqdm import tqdm

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description="合并检索结果工具")
parser.add_argument('--input_folder', type=str, default='mcontriever_output',
                    help='包含检索结果JSONL文件的输入目录路径')
parser.add_argument('--input_dataFile', type=str, default='inputData.jsonl',
                    help='原始输入数据JSONL文件名')
parser.add_argument('--output_dataFile', type=str, default='DATA.jsonl',
                    help='合并后的输出数据JSONL文件名')
args = parser.parse_args()


def merge_text(jsonl_file):
    """处理单个JSONL文件，提取检索结果ID
    
    参数:
        jsonl_file (str): JSONL文件路径
        
    返回:
        list: 包含每个问题的检索ID列表的数据字典
    """
    # 读取并解析JSONL文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data_list = list(map(lambda l: json.loads(l), f.readlines()))

    output_data_list = []
    for d in data_list:
        context_list = d['ctxs']  # 获取检索到的上下文列表
        
        # 提取所有检索结果的ID
        retrieved_ids = list(map(lambda item: item['id'], context_list))
        
        # 构建输出数据结构
        output_data = {
            'index': d['index'],          # 问题索引
            'input': d['question'],       # 原始问题文本
            'id': d['id'],                # 问题唯一ID
            'retrieved_ids': retrieved_ids  # 检索结果ID列表
        }
        output_data_list.append(output_data)
    return output_data_list


def process_all_jsonl_files(args):
    """处理目录下所有JSONL文件并合并结果
    
    参数:
        args (Namespace): 命令行参数对象
    """
    input_folder = args.input_folder
    output_data_list = []
    
    # 遍历输入目录中的所有文件
    loop = tqdm(filter(lambda x: x.endswith('.jsonl'), os.listdir(input_folder)))
    for filename in loop:
        jsonl_file = os.path.join(input_folder, filename)
        # 处理当前文件并添加到结果列表
        _output_data_list = merge_text(jsonl_file)
        output_data_list += _output_data_list
    
    # 按索引排序所有结果
    output_data_list = sorted(output_data_list, key=lambda x: x['index'])
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_dataFile), exist_ok=True)
    
    # 读取原始输入数据
    with open(args.input_dataFile, 'r', encoding='utf-8') as in_data:
        in_data_list = list(map(lambda l: json.loads(l), in_data.readlines()))
        for i in range(len(in_data_list)):
            in_data_list[i]['index'] = i
    
    # 将检索结果合并到原始数据
    with open(args.output_dataFile, 'w', encoding='utf-8') as out_data:
        for orig_data, retrieved_data in zip(in_data_list, output_data_list):
            # 验证数据一致性
            assert orig_data['index'] == retrieved_data['index'], "问题索引不匹配"
            assert orig_data['input'] == retrieved_data['input'], "问题文本不匹配"
            assert orig_data['_id'] == retrieved_data['id'], "问题ID不匹配"
            
            # 添加检索结果ID到原始数据
            orig_data['retrieved_ids'] = retrieved_data['retrieved_ids']

        # 验证所有检索ID都存在
        assert all([_['_id'] in pid for _ in in_data_list for pid in _['retrieved_ids']])
        # 写入合并后的数据
        for data_l in in_data_list:
            out_data.write(json.dumps(data_l, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    process_all_jsonl_files(args)

