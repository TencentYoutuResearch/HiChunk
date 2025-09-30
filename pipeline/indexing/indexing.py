import argparse
import concurrent.futures
import json
import os
import pickle
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer

from splitter import split_long_sentence, build_hi_tree, merge_over_chunk


def load_chunk_result(filename):
    """Load chunking results from a precomputed file
    
    Args:
        filename (str): Name of the file being processed
        
    Returns:
        dict: Precomputed chunking results or None if not applicable
    """
    # Check if output folder indicates hierarchical chunking (HC, SC, LC)
    if all([cm not in args.output_folder.split('/')[-2] for cm in ['HC', 'SC', 'LC']]):
        return None
    
    # Load chunk results from JSON file
    with open(f'{args.input_folder}/../{args.tree_chunk_model.format(dataset=filename)}', 'r') as sf:
        chunk_result = {}
        for k, v in json.loads(sf.read()).items():
            # Normalize filename keys by removing path and extension
            normalized_key = k.split('/')[-1].rsplit('.', 1)[0]
            chunk_result[normalized_key] = v
            # Also store version without spaces
            chunk_result[normalized_key.replace(' ', '')] = v
    return chunk_result


def process_fix_size_chunk(filename, data, chunk_size):
    # Standard chunking approach
    with open(f'{args.input_folder}/../docs/{filename}/{data["_id"]}.txt', 'r') as f_c:
        context = f_c.read()
    splits = split_long_sentence(context, chunk_size, filename)
    chunks = [
        {
            'id': data['_id'] + '_' + str(i),
            'text': split.strip(),
            'title': ''
        }
        for i, split in enumerate(splits)
    ]

    return chunks


def merge_max_level(splits, max_level):
    _splits = []
    for [c, l] in splits:
        if l > max_level:
            # Merge with previous chunk if level exceeds max
            _splits[-1][0] += c
        else:
            _splits.append([c, l])
    return _splits


def process_with_splits(filename, data, chunk_size, splits):
    # Apply max_level filtering for hierarchical chunking
    if 'HC' in args.output_folder.split('/')[-2] and args.max_level:
        splits = merge_max_level(splits, args.max_level)

        if args.merge_over_chunk == 1:
            # Merge over-chunking segments
            splits = merge_over_chunk(splits, tokenizer, chunk_size)

    # Build hierarchical tree structure
    root, chunks = build_hi_tree(splits, data, [], 0, len(splits), 1, chunk_size, filename)

    return chunks


def load_data(input_file):
    all_data = list(map(lambda line: json.loads(line), open(input_file, 'r', encoding='utf-8').readlines()))
    for i, d in enumerate(all_data):
        d['index'] = i  # Add original index position
    all_data = sorted(all_data, key=lambda x: x['_id'])  # Sort by document ID
    return all_data


def save_chunks(chunks, output_index_file):
    # Save chunked results to pickle file
    print(output_index_file)
    with open(output_index_file, 'wb') as of:
        pickle.dump(chunks, of)


def process_jsonl_file(input_file, output_folder, chunk_size, filename):
    """Process a single JSONL file containing QA data
    
    Args:
        input_file (str): Path to input JSONL file
        output_folder (str): Directory to save processed outputs
        chunk_size (int): Maximum size for text chunks
        filename (str): Base name of the file being processed
    """
    # Create output directory structure
    output_folder_name = str(os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0]))
    os.makedirs(output_folder_name, exist_ok=True)
    
    # Load and index all data
    doc2qas_dict = defaultdict(list)
    all_data = load_data(input_file)
    
    # Load precomputed chunk results if available
    chunk_result = load_chunk_result(filename)
    
    # Process each QA entry
    for i, data in tqdm(enumerate(all_data), desc=filename):
        # Define output file paths
        output_index_file = os.path.join(output_folder_name, data['_id'] + '.pkl')
        output_jsonl_file = os.path.join(output_folder_name, data['_id'] + '.jsonl')
        
        # Initialize document-QA mapping
        doc2qas_dict[output_jsonl_file].append({
            'index': data['index'],
            'id': data['_id'],
            'question': data.get('input', ''),
            'answers': []
        })
        
        # Skip processing if output exists and multiple QAs share same document
        if os.path.exists(output_index_file) and len(doc2qas_dict[output_jsonl_file]) > 1:
            continue
        
        # Handle hierarchical chunking models (HC, SC, LC)
        if any([cm in args.output_folder.split('/')[-2] for cm in ['HC', 'SC', 'LC']]):
            splits = list(filter(lambda s: s[0].strip() != '', chunk_result[data['_id']]['splits']))
            chunks = process_with_splits(filename, data, chunk_size, splits)
        else:
            chunks = process_fix_size_chunk(filename, data, chunk_size)
        
        save_chunks(chunks, output_index_file)
    
    # Write QA data to JSONL files
    save_jsonl(doc2qas_dict)


def save_jsonl(doc2qas_dict):
    for p, qas in doc2qas_dict.items():
        with open(p, 'w', encoding='utf-8') as f_out:
            for qa in qas:
                f_out.write(json.dumps(qa, ensure_ascii=False) + '\n')


def process_all_jsonl_files(input_folder, output_folder, chunk_size):
    """Process all JSONL files in a directory
    
    Args:
        input_folder (str): Directory containing JSONL files
        output_folder (str): Output directory for processed files
        chunk_size (int): Maximum size for text chunks
    """
    # Create output directory if needed
    os.makedirs(output_folder, exist_ok=True)
    
    # Process files with thread pool
    allowed_files = list(map(lambda af: af+'.jsonl', args.allowed_files.split(',')))
    loop = tqdm(filter(lambda fn: fn in allowed_files, os.listdir(input_folder)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for filename in loop:
            print(filename)
            input_file = os.path.join(input_folder, filename)
            loop.set_description(f"Processing {filename}")
            # process_jsonl_file(input_file, output_folder, chunk_size, filename.replace('.jsonl', ''))
            executor.submit(
                process_jsonl_file,
                input_file, output_folder, chunk_size, filename.replace('.jsonl', '')
            )


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default='../source/docqa_only', 
                        help="Input directory containing JSONL files")
    parser.add_argument("--allowed_files", type=str, default='', 
                        help="Comma-separated list of allowed files")
    parser.add_argument("--chunk_size", type=int, default=200, 
                        help="Maximum token size for text chunks")
    parser.add_argument("--output_folder", type=str, default='../datasets/C200_t/split', 
                        help="Output directory for processed files")
    parser.add_argument("--model_name_or_path", type=str, default='./contriever/mcontriever', 
                        help="Path to pretrained tokenizer model")
    parser.add_argument("--tree_chunk_model", type=str, default=None, 
                        help="Template for hierarchical chunking model files")
    parser.add_argument("--merge_over_chunk", type=int, default=0, 
                        help="Enable merging overlapping chunks (0=disable, 1=enable)")
    parser.add_argument("--max_level", type=int, default=None, 
                        help="Maximum level for hierarchical chunking")
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Start processing all files
    process_all_jsonl_files(args.input_folder, args.output_folder, args.chunk_size)
