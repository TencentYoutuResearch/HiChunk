"""
Retrieval Algorithms for Hierarchical Chunk Structures

This module implements retrieval algorithms designed for tree-structured chunk hierarchies.
The algorithms efficiently retrieve relevant text passages while respecting token limits
and automatically merging related chunks when beneficial.
"""
import os
from typing import List, Dict, Tuple, Callable

import numpy as np
from transformers import PreTrainedTokenizer

from metrics import cal_evidence_recall


def tree_chunk_retrieval(chunks: Dict[str, List[Dict]], json_obj: Dict, tokenizer: PreTrainedTokenizer,
                         token_num: int, chunk_size: int = None) -> Tuple[str, int]:
    """
    Basic tree-based chunk retrieval algorithm
    
    Retrieves chunks in order until the token limit is reached, skipping duplicates.
    
    Args:
        chunks: List of chunk dictionaries containing 'text'
        json_obj: Contains 'retrieved_ids' list of passage IDs
        tokenizer: Tokenizer for encoding text
        token_num: Maximum token limit for the context
        chunk_size: Optional chunk size parameter (unused in this implementation)
        
    Returns:
        Tuple: (context string, 0 for compatibility)
    """
    context_chunks, idx = [], 0
    # Retrieve chunks until token limit is reached or all passages processed
    while len(tokenizer.encode("\n\n".join(context_chunks))) < token_num and idx < len(json_obj['retrieved_ids']):
        # Skip duplicate passages
        if json_obj['retrieved_ids'][idx] in json_obj['retrieved_ids'][:idx]:
            idx += 1
            continue
        doc_id, p_idx = json_obj['retrieved_ids'][idx].rsplit('_', 1)
        p_idx = int(p_idx)
        context_chunks.append(chunks[doc_id][p_idx]['text'])
        idx += 1
    # Truncate to token limit and decode
    context_tokens = tokenizer.encode("\n\n".join(context_chunks))[:token_num]
    return tokenizer.decode(context_tokens, skip_special_tokens=True), 0


def check_merge_condition(chunks: List[Dict], retrieved_passages: List[Dict], parent: Dict,
                          tokenizer: PreTrainedTokenizer, token_num: int, 
                          chunk_size: int, context: str) -> Tuple[bool, List[Dict]]:
    """
    Determines if child passages under a parent should be merged
    
    Args:
        chunks: Complete list of text chunks
        retrieved_passages: Currently retrieved passages
        parent: Parent passage being considered for merge
        tokenizer: Tokenizer for text processing
        token_num: Token budget limit
        chunk_size: Size of chunks (unused)
        context: Current context string
        
    Returns:
        Tuple: (True if should merge, list of same-parent passages)
    """
    # Find all passages with the same parent
    same_parent_p = [rp for rp in retrieved_passages if rp['parent'] == parent]
    # Calculate character counts for children and parent
    child_char_num = sum(
        [
            len(chunks[c_idx]['text'])
            for rp in same_parent_p
            for c_idx in range(rp['left_index_idx'], rp['right_index_idx'])
        ]
    )
    parent_char_num = sum(
        [
            len(chunks[c_idx]['text'])
            for c_idx in range(parent['left_index_idx'], parent['right_index_idx'])
        ]
    )
    
    # Calculate merge threshold based on current context usage
    threshold = (len(tokenizer.encode(context)) / token_num) * parent_char_num / 3 + parent_char_num / 3
    
    # Condition 1: Children cover significant portion of parent
    condition_1 = child_char_num >= threshold
    # Condition 2: Multiple children exist
    condition_2 = len(same_parent_p) >= 2
    
    # Calculate token usage for already retrieved children
    had_retrieved_token_num = len(
        tokenizer.encode('\n'.join([
            chunks[idx]['text']
            for rp in retrieved_passages
            for idx in range(rp['left_index_idx'], rp['right_index_idx'])
            if parent['left_index_idx'] <= idx < parent['right_index_idx']
        ]))
    )
    
    # Calculate total tokens in parent
    parent_token_num = len(
        tokenizer.encode('\n'.join([
            chunks[idx]['text']
            for idx in range(parent['left_index_idx'], parent['right_index_idx'])
        ]))
    )
    
    # Condition 3: Remaining token budget can accommodate parent
    remain_tokens_budget = token_num - len(tokenizer.encode(context))
    condition_3 = (parent_token_num - had_retrieved_token_num) <= remain_tokens_budget

    return condition_1 and condition_2 and condition_3, same_parent_p


def remove_repeat_passages(retrieved_passages: List[Dict], parent: Dict) -> List[Dict]:
    """
    Removes passages that are covered by a parent passage
    
    Args:
        retrieved_passages: Current list of retrieved passages
        parent: Parent passage that may cover child passages
        
    Returns:
        List: Filtered passages without duplicates covered by parent
    """
    remain_passages = []
    for rp in retrieved_passages:
        # Skip if passage is already covered by parent
        if rp == parent:
            remain_passages.append(rp)
        elif rp['left_index_idx'] >= parent['right_index_idx']:
            remain_passages.append(rp)
        elif rp['right_index_idx'] <= parent['left_index_idx']:
            remain_passages.append(rp)
    return remain_passages


def tree_chunk_retrieval_auto_merge(chunks: Dict[str, List[Dict]], json_obj: Dict, tokenizer: PreTrainedTokenizer,
                                    token_num: int, chunk_size: int) -> Tuple[str, int]:
    """
    Advanced retrieval with automatic chunk merging
    
    Retrieves passages while automatically merging related chunks when beneficial,
    respecting token limits and hierarchical relationships.
    
    Args:
        chunks: List of chunk dictionaries
        json_obj: Contains 'retrieved_ids' list
        tokenizer: Tokenizer for text processing
        token_num: Token budget limit
        chunk_size: Size of chunks
        
    Returns:
        Tuple: (context string, count of auto-merge operations performed)
    """
    def get_doc_id(retrieved_passage: Dict):
        """Retrieves document IDs from passages"""
        temp = retrieved_passage
        while temp['children']:
            temp = temp['children'][0]
        return temp['id'].rsplit('_', 1)[0]

    def build_context_from_passages(_retrieved_passages: List[Dict]) -> str:
        """Builds context string from retrieved passages"""
        c_ids = [__ for _ in _retrieved_passages for __ in range(_['left_index_idx'], _['right_index_idx'])]
        doc_ids = [get_doc_id(_) for _ in _retrieved_passages]
        # Ensure no duplicate chunks
        assert len(c_ids) == len(set(c_ids)), c_ids
        _context = '\n\n'.join([
            '\n'.join([chunks[doc_id][__]['text'] for __ in range(_['left_index_idx'], _['right_index_idx'])])
            for doc_id, _ in zip(doc_ids, _retrieved_passages)
        ])
        return _context

    auto_merge_count = 0
    retrieved_passages = []
    context = build_context_from_passages(retrieved_passages)
    
    # Process each retrieved ID
    for retrieval_id in json_obj['retrieved_ids']:
        doc_id, p_idx = retrieval_id.rsplit('_', 1)
        p_idx = int(p_idx)
        passage = chunks[doc_id][p_idx]
        
        # Skip if passage is already covered
        if any([
            rp['left_index_idx'] <= passage['left_index_idx'] < passage['right_index_idx'] <= rp['right_index_idx']
            for i, rp in enumerate(retrieved_passages)
        ]):
            continue
            
        # If passage is the only child of its parent, use parent instead
        if len(passage['parent']['children']) == 1:
            passage = passage['parent']
            
        # Track chunk count before adding new passage
        o_chunk_count = len([__ for _ in retrieved_passages for __ in range(_['left_index_idx'], _['right_index_idx'])])
        retrieved_passages.append(passage)
        
        # Attempt to merge with ancestors
        parent = passage['parent']
        while parent is not None:
            # Check if we should merge with this parent
            merge_condition, same_parent_psg = check_merge_condition(
                chunks[doc_id], retrieved_passages, parent, tokenizer, token_num, chunk_size, context
            )
            
            if merge_condition:
                # Replace first child with parent
                retrieved_passages[retrieved_passages.index(same_parent_psg[0])] = parent
                # Remove chunks covered by parent
                retrieved_passages = remove_repeat_passages(retrieved_passages, parent)

                # Move up hierarchy
                passage = passage['parent']
                parent = parent['parent']
                context = build_context_from_passages(retrieved_passages)
                
                # Stop if token limit reached
                if len(tokenizer.encode(context)) >= token_num:
                    break
            else:
                break
                
        # Track auto-merge operations
        n_chunk_count = len([__ for _ in retrieved_passages for __ in range(_['left_index_idx'], _['right_index_idx'])])
        auto_merge_count += (n_chunk_count - o_chunk_count - 1)
        context = build_context_from_passages(retrieved_passages)
        
        # Stop if token limit reached
        if len(tokenizer.encode(context)) >= token_num:
            break

    # Final context truncation
    context = tokenizer.decode(tokenizer.encode(context)[:token_num], skip_special_tokens=True)
    return context, auto_merge_count


# Algorithm registry
auto_merge_algors: List[Callable] = [
    tree_chunk_retrieval,
    tree_chunk_retrieval_auto_merge,
]


if __name__ == '__main__':
    import json
    import pickle
    from tqdm import tqdm
    from transformers import AutoTokenizer

    preds = [json.loads(l) for l in open(
        f'/home/tione/notebook/vincentwslu/projects/HiCBench/pred/llama3.1-8b/BgeM3_new/HC200_L10_tk4096_AM1/HiCBench.jsonl').readlines()]
    preds_dict = {p['input'] + p['answers'][0]: p for p in preds}

    tokenizer = AutoTokenizer.from_pretrained('/home/tione/notebook/vincentwslu/models/Llama-3.1-8B-Instruct')
    # tokenizer = AutoTokenizer.from_pretrained('/home/tione/notebook/boke/llm/models/qwen-ckpts/Qwen3-32B')
    data = 'HC200_L10'
    context_same = []
    evidence_recall = []
    objs = [json.loads(l) for l in open(
        f'/home/tione/notebook/vincentwslu/projects/HiCBench/dataset/BgeM3_new/{data}/data/HiCBench.jsonl'
    ).readlines()]
    passages_dict = {}
    for fn in os.listdir(f"./dataset/BgeM3_new/{data}/index/HiCBench"):
        if not fn.endswith(".pkl"):
            continue
        passages = pickle.load(open(f"./dataset/BgeM3_new/{data}/index/HiCBench/{fn}", 'rb'))
        passages_dict[fn.replace('.pkl', '')] = passages

    for i, o in tqdm(enumerate(objs)):
        key = o['input'] + o['answers'][0]
        c, _ = tree_chunk_retrieval_auto_merge(passages_dict, o, tokenizer, 4096, 200)
        p = preds_dict[key]
        context_same.append(p['context'] == c)
        evidence_recall.append(
            sum([cal_evidence_recall(e.lstrip('# ').replace('. ', '.'), c) * len(e) for e in o['evidences']])
            /
            (sum([len(e) for e in o['evidences']]) + 1e-20)
        )
    print(len(context_same), sum(context_same) / len(context_same), np.mean(evidence_recall))
