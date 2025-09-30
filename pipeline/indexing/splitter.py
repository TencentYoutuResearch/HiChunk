"""
Text Splitting Utilities

This module provides functions for splitting text into sentences and chunks,
with special handling for Chinese text segmentation. It also includes utilities
for building hierarchical text structures.
"""


import copy
import re

from nltk import sent_tokenize
from transformers import AutoTokenizer


def split_sentence_zh(sentence: str) -> list[str]:
    """
    Split Chinese text into sentences using custom regex rules.
    
    Args:
        sentence: Input Chinese text string
        
    Returns:
        List of segmented sentences
    """
    # Regex pattern for Chinese sentence boundaries
    regex = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')
    s = sentence
    slist = []
    for i in regex.split(s):
        if regex.match(i) and slist:
            # Append to last sentence if matched punctuation
            slist[-1] += i
        elif i:
            # Start new sentence
            slist.append(i)
    return slist


def split_long_sentence(sentence: str, chunk_size: int, filename: str, tokenizer=None) -> list[str]:
    """
    Split long text into chunks of specified size, handling Chinese/English differently.
    
    Args:
        sentence: Input text to split
        chunk_size: Target token count per chunk
        filename: Source filename (determines language handling)
        tokenizer: Optional tokenizer for accurate length calculation
        
    Returns:
        List of text chunks
    """
    chunks = []
    # Choose segmentation method based on filename
    sentences = split_sentence_zh(sentence) if filename in ['multifieldqa_zh', 'dureader'] else sent_tokenize(sentence)
    current_chunk = ""
    for s in sentences:
        if current_chunk and get_word_len(current_chunk, tokenizer) + get_word_len(s, tokenizer) <= chunk_size:
            current_chunk += (' ' if s == '' else s)
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_len = get_word_len(current_chunk, tokenizer)
                if current_len > chunk_size * 1.5:
                    print(f"\n{filename}-{len(chunks) - 1} Chunk size: {current_len}")

            current_chunk = s

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def get_word_list(s1: str) -> list[str]:
    """
    Tokenize text into words, handling Chinese characters individually.
    
    Args:
        s1: Input text string
        
    Returns:
        List of words/tokens
    """
    # Regex for non-word characters
    regEx = re.compile('[\W]')
    # Regex for Chinese characters
    res = re.compile(r"([\u4e00-\u9fa5])")

    p1 = regEx.split(s1.lower())
    str1_list = []
    
    # Process each segment
    for s in p1:
        if res.split(s) is None:
            str1_list.append(s)
        else:
            # Split Chinese characters
            str1_list.extend(res.split(s))

    # Filter empty tokens
    return list(filter(lambda w: len(w.strip()) > 0, str1_list))


def get_word_len(s1: str, tokenizer=None) -> int:
    """
    Calculate word/token length of text.
    
    Args:
        s1: Input text
        tokenizer: Optional tokenizer for accurate count
        
    Returns:
        Length in words/tokens
    """
    return get_token_len(s1, tokenizer) if tokenizer else len(get_word_list(s1))


def get_token_len(s1: str, tokenizer) -> int:
    """
    Get token count using specified tokenizer.
    
    Args:
        s1: Input text
        tokenizer: Tokenizer instance
        
    Returns:
        Number of tokens
    """
    return len(tokenizer.tokenize(s1))


def merge_over_chunk(splits: list, tokenizer, chunk_size: int) -> list:
    """
    Merge adjacent text splits that together are under chunk size limit.
    
    Args:
        splits: List of [text, level] pairs
        tokenizer: Tokenizer for length calculation
        chunk_size: Maximum allowed token count
        
    Returns:
        Merged list of splits
    """
    splits_copy = copy.deepcopy(splits)
    max_level = max(list(map(lambda s: s[1], splits_copy)))

    # Process levels from highest to lowest
    for ml in range(max_level, 0, -1):
        i = 0
        while i < len(splits_copy):
            if splits_copy[i][1] == ml:
                j = i+1
                max_level = ml+1

                # Find mergeable range
                while j < len(splits_copy) and splits_copy[j][1] >= max_level:
                    max_level = max(max_level, splits_copy[j][1])
                    j += 1

                # Check if mergeable within chunk size
                merged_text = ''.join(map(lambda s: s[0], splits_copy[i: j]))
                if j > i+1 and get_token_len(merged_text, tokenizer) <= chunk_size:
                    # Perform merge
                    new_split = [merged_text, ml]
                    splits_copy = splits_copy[:i] + [new_split] + splits_copy[j:]
            i += 1

    return splits_copy


def build_hi_tree(splits: list, data: dict, chunks: list, left_idx: int, right_idx: int, 
                  pre_level: int, chunk_size: int, filename: str, tokenizer=None) -> tuple:
    """
    Recursively build hierarchical text structure from splits.
    
    Args:
        splits: List of [text, level] pairs
        data: Original document metadata
        chunks: Accumulator for final text chunks
        left_idx: Current left index in splits
        right_idx: Current right index in splits
        pre_level: Previous hierarchy level
        chunk_size: Maximum chunk size
        filename: Source filename
        tokenizer: Optional tokenizer
        
    Returns:
        Tuple: (root node, updated chunks list)
    """
    # Initialize root node
    root = {
        'text': '',
        'title': '',
        'id': '',
        'left_index_idx': 0,
        'right_index_idx': 0,
        'children': [],
        'parent': None
    }
    
    # Base case: single text segment
    if left_idx + 1 == right_idx:
        root['text'] = splits[left_idx][0]
        
        # Split long segments into chunks
        _chunks = [
            {
                'text': c,
                'title': '',
                'id': data['_id'] + '_' + str(len(chunks) + i),
                'left_index_idx': len(chunks) + i,
                'right_index_idx': len(chunks) + i + 1,
                'children': [],
                'parent': root
            }
            for i, c in enumerate(split_long_sentence(splits[left_idx][0], chunk_size, filename, tokenizer))
        ]
        # Update chunk indices
        root['left_index_idx'] = len(chunks)
        chunks.extend(_chunks)
        root['children'].extend(_chunks)
        root['right_index_idx'] = len(chunks)
        return root, chunks

    # Recursive case: process child segments
    temp_left_idx = left_idx + 1
    while temp_left_idx < right_idx:
        [_, cur_level] = splits[temp_left_idx]
        
        # Found sibling at same level
        if cur_level == pre_level:
            # Build subtree for left segment
            root['children'].append(
                build_hi_tree(
                    splits, data, chunks, left_idx, temp_left_idx, pre_level+1, chunk_size, filename, tokenizer
                )[0]
            )
            # Update parent reference
            root['children'][-1]['parent'] = root
            left_idx = temp_left_idx

        temp_left_idx += 1

    # Process remaining right segment
    root['children'].append(
        build_hi_tree(
            splits, data, chunks, left_idx, right_idx, pre_level+1, chunk_size, filename, tokenizer
        )[0]
    )
    # Update parent reference
    root['children'][-1]['parent'] = root

    # Update index range
    root['left_index_idx'] = root['children'][0]['left_index_idx']
    root['right_index_idx'] = root['children'][-1]['right_index_idx']
    return root, chunks


if __name__ == '__main__':
    _splits = [
        [
            "a",
            1
        ],
        [
            "b",
            2
        ],
        [
            "c",
            3
        ],
        [
            "d",
            1
        ],
        [
            "e",
            2
        ],
        [
            "f",
            3
        ],
        [
            "g",
            3
        ],
        [
            "h",
            3
        ],
        [
            "i",
            3
        ],
        [
            "j",
            4
        ],
        [
            "k",
            4
        ],
        [
            "l",
            3
        ],
        [
            "m",
            4
        ],
        [
            "n",
            1
        ],
        [
            "o",
            2
        ],
        [
            "p",
            3
        ],
        [
            "q",
            4
        ],
        [
            "r",
            4
        ],
        [
            "s",
            4
        ],
        [
            "t",
            4
        ],
        [
            "u",
            4
        ],
        [
            "v",
            4
        ],
        [
            "w",
            4
        ],
        [
            "x",
            4
        ],
        [
            "y",
            4
        ],
        [
            "z",
            4
        ]
    ]
    _data = {'_id': 'text.md'}
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    
    # Build and print tree structure
    r, c = build_hi_tree(_splits, _data, [], 0, len(_splits), 1, 200, 'test.md')
    print(r)
    print(c)
