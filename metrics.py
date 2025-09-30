import re
import string

import jieba
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

"""
Metrics Module

This module contains various text similarity and evaluation metrics functions
for natural language processing tasks. It includes implementations for:
- Text normalization (English and Chinese)
- Numeric matching scores
- Retrieval evaluation metrics
- Code similarity
- Classification metrics
- BLEU and ROUGE scores
- F1 scores (English and Chinese)
- Longest common substring/subsequence
- Evidence recall metrics
"""


# Normalizes English text by:
# 1. Converting to lowercase
# 2. Removing punctuation
# 3. Removing articles (a, an, the)
# 4. Fixing whitespace
def normalize_answer(s):
    """
    Args:
        s (str): Input string to normalize
        
    Returns:
        str: Normalized string
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(filter(lambda ch: ch not in exclude, text))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Normalizes Chinese text by:
# 1. Converting to lowercase
# 2. Removing punctuation (including Chinese punctuation)
# 3. Fixing whitespace
def normalize_zh_answer(s):
    """
    Args:
        s (str): Input Chinese string to normalize
        
    Returns:
        str: Normalized string
    """
    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(filter(lambda ch: ch not in all_punctuation, text))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


# Calculates BLEU score
def bleu_score(prediction, ground_truth, **kwargs):
    """
    Args:
        prediction (str): Model prediction
        ground_truth (str): Reference text
        
    Returns:
        float: BLEU score
    """
    ground_truth_words = [ground_truth.split()]
    prediction_words = prediction.split()
    return sentence_bleu(ground_truth_words, prediction_words, weights=(0.25, 0.25, 0.25, 0.25))


# Calculates ROUGE-L score for English text
def rouge_score(prediction, ground_truth, **kwargs):
    """
    Args:
        prediction (str): Model prediction
        ground_truth (str): Reference text
        
    Returns:
        float: ROUGE-L F1 score
    """
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


# Calculates ROUGE-L score for Chinese text
def rouge_zh_score(prediction, ground_truth, **kwargs):
    """
    Args:
        prediction (str): Model prediction in Chinese
        ground_truth (str): Reference text in Chinese
        
    Returns:
        float: ROUGE-L F1 score
    """
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score


# Calculates F1 score for English text
def f1_score(prediction, ground_truth, **kwargs):
    """
    Args:
        prediction (list): List of predicted tokens
        ground_truth (list): List of reference tokens
        
    Returns:
        float: F1 score
    """
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# Question Answering F1 score for English text
def qa_f1_score(prediction, ground_truth, **kwargs):
    """
    Args:
        prediction (str): Model prediction
        ground_truth (str): Reference answer
        
    Returns:
        float: F1 score based on normalized tokens
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


# Question Answering F1 score for Chinese text
def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    """
    Args:
        prediction (str): Model prediction in Chinese
        ground_truth (str): Reference answer in Chinese
        
    Returns:
        float: F1 score based on normalized tokens
    """
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = list(map(lambda token: normalize_zh_answer(token), prediction_tokens))
    ground_truth_tokens = list(map(lambda token: normalize_zh_answer(token), ground_truth_tokens))
    prediction_tokens = list(filter(lambda token: len(token) > 0, prediction_tokens))
    ground_truth_tokens = list(filter(lambda token: len(token) > 0, ground_truth_tokens))
    return f1_score(prediction_tokens, ground_truth_tokens)


# Binary evidence recall metric
def cal_evidence_recall(gt, hyp):
    """
    Args:
        gt (str): Ground truth text
        hyp (str): Hypothesis text
        
    Returns:
        float: 1.0 if ground truth is substring of hypothesis, else 0.0
    """
    temp_gt = normalize_answer(gt)
    temp_hyp = normalize_answer(hyp)
    if temp_gt in temp_hyp:
        return 1.0
    else:
        return 0.0
