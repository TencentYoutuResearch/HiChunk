import json
import multiprocessing
import os
import re
import sys
import time
from typing import List, Dict

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

# Prompt template for extracting factual statements from reference answers
FACT_EXTRACTION_PROMPT = """
### Task
Extract distinct factual statements from the reference answer that could be independently verified.
Respond ONLY with a JSON object containing a "facts" list of strings.

### Example
Input:
  Question: "What causes seasons?"
  Reference: "Seasonal changes result from Earth's axial tilt. This tilt causes different hemispheres to receive varying sunlight."

Output:
{{
  "facts": [
    "Seasonal changes result from Earth's axial tilt",
    "The axial tilt causes different hemispheres to receive varying sunlight"
  ]
}}

### Actual Input
Question: "{question}"
Reference Answer: "{reference}"

### Your Response:
"""

# Prompt template for checking fact coverage in generated responses
FACT_COVERAGE_PROMPT = """
### Task
For each factual statement from the reference, determine if it's covered in the response.
Respond ONLY with a JSON object containing a "classifications" list. Each item should have:
- "statement": the exact fact from reference
- "attributed": 1 if covered, 0 if not

### Example
Response: "Seasons are caused by Earth's tilted axis"
Reference Facts: [
  "Seasonal changes result from Earth's axial tilt",
  "The axial tilt causes different hemispheres to receive varying sunlight"
]

Output:
{{
  "classifications": [
    {{"statement": "Seasonal changes result from Earth's axial tilt", "attributed": 1}},
    {{"statement": "The axial tilt causes different hemispheres to receive varying sunlight", "attributed": 0}}
  ]
}}

### Actual Input
Question: "{question}"
Response: "{response}"
Reference Facts: {facts}

### Your Response:
"""


def call_api(prompt, max_gen):
    """Call the DeepSeek API to generate text completions"""
    _prompt = f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\nOk, done thinking.\n</think>\n\n"
    post_data = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "prompt": _prompt,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_gen,
        "n": 5,  # Number of completions to generate
    }
    while True:
        try:
            # Send request to DeepSeek API
            response = requests.post(f"{os.environ['DS_BASE_URL']}/v1/completions", json=post_data, timeout=600)
            res = response.json()
            response.close()
            # Extract generated text from all choices
            return list(map(lambda r: r['text'], res['choices']))
        except:
            # Retry on failure with delay
            time.sleep(30)


def compute_coverage_score(
        question: str,
        reference: str,
        response: str,
        reference_facts: list[str] = None,
) -> (float, list[str], list[dict]):
    """
    Calculate coverage score (0.0-1.0) by measuring what percentage of
    reference facts are covered in the response.
    
    Returns:
        tuple: (coverage_score, extracted_facts, coverage_classifications)
    """
    # Handle edge cases
    if not reference.strip():
        return 1.0, [], []  # Perfect coverage for empty reference

    # Step 1: Extract facts from reference
    facts = reference_facts
    if facts is None:
        facts = _extract_facts(question, reference)

    if not facts:
        return np.nan, [], []  # Failed to extract facts

    # Step 2: Check fact coverage in response
    coverage = _check_fact_coverage(question, facts, response)

    # Calculate coverage score
    if coverage:
        # Compute attribution rates for all completions
        attributed = list(map(lambda cs: [a["attributed"] for c in cs for a in c], coverage))
        return list(map(lambda a: sum(a)/len(a), attributed)), facts, coverage
    return np.nan, [], []  # Return NaN on failure


def _extract_facts(
        question: str,
        reference: str,
) -> List[str]:
    """Extract factual statements from reference answer using LLM"""
    prompt = FACT_EXTRACTION_PROMPT.format(
        question=question,
        reference=reference  # Truncate long references
    )

    while True:
        try:
            # Call API to extract facts
            response = call_api(prompt, max_gen=16*1024)
            # Parse JSON response from generated text
            json_str = re.search(r'\{.+\}', response.rsplit('</think>', 1)[-1], re.M | re.S | re.U).group(0)
            data = json.loads(json_str)
            return _validate_facts(data.get("facts", []))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(e)
            continue  # Retry on parsing failure


def _validate_facts(facts: List) -> List[str]:
    """Ensure facts are valid non-empty strings"""
    return list(filter(lambda x: x and str(x).strip(), facts))


def _check_fact_coverage(
        question: str,
        facts: List[str],
        response: str,
):
    """Check which facts are covered in the response using LLM"""
    results = []
    # Normalize response to list
    if isinstance(response, str):
        response = [response]
        
    for r in response:
        prompt = FACT_COVERAGE_PROMPT.format(
            question=question,
            response=r,  # Truncate long responses
            facts=json.dumps(facts)
        )
        while True:
            try:
                res = call_api(prompt, max_gen=16*1024)
                # Parse classification results from all completions
                json_str = [re.search(r'\{.+\}', r.rsplit('</think>', 1)[-1], re.M | re.S | re.U).group(0) for r in res]
                data = map(lambda js: json.loads(js), json_str)
                results.append([_validate_classifications(d.get("classifications", [])) for d in data])
                break
            except:
                continue  # Retry on failure
    return results  # Return empty list after max retries


def _validate_classifications(classifications: List) -> List[Dict]:
    """Ensure classifications have required fields and proper types"""
    valid = []
    for item in classifications:
        try:
            # Validate required fields and types
            if item["attributed"] in {0, 1}:
                valid.append({
                    "statement": str(item["statement"]),
                    "attributed": int(item["attributed"])
                })
        except (TypeError, ValueError) as e:
            print(e)
            continue
    return valid


# Main script execution starts here
answer_llm = sys.argv[1]  # First CLI argument: answer LLM name
token_num = sys.argv[2]   # Second CLI argument: token number
dataset = 'HiCBench'      # Dataset name
origin_data_file = f'./dataset/qas/{dataset}.jsonl'  # Original data file path

# Result paths for different configurations
result_paths = [
    f'./pred/{answer_llm}/BgeM3/C200_tk{token_num}/eval_{dataset}.jsonl',
    f'./pred/{answer_llm}/BgeM3/SC100000_tk{token_num}/eval_{dataset}.jsonl',
    f'./pred/{answer_llm}/BgeM3/LC100000_tk{token_num}/eval_{dataset}.jsonl',
    f'./pred/{answer_llm}/BgeM3/HC200_L10_tk{token_num}/eval_{dataset}.jsonl',
    f'./pred/{answer_llm}/BgeM3/HC200_L10_tk{token_num}_AM1/eval_{dataset}.jsonl',
]


def process_sample(res):
    """Process a single sample to compute fact coverage metrics"""
    i, q, ref, pred = res['index'], res['input'], res['answers'][0], res['pred']
    if 'ref_facts_metrics' not in res:
        # Compute coverage metrics if not already present
        res['ref_facts_metrics'] = []
        task_results1 = compute_coverage_score(q, ref, pred, reference_facts=res.get('facts', None))
        (recall, ref_facts, ref_facts_cov) = task_results1
        res['ref_facts_metrics'].append(
            {
                'facts_recall': recall,
                'ref_facts': ref_facts,
                'ref_facts_cov': ref_facts_cov,
            }
        )
    print(res['index'])  # Progress indicator
    return res


def main():
    """Main processing function"""
    for result_path1 in result_paths:
        # Load all task results
        tasks = list(map(lambda x: json.loads(x), open(result_path1).readlines()))

        # Process tasks in parallel
        with multiprocessing.Pool(processes=15) as pool:
            sample_results = [p for p in pool.imap_unordered(process_sample, tqdm(tasks))]

        # Sort results by index
        sample_results = sorted(sample_results, key=lambda x: x['index'])
        for i, r in zip(range(len(tasks)), sample_results):
            assert r['index'] == i
            tasks[i] = r
        print(f"Completed all samples in {result_path1}.")
        show_metrics(tasks)  # Display metrics

        # Save results with coverage metrics
        with open(result_path1.replace(f'eval_{dataset}.jsonl', f'FC_eval_{dataset}.jsonl'), 'w') as f:
            for result in tasks:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def show_metrics(sample_results):
    """Calculate and display evaluation metrics"""
    fact_ref_num, fact_pt_num, fact_ref_num_std, fact_pt_num_std = 0, 0, 0, 0
    sample_num = 0
    for result in filter(lambda x: 'ref_facts_metrics' in x, sample_results):
        sample_num += 1

        # Calculate statistics
        _fact_ref_num = list(map(lambda x: x*len(x['ref_facts']), result['ref_facts_metrics']))
        _fact_pt_num = [len(res['ref_facts'])*fr for res in result['ref_facts_metrics'] for fr in res['facts_recall']]
        fact_ref_num += np.mean(_fact_ref_num)
        fact_pt_num += np.mean(_fact_pt_num)
        fact_ref_num_std += np.std(_fact_ref_num)
        fact_pt_num_std += np.std(_fact_pt_num)

    # Print metrics
    print("sample_num: ", sample_num)
    print('fact_ref_num: ', fact_ref_num)
    print('fact_pt_num: ', fact_pt_num)
    print('fact_ref_num_std: ', fact_ref_num_std/sample_num)
    print('fact_pt_num_std: ', fact_pt_num_std/sample_num)
    print(f'{dataset}_fr: {fact_pt_num*100/(fact_ref_num+1e-20):.3f}')


if __name__ == '__main__':
    main()
