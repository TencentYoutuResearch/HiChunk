import json
import time
import re
import requests
import argparse
import sys
import os

from openai import OpenAI


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


# If using ChatGPT
client = OpenAI(api_key="Insert OpenAI key here")


# Argument parsing
parser = argparse.ArgumentParser(description="Process some text.")
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--books_dir', type=str, required=True)


args = parser.parse_args()
model_type = args.model_type

if model_type not in ["Gemini", "ChatGPT", "Deepseek", "Qwen3-4B"]:
    print("Choose Valid Model Type")
    sys.exit(1)


# Count_Words idea is to approximate the number of tokens in the sentence. We are assuming 1 word ~ 1.2 Tokens
def count_words(input_string):
    words = input_string.split()
    return round(1.2*len(words))


# Function to add IDs to each Dataframe Row
# def add_ids(p):
#     global current_id
#     # Add ID to the chunk
#     p = f'ID {current_id}: {p}'
#     current_id += 1
#     return p


def add_ids_(p, current_id):
    # Add ID to the chunk
    p = f'ID {current_id}: {p}'
    return p


system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph(not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. 

Aim for a good balance between identifying content shifts and keeping groups manageable."""


def ds_prompt(user_prompt):
    while True:
        try:
            _prompt = (f"<｜begin▁of▁sentence｜>{system_prompt}<｜User｜>{user_prompt}<｜Assistant｜>"
                       f"<think>\nOk, done thinking.\n</think>\n\n")
            _post_data = {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "temperature": 0.1,
                "prompt": _prompt,
                'max_tokens': 4096,
            }
            response = requests.post(f"{os.environ['DS_BASE_URL']}/v1/completions", json=_post_data, timeout=600)
            res = response.json()
            return res['choices'][0]['text']
        except Exception as e:
            if str(e) == "list index out of range":
                print("Deepseek thinks prompt is unsafe")
                return "content_flag_increment"
            else:
                print(f"An error occurred: {e}. Retrying in 1 minute...")
                time.sleep(60)  # Wait for 1 minute before retrying


def qw_prompt(user_prompt):
    while True:
        try:
            _prompt = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n"
                       f"<|im_start|>assistant\n<think>\n</think>\n\n")
            _post_data = {
                "model": "/home/tione/notebook/vincentwslu/models/Qwen3-4B",
                "temperature": 0.6,
                "prompt": _prompt,
                'max_tokens': 4096,
            }
            response = requests.post(f"{os.environ['DS_BASE_URL']}/v1/completions", json=_post_data, timeout=600)
            res = response.json()
            return res['choices'][0]['text']
        except Exception as e:
            if str(e) == "list index out of range":
                print("Deepseek thinks prompt is unsafe")
                return "content_flag_increment"
            else:
                print(f"An error occurred: {e}. Retrying in 1 minute...")
                time.sleep(60)  # Wait for 1 minute before retrying


def gpt_prompt(user_prompt):
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ])
            return completion.choices[0].message.content
        except Exception as e:
            if str(e) == "list index out of range":
                print("GPT thinks prompt is unsafe")
                return "content_flag_increment"
            else:
                print(f"An error occurred: {e}. Retrying in 1 minute...")
                time.sleep(60)  # Wait for 1 minute before retrying


def LLM_prompt(model_type, user_prompt):
    if model_type == "Deepseek":
        return ds_prompt(user_prompt)
    elif model_type == "Qwen3-4B":
        return qw_prompt(user_prompt)
    elif model_type == "ChatGPT":
        return gpt_prompt(user_prompt)


def build_prompt(chunk_number, id_chunks_without_title):
    word_count = 0
    i = 0
    while word_count < 550 and i + chunk_number < len(id_chunks_without_title) - 1:
        i += 1
        final_document = "\n".join(map(lambda k: id_chunks_without_title[k], range(chunk_number, i + chunk_number)))
        word_count = count_words(final_document)

    if i == 1:
        final_document = "\n".join(map(lambda k: id_chunks_without_title[k], range(chunk_number, i + chunk_number)))
    else:
        final_document = "\n".join(map(lambda k: id_chunks_without_title[k], range(chunk_number, i - 1 + chunk_number)))

    question = f"\nDocument:\n{final_document}"

    word_count = count_words(final_document)
    chunk_number = chunk_number + i - 1

    return question, word_count, chunk_number


def get_final_chunks(new_id_list, id_chunks):
    # Create final dataframe from chunks
    new_final_chunks = []
    for i in range(len(new_id_list)):
        # Calculate the start and end indices of each chunk
        start_idx = new_id_list[i - 1] if i > 0 else 0
        end_idx = new_id_list[i]
        new_final_chunks.append('\n'.join(id_chunks[start_idx: end_idx]))

    return new_final_chunks


def process(result):
    if 'splits' in result:
        print('load', result['filepath'])
        return result

    print(result['filepath'])
    start_time = time.time()
    paragraph_chunks = open(result['filepath']).read().splitlines()
    current_id = 0
    id_chunks = []
    id_chunks_without_title = []
    for c in paragraph_chunks:
        id_chunks.append(add_ids_(c, current_id))
        id_chunks_without_title.append(add_ids_(c.lstrip('# '), current_id))
        current_id += 1

    chunk_number = 0
    new_id_list = []
    word_count_aux = []
    while chunk_number < len(id_chunks_without_title) - 5:
        question, word_count, chunk_number = build_prompt(chunk_number, id_chunks_without_title)
        word_count_aux.append(word_count)

        prompt = system_prompt + question
        gpt_output = LLM_prompt(model_type=model_type, user_prompt=prompt)
        gpt_output = gpt_output.rsplit("</think>", 1)[-1]

        # For books where there is dubious content, Gemini refuses to run the prompt and returns mistake.
        # This is to avoid being stalled here forever.
        if gpt_output == "content_flag_increment":
            chunk_number = chunk_number + 1
            continue

        pattern = r"Answer: ID \d+"
        match = re.search(pattern, gpt_output)

        if match is None:
            print(gpt_output)
            print("repeat this one")
            continue

        gpt_output1 = match.group(0)
        print(gpt_output1)
        pattern = r'\d+'
        match = re.search(pattern, gpt_output1)
        chunk_number = int(match.group())
        new_id_list.append(chunk_number)
        if new_id_list[-1] == chunk_number:
            chunk_number = chunk_number + 1

    # Add the last chunk to the list
    new_id_list.append(len(id_chunks))

    # Remove IDs as they no longer make sense here.
    id_chunks = list(map(lambda c: re.sub(r'^ID \d+:\s*', '', c), id_chunks))

    # Get final dataframe from chunks
    new_final_chunks = get_final_chunks(new_id_list, id_chunks)

    end_time = time.time()
    # Write new Chunks Dataframe
    result['splits'] = list(map(lambda c: [c, 1], new_final_chunks))
    result['time_cost'] = end_time - start_time
    with open(f"{out_path}/{model_type}_{result['filepath'].split('/')[-1]}.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


book_files = os.listdir(args.books_dir)
out_path = f'{args.books_dir.replace("/docs/", "/LumberChunker_wot_dt/")}'
create_directory(out_path)
output_file = f"{out_path}/{model_type}.json"
try:
    results = json.load(open(output_file))
except Exception as e:
    results = {}
    for book_file in book_files:
        if os.path.exists(os.path.join(out_path, f'{model_type}_{book_file}.json')):
            results[book_file] = json.load(open(os.path.join(out_path, f'{model_type}_{book_file}.json')))


for bf in book_files:
    if bf in results:
        results[bf]['filepath'] = os.path.join(args.books_dir, bf)
tasks = [results[bf] if bf in results else {'filepath': os.path.join(args.books_dir, bf)} for bf in book_files]
# with multiprocessing.Pool(processes=4) as pool:
#     results = [r for r in pool.imap_unordered(process,tqdm(tasks))]
results = [process(t) for t in tasks]
results = {r['filepath'].split('/')[-1]: r for r in results}
results = {bf: results[bf] for bf in book_files}
with open(f"{out_path}/{model_type}.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
