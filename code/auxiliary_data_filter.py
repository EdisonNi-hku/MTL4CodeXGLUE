from transformers import RobertaTokenizer, T5Tokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import multiprocessing
import random
import json

codet5_tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base', cache_dir='cache', local_files_only=True)
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base', cache_dir='cache', local_files_only=True)


def count_tokens(item):
    s, tokenizer = item
    s = s.replace('</s>', '<unk>')
    s_ids = tokenizer.encode(s)
    return len(s_ids)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_fn", type=str, default='srl/python.jsonl')
    parser.add_argument("--output_fn", type=str, default='srl/filtered_python.jsonl')
    args = parser.parse_args()

    random.seed(1234)
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    all_data = []
    with open(args.input_fn, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            all_data.append(json.loads(line))
    print("Number of data before filter:", len(all_data))
    remove_null = [d for d in all_data if d['srl'] is not None]
    print("After remove null:", len(remove_null))
    targets = [d['sum'] + ' ' + d['srl'] for d in remove_null]
    t5_items = [(tgt, t5_tokenizer) for tgt in targets]
    codet5_items = [(tgt, codet5_tokenizer) for tgt in targets]
    t5_count = pool.map(count_tokens, tqdm(t5_items, total=len(t5_items), desc='t5 code'))
    codet5_count = pool.map(count_tokens, tqdm(codet5_items, total=len(codet5_items), desc='codet5 code'))
    filtered_data = []
    for i in range(len(t5_count)):
        if t5_count[i] <= 509 and codet5_count[i] <= 509:
            filtered_data.append(remove_null[i])
    print("After filtered by length:", len(filtered_data))
    with open(args.output_fn, 'w') as f:
        for d in filtered_data:
            json.dump(d, f)
            f.write('\n')


if __name__ == '__main__':
    main()







