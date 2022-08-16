from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
import math
from _utils import *

logger = logging.getLogger(__name__)


class PlainCodeDataset(torch.utils.data.Dataset):
    def __init__(self, codes):
        self.codes = codes

    def __getitem__(self, idx):
        return self.codes[idx]

    def __len__(self):
        return len(self.codes)


def get_src_lang_from_task(task, sub_task):
    if task in ['summarize', 'identifier', 'dataflow']:
        return sub_task
    elif task == 'defect':
        return 'c'
    elif task == 'translate' and sub_task == 'cs-java':
        return 'c_sharp'
    elif task == 'translate' and sub_task == 'java-cs':
        return 'java'
    elif task == 'clone':
        return 'java'
    else:
        raise ValueError("undefined task in get_src_lang_from_task")


def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    examples = read_examples(filename, args.data_num, args.task)

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_clone_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + '_all' if args.data_num == -1 else '_%d' % args.data_num)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_clone_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_multi_gen_data(args, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        examples_data_dict = torch.load(cache_fn)
    else:
        examples_data_dict = {}

        task_list = ['summarize', 'translate', 'refine', 'concode', 'defect']
        for task in task_list:
            if task == 'summarize':
                sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
            elif task == 'translate':
                sub_tasks = ['java-cs', 'cs-java']
            elif task == 'refine':
                sub_tasks = ['small', 'medium']
            else:
                sub_tasks = ['none']
            args.task = task
            for sub_task in sub_tasks:
                args.sub_task = sub_task
                if task == 'summarize':
                    args.max_source_length = 256
                    args.max_target_length = 128
                elif task == 'translate':
                    args.max_source_length = 320
                    args.max_target_length = 256
                elif task == 'refine':
                    if sub_task == 'small':
                        args.max_source_length = 130
                        args.max_target_length = 120
                    else:
                        args.max_source_length = 240
                        args.max_target_length = 240
                elif task == 'concode':
                    args.max_source_length = 320
                    args.max_target_length = 150
                elif task == 'defect':
                    args.max_source_length = 512
                    args.max_target_length = 3  # as do not need to add lang ids

                filename = get_filenames(args.data_dir, args.task, args.sub_task, split_tag)
                examples = read_examples(filename, args.data_num, args.task)
                if is_sample:
                    examples = random.sample(examples, min(5000, len(examples)))
                if split_tag == 'train':
                    calc_stats(examples, tokenizer, is_tokenize=True)
                else:
                    calc_stats(examples)

                tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
                if args.data_num == -1:
                    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
                else:
                    features = [convert_examples_to_features(x) for x in tuple_examples]
                all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
                if only_src:
                    data = TensorDataset(all_source_ids)
                else:
                    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
                    data = TensorDataset(all_source_ids, all_target_ids)
                examples_data_dict['{}_{}'.format(task, sub_task) if sub_task != 'none' else task] = (examples, data)

        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(examples_data_dict, cache_fn)
            logger.info("Save data into %s", cache_fn)
    return examples_data_dict


def load_and_cache_single_task_aux_data(args, single_task, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        examples_data_dict = torch.load(cache_fn)
    else:
        examples_data_dict = {}

        task_list = [single_task, 'dataflow', 'identifier', 'summarize_srl', 'translate_cloze']
        if '0' not in args.aux_type:
            task_list.remove('dataflow')
        if '1' not in args.aux_type:
            task_list.remove('identifier')
        if '2' not in args.aux_type:
            task_list.remove('summarize_srl')
        if '3' not in args.aux_type:
            task_list.remove('translate_cloze')
        for task in task_list:
            if task in ['identifier', 'dataflow', 'summarize_srl', 'translate_cloze'] and split_tag != 'train':
                continue
            args.task = task
            if task == 'summarize':
                args.max_source_length = 256
                args.max_target_length = 128
            elif task == 'summarize_srl':
                args.max_source_length = 256
                args.max_target_length = 512
            elif task in ['translate', 'translate_cloze']:
                args.max_source_length = 320
                args.max_target_length = 256
            elif task in ['identifier', 'dataflow']:
                args.max_source_length = 512
                args.max_target_length = 512

            filename = get_filenames(args.data_dir, args.task, args.sub_task, split_tag)
            examples = read_examples(filename, args.data_num, args.task)
            if is_sample:
                examples = random.sample(examples, min(5000, len(examples)))
            if split_tag == 'train':
                calc_stats(examples, tokenizer, is_tokenize=True)
            else:
                calc_stats(examples)

            if task != 'identifier':
                tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
                if args.data_num == -1:
                    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
                else:
                    features = [convert_examples_to_features(x) for x in tuple_examples]
                if task in ['dataflow', 'translate_cloze', 'summarize_srl']:
                    features = random.sample(features, math.ceil((args.aux_percentage / 100) * len(features)))
                all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
                if only_src:
                    data = TensorDataset(all_source_ids)
                else:
                    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
                    data = TensorDataset(all_source_ids, all_target_ids)
                examples_data_dict['{}_{}'.format(task, args.sub_task) if args.sub_task != 'none' else task] = (examples, data)
            else:
                codes = [example.source for example in examples]
                codes = random.sample(codes, math.ceil((args.aux_percentage / 100) * len(codes)))
                data = PlainCodeDataset(codes)
                examples_data_dict['{}_{}'.format(task, args.sub_task) if args.sub_task != 'none' else task] = (examples, data)

        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(examples_data_dict, cache_fn)
            logger.info("Save data into %s", cache_fn)
    return examples_data_dict


def load_and_cache_all_aux_gen_data(args, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        examples_data_dict = torch.load(cache_fn)
    else:
        examples_data_dict = {}

        task_list = ['summarize', 'translate', 'refine', 'concode', 'defect', 'dataflow', 'identifier', 'summarize_srl', 'translate_cloze']
        if '0' not in args.aux_type:
            task_list.remove('dataflow')
        if '1' not in args.aux_type:
            task_list.remove('identifier')
        if '2' not in args.aux_type:
            task_list.remove('summarize_srl')
        if '3' not in args.aux_type:
            task_list.remove('translate_cloze')
        for task in task_list:
            if task in ['identifier', 'dataflow', 'summarize_srl', 'translate_cloze'] and split_tag != 'train':
                continue
            if task in ['summarize', 'summarize_srl']:
                sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
            elif task in ['identifier', 'dataflow']:
                sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php', 'c_sharp']
            elif task in ['translate', 'translate_cloze']:
                sub_tasks = ['java-cs', 'cs-java']
            elif task == 'refine':
                sub_tasks = ['small', 'medium']
            else:
                sub_tasks = ['none']
            args.task = task
            for sub_task in sub_tasks:
                args.sub_task = sub_task
                if task == 'summarize':
                    args.max_source_length = 256
                    args.max_target_length = 128
                elif task == 'summarize_srl':
                    args.max_source_length = 256
                    args.max_target_length = 512
                elif task in ['translate', 'translate_cloze']:
                    args.max_source_length = 320
                    args.max_target_length = 256
                elif task == 'refine':
                    if sub_task == 'small':
                        args.max_source_length = 130
                        args.max_target_length = 120
                    else:
                        args.max_source_length = 240
                        args.max_target_length = 240
                elif task == 'concode':
                    args.max_source_length = 320
                    args.max_target_length = 150
                elif task == 'defect':
                    args.max_source_length = 512
                    args.max_target_length = 3  # as do not need to add lang ids
                elif task in ['identifier', 'dataflow']:
                    args.max_source_length = 512
                    args.max_target_length = 512

                filename = get_filenames(args.data_dir, args.task, args.sub_task, split_tag)
                examples = read_examples(filename, args.data_num, args.task)
                if is_sample:
                    examples = random.sample(examples, min(5000, len(examples)))
                if split_tag == 'train':
                    calc_stats(examples, tokenizer, is_tokenize=True)
                else:
                    calc_stats(examples)

                if task != 'identifier':
                    tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
                    if args.data_num == -1:
                        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
                    else:
                        features = [convert_examples_to_features(x) for x in tuple_examples]
                    if task in ['dataflow', 'summarize_srl', 'translate_cloze']:
                        features = random.sample(features, math.ceil((args.aux_percentage / 100) * len(features)))
                    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
                    if only_src:
                        data = TensorDataset(all_source_ids)
                    else:
                        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
                        data = TensorDataset(all_source_ids, all_target_ids)
                    examples_data_dict['{}_{}'.format(task, sub_task) if sub_task != 'none' else task] = (examples, data)
                else:
                    codes = [example.source for example in examples]
                    codes = random.sample(codes, math.ceil((args.aux_percentage / 100) * len(codes)))
                    data = PlainCodeDataset(codes)
                    examples_data_dict['{}_{}'.format(task, sub_task) if sub_task != 'none' else task] = (examples, data)

        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(examples_data_dict, cache_fn)
            logger.info("Save data into %s", cache_fn)
    return examples_data_dict


def get_filenames(data_root, task, sub_task, split=''):
    if task == 'concode':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
    elif task == 'summarize':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'refine':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.buggy-fixed.buggy,{}/train.buggy-fixed.fixed'.format(data_dir, data_dir)
        dev_fn = '{}/valid.buggy-fixed.buggy,{}/valid.buggy-fixed.fixed'.format(data_dir, data_dir)
        test_fn = '{}/test.buggy-fixed.buggy,{}/test.buggy-fixed.fixed'.format(data_dir, data_dir)
    elif task == 'translate':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.java-cs.txt.cs,{}/train.java-cs.txt.java'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.cs,{}/valid.java-cs.txt.java'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.cs,{}/test.java-cs.txt.java'.format(data_dir, data_dir)
        else:
            train_fn = '{}/train.java-cs.txt.java,{}/train.java-cs.txt.cs'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.java,{}/valid.java-cs.txt.cs'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.java,{}/test.java-cs.txt.cs'.format(data_dir, data_dir)
    elif task == 'translate_cloze':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.cs-java.cs,{}/train.cs-java.java'.format(data_dir, data_dir)
            dev_fn = ''
            test_fn = ''
        else:
            train_fn = '{}/train.java-cs.java,{}/train.java-cs.cs'.format(data_dir, data_dir)
            dev_fn = ''
            test_fn = ''
    elif task == 'clone':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.txt'.format(data_dir)
        dev_fn = '{}/valid.txt'.format(data_dir)
        test_fn = '{}/test.txt'.format(data_dir)
    elif task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'dataflow':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = ''
        test_fn = ''
    elif task == 'identifier':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = ''
        test_fn = ''
    elif task == 'summarize_srl':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = ''
        test_fn = ''
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'summarize': read_summarize_examples,
        'refine': read_refine_examples,
        'translate': read_translate_examples,
        'concode': read_concode_examples,
        'clone': read_clone_examples,
        'defect': read_defect_examples,
        'dataflow': read_dataflow_examples,
        'identifier': read_dataflow_examples,
        'summarize_srl': read_summarize_srl_examples,
        'translate_cloze': read_translate_examples,
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def save_checkpoint(training_state, optimizer, scheduler, model_to_save, output_model_file,
                    output_optimizer_file, output_scheduler_file, last_output_dir):
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(optimizer.state_dict(), output_optimizer_file)
    torch.save(scheduler.state_dict(), output_scheduler_file)
    with open(os.path.join(last_output_dir, "training_state.json"), 'w') as f:
        json.dump(training_state, f)


def convert_src_tgt_to_features(item):
    src, tgt, idx, tokenizer, args, task, sub_task = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if sub_task != 'none':
            source_str = "{} {}: {}".format(task, sub_task, src)
        else:
            source_str = "{}: {}".format(task, src)
    else:
        source_str = src

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=512, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1

    target_str = tgt
    if args.add_lang_ids:
        target_str = add_lang_by_task(tgt, task, sub_task)

    target_str = target_str.replace('</s>', '<unk>')
    target_ids = tokenizer.encode(target_str, max_length=512, padding='max_length',
                                  truncation=True)
    assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        idx,
        source_ids,
        target_ids,
    )