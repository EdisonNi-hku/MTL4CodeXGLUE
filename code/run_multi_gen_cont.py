# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import torch
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
from itertools import cycle
import multiprocessing
import time
import sys
import pdb
import json

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_elapse_time, load_and_cache_multi_gen_data, save_checkpoint
from configs import add_args, set_seed, set_dist

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
WORKER_NUM = 0


def get_max_trg_len_by_task(task, sub_task):
    if task == 'summarize':
        max_target_length = 128
    elif task == 'translate':
        max_target_length = 256
    elif task == 'refine':
        if sub_task == 'small':
            max_target_length = 120
        else:
            max_target_length = 240
    elif task == 'concode':
        max_target_length = 150
    elif task == 'defect':
        max_target_length = 3
    elif task in ['identifier', 'dataflow']:
        max_target_length = 512
    return max_target_length


def get_bs(cur_task, model_tag, gas, times):
    task = cur_task.split('_')[0]
    sub_task = cur_task.split('_')[-1]
    if 'codet5_small' in model_tag:
        bs = math.ceil(32 * times)
        if task == 'summarize' or task == 'translate' or (task == 'refine' and sub_task == 'small'):
            bs = math.ceil(64 * times)
    else:
        # codet5_base
        bs = math.ceil(32 * times)
        if task == 'translate':
            bs = math.ceil(24 * times)
        elif task == 'summarize':
            bs = math.ceil(40 * times)
    bs = int(bs / gas)
    return bs


def eval_bleu(args, eval_data, eval_examples, model, tokenizer, split_tag, cur_task, criteria):
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    task = cur_task.split('_')[0]
    sub_task = cur_task.split('_')[-1]
    max_target_length = get_max_trg_len_by_task(task, sub_task)

    model.eval()
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                model_to_generate = model.module if hasattr(model, 'module') else model
                preds = model_to_generate.generate(source_ids,
                                                   attention_mask=source_mask,
                                                   use_cache=True,
                                                   num_beams=5,
                                                   max_length=max_target_length,  # length_penalty=0.6,
                                                   early_stopping=task == 'summarize')
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    if task == 'defect':
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc, 'bleu': 0, 'codebleu': 0}

    else:
        dev_accs = []
        predictions = []
        res_dir = os.path.join(args.res_dir, cur_task)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        output_fn = os.path.join(res_dir, "test_{}.output".format(criteria))
        gold_fn = os.path.join(res_dir, "test_{}.gold".format(criteria))
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if task == 'summarize':
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')

        if task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if split_tag == 'test':
                if task in ['summarize', 'search']:
                    cur_lang = sub_task
                elif task in ['refine', 'concode', 'clone']:
                    cur_lang = 'java'
                elif task == 'defect':
                    cur_lang = 'c'
                elif task == 'translate':
                    cur_lang = 'c_sharp' if sub_task == 'java-cs' else 'java'
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, cur_lang)

        result = {}
        em = np.mean(dev_accs) * 100
        result['em'] = em
        result['bleu'] = bleu
        if not args.task == 'summarize' and split_tag == 'test':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results [%s] *****", cur_task)
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    if args.cont:
        model_file = os.path.join(args.output_dir, "checkpoint-last/pytorch_model.bin")
        model.load_state_dict(torch.load(model_file))
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    fa_dict = {}
    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = './tensorboard/{}'.format('/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples_data_dict = load_and_cache_multi_gen_data(args, pool, tokenizer, 'train', is_sample=False)
        train_data_list = [v[1] for k, v in train_examples_data_dict.items()]
        all_tasks = [k for k, v in train_examples_data_dict.items()]
        total_train_data_num = sum([len(v[0]) for k, v in train_examples_data_dict.items()])

        for cur_task in all_tasks:
            summary_dir = os.path.join(args.output_dir, 'summary')
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            fa_dict[cur_task] = open(os.path.join(summary_dir, '{}_summary.log'.format(cur_task)), 'a+')

        train_dataloader_dict = dict()
        for train_data, cur_task in zip(train_data_list, all_tasks):
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            if args.data_num == -1:
                train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                              batch_size=get_bs(cur_task, args.model_name_or_path, args.gradient_accumulation_steps, args.times),
                                              num_workers=WORKER_NUM, pin_memory=True)
            else:
                train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                              batch_size=get_bs(cur_task, args.model_name_or_path, args.gradient_accumulation_steps, args.times))

            train_dataloader_dict[cur_task] = cycle(train_dataloader)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
        if args.cont:
            optimizer_state = torch.load(os.path.join(args.output_dir, 'checkpoint-last/optimizer.pt'), map_location="cpu")
            scheduler_state = torch.load(os.path.join(args.output_dir, 'checkpoint-last/scheduler.pt'), map_location="cpu")
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Total train data num = %d", total_train_data_num)
        logger.info("  Max step = %d, Save step = %d", args.max_steps, args.save_steps)

        dev_dataset = {}
        if args.cont:
            with open(os.path.join(args.output_dir, "checkpoint-last/training_state.json"), 'r') as f:
                training_state = json.load(f)
        else:
            training_state = {}
            training_state['step'] = 0
            training_state['global_step'] = 0
            training_state['best_bleu_em'] = dict([(k, -1) for k in all_tasks])
            training_state['best_loss'] = dict([(k, 1e6) for k in all_tasks])
            training_state['not_bleu_em_inc_cnt'] = dict([(k, 0) for k in all_tasks])
            training_state['is_early_stop'] = dict([(k, 0) for k in all_tasks])
            training_state['tr_loss'] = 0
            training_state['loss'] = dict([(k, []) for k in all_tasks])
            training_state['bleu_em'] = dict([(k, []) for k in all_tasks])
            training_state['skip_cnt'] = 0
            training_state['nb_tr_steps'] = 0
            training_state['nb_tr_examples'] = 0
            training_state['tr_nb'] = 0
            training_state['logging_loss'] = 0

        patience_pairs = []
        for cur_task in all_tasks:
            task = cur_task.split('_')[0]
            if task == 'summarize':
                patience_pairs.append((cur_task, 2))
            elif task == 'translate':
                patience_pairs.append((cur_task, 5))
            elif task == 'refine':
                patience_pairs.append((cur_task, 5))
            elif task == 'concode':
                patience_pairs.append((cur_task, 3))
            elif task == 'defect':
                patience_pairs.append((cur_task, 2))
        patience_dict = dict(patience_pairs)
        logger.info('Patience: %s', patience_dict)

        probs = [len(x) for x in train_data_list]
        probs = [x / sum(probs) for x in probs]
        probs = [x ** 0.7 for x in probs]
        probs = [x / sum(probs) for x in probs]

        starting_step = training_state['global_step']
        bar = tqdm(total=args.max_steps - starting_step, desc="Training")
        while True:
            cur_task = np.random.choice(all_tasks, 1, p=probs)[0]
            train_dataloader = train_dataloader_dict[cur_task]
            if training_state['is_early_stop'][cur_task]:
                training_state['skip_cnt'] += 1
                if training_state['skip_cnt'] > 50:
                    logger.info('All tasks have early stopped at %d', training_state['step'])
                    break
                continue
            else:
                training_state['skip_cnt'] = 0

            for _ in range(args.gradient_accumulation_steps):
                training_state['step'] += 1
                batch = next(train_dataloader)

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                # logger.info('cur_task: %s, bs: %d', cur_task, source_ids.shape[0])
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)
                # pdb.set_trace()

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                training_state['tr_loss'] += loss.item()

                training_state['nb_tr_examples'] += source_ids.size(0)
                training_state['nb_tr_steps'] += 1
                loss.backward()

            assert(training_state['nb_tr_steps'] % args.gradient_accumulation_steps == 0)
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            training_state['global_step'] += 1
            train_loss = round((training_state['tr_loss'] - training_state['logging_loss']) / (training_state['global_step'] - training_state['tr_nb']), 6)
            bar.update(1)
            bar.set_description("[{}] Train loss {}".format(training_state['step'], round(train_loss, 3)))

            if args.local_rank in [-1, 0] and args.log_steps > 0 and training_state['global_step'] % args.log_steps == 0:
                training_state['logging_loss'] = train_loss
                training_state['tr_nb'] = training_state['global_step']

            if args.do_eval and args.local_rank in [-1, 0] \
                    and args.save_steps > 0 and training_state['global_step'] % args.save_steps == 0:
                # save last checkpoint
                if args.data_num == -1 and args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    output_optimizer_file = os.path.join(last_output_dir, "optimizer.pt")
                    output_scheduler_file = os.path.join(last_output_dir, "scheduler.pt")
                    save_checkpoint(training_state, optimizer, scheduler, model_to_save, output_model_file,
                                    output_optimizer_file, output_scheduler_file, last_output_dir)
                    logger.info("Save the last model into %s", output_model_file)
                    logger.info("Save the optimizer and scheduler into %s and %s" % (
                        output_optimizer_file, output_scheduler_file))
                if training_state['global_step'] % 100000 == 0:
                    step_tag = '{}00k'.format(training_state['global_step'] // 100000)
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-step-{}'.format(step_tag))
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    output_optimizer_file = os.path.join(last_output_dir, "optimizer.pt")
                    output_scheduler_file = os.path.join(last_output_dir, "scheduler.pt")
                    save_checkpoint(training_state, optimizer, scheduler, model_to_save, output_model_file,
                                    output_optimizer_file, output_scheduler_file, last_output_dir)
                    logger.info("Save the last model into %s", output_model_file)
                    logger.info("Save the optimizer and scheduler into %s and %s" % (
                        output_optimizer_file, output_scheduler_file))
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples_data_dict = dev_dataset['dev_loss']
                else:
                    eval_examples_data_dict = load_and_cache_multi_gen_data(args, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples_data_dict

                for cur_task in eval_examples_data_dict.keys():
                    if training_state['is_early_stop'][cur_task]:
                        continue
                    eval_examples, eval_data = eval_examples_data_dict[cur_task]
                    eval_sampler = SequentialSampler(eval_data)
                    if args.data_num == -1:
                        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                                     batch_size=args.eval_batch_size,
                                                     num_workers=4, pin_memory=True)
                    else:
                        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                                     batch_size=args.eval_batch_size)

                    logger.info("  " + "***** Running ppl evaluation on [{}] *****".format(cur_task))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    # Start Evaluating model
                    model.eval()
                    eval_loss, batch_num = 0, 0
                    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
                        batch = tuple(t.to(args.device) for t in batch)
                        source_ids, target_ids = batch
                        source_mask = source_ids.ne(tokenizer.pad_token_id)
                        target_mask = target_ids.ne(tokenizer.pad_token_id)

                        with torch.no_grad():
                            if args.model_type == 'roberta':
                                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                                   target_ids=target_ids, target_mask=target_mask)
                            else:
                                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                                labels=target_ids, decoder_attention_mask=target_mask)
                                loss = outputs.loss
                        if args.n_gpu > 1:
                            loss = loss.mean()
                        eval_loss += loss.item()
                        batch_num += 1
                    # Print loss of dev dataset
                    eval_loss = eval_loss / batch_num
                    result = {'cur_task': cur_task,
                              'global_step': training_state['global_step'],
                              'eval_ppl': round(np.exp(eval_loss), 5),
                              'train_loss': round(train_loss, 5)}
                    training_state['loss'][cur_task].append(result)
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                    logger.info("  " + "*" * 20)

                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_ppl_{}'.format(cur_task),
                                             round(np.exp(eval_loss), 5),
                                             training_state['global_step'])

                    if eval_loss < training_state['best_loss'][cur_task]:
                        logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                        logger.info("  " + "*" * 20)
                        fa_dict[cur_task].write(
                            "[%d: %s] Best ppl changed into %.4f\n" % (training_state['global_step'], cur_task, np.exp(eval_loss)))
                        training_state['best_loss'][cur_task] = eval_loss

                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl', cur_task)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
                            output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
                            save_checkpoint(training_state, optimizer, scheduler, model_to_save, output_model_file,
                                            output_optimizer_file, output_scheduler_file, output_dir)
                            logger.info("Save the best ppl model into %s", output_model_file)

                if args.do_eval_bleu:
                    eval_examples_data_dict = load_and_cache_multi_gen_data(args, pool, tokenizer, 'dev',
                                                                            only_src=True, is_sample=True)
                    for cur_task in eval_examples_data_dict.keys():
                        if training_state['is_early_stop'][cur_task]:
                            continue
                        eval_examples, eval_data = eval_examples_data_dict[cur_task]

                        # pdb.set_trace()
                        result = eval_bleu(args, eval_data, eval_examples, model, tokenizer, 'dev', cur_task,
                                           criteria='e{}'.format(training_state['global_step']))
                        dev_bleu, dev_em = result['bleu'], result['em']
                        if args.task == 'summarize':
                            dev_bleu_em = dev_bleu
                        elif args.task in ['defect', 'clone']:
                            dev_bleu_em = dev_em
                        else:
                            dev_bleu_em = dev_bleu + dev_em
                        if args.data_num == -1:
                            training_state['bleu_em'][cur_task].append({'step': training_state['global_step'], 'bleu_em': dev_bleu_em})
                            tb_writer.add_scalar('dev_bleu_em_{}'.format(cur_task), dev_bleu_em, training_state['global_step'])

                        if dev_bleu_em > training_state['best_bleu_em'][cur_task]:
                            training_state['not_bleu_em_inc_cnt'][cur_task] = 0
                            logger.info("  [%d: %s] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                        training_state['global_step'], cur_task, dev_bleu_em, dev_bleu, dev_em)
                            logger.info("  " + "*" * 20)
                            training_state['best_bleu_em'][cur_task] = dev_bleu_em
                            fa_dict[cur_task].write(
                                "[%d: %s] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                                    training_state['global_step'], cur_task, training_state['best_bleu_em'][cur_task], dev_bleu, dev_em))
                            # Save best checkpoint for best bleu
                            output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu', cur_task)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            if args.data_num == -1 or args.always_save_model:
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                                output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
                                output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
                                save_checkpoint(training_state, optimizer, scheduler, model_to_save,
                                                output_model_file,
                                                output_optimizer_file, output_scheduler_file, output_dir)
                                logger.info("Save the best bleu model into %s", output_model_file)
                        else:
                            training_state['not_bleu_em_inc_cnt'][cur_task] += 1
                            logger.info("[%d %s] bleu/em does not increase for %d eval steps",
                                        training_state['global_step'], cur_task, training_state['not_bleu_em_inc_cnt'][cur_task])
                            if training_state['not_bleu_em_inc_cnt'][cur_task] > patience_dict[cur_task]:
                                logger.info("[%d %s] Early stop as bleu/em does not increase for %d eval steps",
                                            training_state['global_step'], cur_task, training_state['not_bleu_em_inc_cnt'][cur_task])
                                training_state['is_early_stop'][cur_task] = 1
                                fa_dict[cur_task].write(
                                    "[%d %s] Early stop as bleu/em does not increase for %d eval steps, takes %s" %
                                    (training_state['global_step'], cur_task, training_state['not_bleu_em_inc_cnt'][cur_task], get_elapse_time(t0)))

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
            if training_state['global_step'] >= args.max_steps:
                logger.info("Reach the max step: %d", args.max_steps)
                break

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %.2f", time.time() - t0)
        for cur_task in all_tasks:
            fa_dict[cur_task].close()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_examples_data_dict = load_and_cache_multi_gen_data(args, pool, tokenizer, 'test', only_src=True)
        all_tasks = list(eval_examples_data_dict.keys())
        for cur_task in all_tasks:
            summary_dir = os.path.join(args.output_dir, 'summary')
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            fa_dict[cur_task] = open(os.path.join(summary_dir, '{}_summary.log'.format(cur_task)), 'a+')

        for cur_task in all_tasks:
            eval_examples, eval_data = eval_examples_data_dict[cur_task]
            args.task = cur_task.split('_')[0]
            args.sub_task = cur_task.split('_')[-1]

            for criteria in ['best-bleu', 'best-ppl', 'last']:
                if criteria == 'last':
                    file = os.path.join(args.output_dir, 'checkpoint-last/pytorch_model.bin')
                else:
                    file = os.path.join(args.output_dir,
                                        'checkpoint-{}/{}/pytorch_model.bin'.format(criteria, cur_task))
                if args.n_gpu > 1:
                    model.module.load_state_dict(torch.load(file))
                else:
                    model.load_state_dict(torch.load(file))

                result = eval_bleu(args, eval_data, eval_examples, model, tokenizer, 'test', cur_task, criteria)
                test_bleu, test_em = result['bleu'], result['em']
                test_codebleu = result['codebleu'] if 'codebleu' in result else 0
                result_str = "[%s %s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (
                    cur_task, criteria, test_bleu, test_em, test_codebleu)
                logger.info(result_str)
                fa_dict[cur_task].write(result_str)
                fa.write(result_str)
                if args.res_fn:
                    with open(args.res_fn, 'a+') as f:
                        f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                        f.write(result_str)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    for cur_task in all_tasks:
        fa_dict[cur_task].close()
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
