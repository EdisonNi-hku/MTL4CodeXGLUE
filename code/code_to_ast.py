from tree_sitter import Language, Parser
import os
import json
import random
import math
import torch
import logging
import multiprocessing
from transformers import RobertaTokenizer, T5Tokenizer
from collections import OrderedDict
from utils import get_filenames, get_src_lang_from_task, convert_src_tgt_to_features
from tqdm import tqdm
from argparse import ArgumentParser
from evaluator.CodeBLEU.parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript, DFG_csharp
from evaluator.CodeBLEU.parser import (remove_comments_and_docstrings,
                                       tree_to_token_index,
                                       index_to_code_token,
                                       tree_to_variable_index)

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript,
    'c_sharp': DFG_csharp,
}

root_dir = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

codet5_tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base', cache_dir='cache', local_files_only=True)
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base', cache_dir='cache', local_files_only=True)


def get_data_flow(code, parser, lang):
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        codes = code_tokens
        dfg = new_DFG
    except:
        codes = code.split()
        dfg = []
    # merge nodes
    dic = {}
    for d in dfg:
        if d[1] not in dic:
            dic[d[1]] = d
        else:
            dic[d[1]] = (d[0], d[1], d[2], list(set(dic[d[1]][3] + d[3])), sorted(list(set(dic[d[1]][4] + d[4]))))
    DFG = []
    for d in dic:
        DFG.append(dic[d])
    dfg = DFG
    return dfg, codes


def load_code(data_root, task, subtask):
    if task == 'summarize':
        return load_summarization_code(data_root, subtask)
    elif task == 'translate':
        return load_translation_code(data_root, subtask)
    elif task == 'refine':
        return load_refine_code(data_root, subtask)
    elif task == 'defect':
        return load_defect_code(data_root, subtask)
    else:
        return load_clone_code(data_root, subtask)


def load_summarization_code(data_root, subtask):
    filename = get_filenames(data_root, 'summarize', subtask, 'train')
    code_strings = []
    with open(filename, encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f), desc='summarize_'+subtask):
            line = line.strip()
            js = json.loads(line)
            original_code = js['code']
            try:
                code = remove_comments_and_docstrings(original_code, subtask)
            except:
                pass
            code_strings.append(code)
    return code_strings


def load_translation_code(data_root, subtask):
    src_lang = subtask.split('-')[0]
    tgt_lang = subtask.split('-')[1]
    filename = get_filenames(data_root, 'translate', subtask, 'train')
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    src_code_strings = []
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in tqdm(zip(f1, f2), desc='translate_'+subtask):
            src = line1.strip()
            try:
                src = remove_comments_and_docstrings(src, src_lang)
            except:
                pass
            src_code_strings.append(src)
    return src_code_strings


def load_refine_code(data_root, subtask):
    filename = get_filenames(data_root, 'refine', subtask, 'train')
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    src_code_strings = []
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in tqdm(zip(f1, f2), desc='refine_'+subtask):
            src = line1.strip()
            try:
                src = remove_comments_and_docstrings(src, 'java')
            except:
                pass
            src_code_strings.append(src)
    return src_code_strings


def load_defect_code(data_root, subtask):
    filename = get_filenames(data_root, 'defect', subtask, 'train')
    code_strings = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), desc='defect'):
            line = line.strip()
            js = json.loads(line)
            code = js['func']
            try:
                code = remove_comments_and_docstrings(code, 'c')
            except:
                pass
            code_strings.append(code)
    return code_strings


def load_clone_code(data_root, subtask):
    filename = get_filenames(data_root, 'clone', subtask, 'train')
    code_strings = []
    with open('/'.join(filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in tqdm(f, desc='clone'):
            line = line.strip()
            js = json.loads(line)
            code = js['func']
            try:
                code = remove_comments_and_docstrings(code, 'java')
            except:
                pass
            code_strings.append(code)
    return code_strings


def find_identifiers(code, code_lines, node):
    identifiers = []
    if node.type == 'identifier':
        assert node.start_point[0] == node.end_point[0]
        start = sum([len(code_lines[i]) + 1 for i in range(node.start_point[0])]) + node.start_point[1]
        end = sum([len(code_lines[i]) + 1 for i in range(node.end_point[0])]) + node.end_point[1]
        name = code[start:end]
        identifiers.append([name, start, end])
        return identifiers
    elif len(node.children) == 0:
        return identifiers
    else:
        for child in node.children:
            identifiers.extend(find_identifiers(code, code_lines, child))
        return identifiers


def mask_identifiers(code_string, lang, percentage=0.3):
    parser = PARSERS[lang][0]
    tree = parser.parse(bytes(code_string, "utf8"))
    code_lines = code_string.split('\n')
    root_node = tree.root_node
    identifiers = find_identifiers(code_string, code_lines, root_node)
    names = set([t[0] for t in identifiers])
    to_mask = random.sample(names, math.ceil(percentage * len(names)))
    identifiers = [t for t in identifiers if t[0] in to_mask]
    identifiers = sorted(identifiers, key=lambda t: t[1])
    replace_dict = OrderedDict()
    idx = 0
    for t in identifiers:
        if t[0] in replace_dict.keys():
            replace_dict[t[0]]['count'] += 1
        else:
            replace_dict[t[0]] = {'new_name': "<extra_id_" + str(idx) + ">", 'count': 1, 'len_diff': len("<extra_id_" + str(idx) + ">") - len(t[0])}
            idx += 1

    for i in range(len(identifiers)):
        code_string = code_string[:identifiers[i][1]] + replace_dict[identifiers[i][0]]['new_name'] + code_string[identifiers[i][2]:]
        for j in range(i + 1, len(identifiers)):
            identifiers[j][1] += replace_dict[identifiers[i][0]]['len_diff']
            identifiers[j][2] += replace_dict[identifiers[i][0]]['len_diff']

    tgt_string = ""
    for k, v in replace_dict.items():
        tgt_string += v['new_name'] + ' ' + k + ' '

    return code_string, tgt_string.strip()


class IdentifierCollator(object):
    def __init__(self, args, cur_task, tokenizer, percentage=0.3):
        self.args = args
        self.cur_task = cur_task
        self.tokenizer = tokenizer
        self.percentage = percentage

    def __call__(self, batch):
        task = self.cur_task.split('_')[0]
        sub_task = self.cur_task.split('_')[-1]
        if sub_task == 'sharp':
            sub_task = 'c_sharp'
        lang = get_src_lang_from_task(task, sub_task)
        codes = batch
        masked_codes = []
        targets = []
        for code in codes:
            masked_code, tgt = mask_identifiers(code, lang, self.percentage)
            masked_codes.append(masked_code)
            targets.append(tgt)
        items = [(masked_code, tgt, idx, self.tokenizer, self.args, task, sub_task) for idx, (masked_code, tgt) in
                 enumerate(zip(masked_codes, targets))]
        features = []
        for item in items:
            features.append(convert_src_tgt_to_features(item))
        source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        return source_ids, target_ids


def get_parser(lang_dir, lang):
    lang = Language(lang_dir, lang)
    parser = Parser()
    parser.set_language(lang)
    return parser


LANGUAGES = ['python', 'java', 'c_sharp', 'javascript', 'php', 'ruby', 'go']
LANG_DIR = root_dir + '/evaluator/CodeBLEU/parser/my-languages.so'
PARSERS = {}
for lang in LANGUAGES:
    PARSERS[lang] = (get_parser(LANG_DIR, lang), dfg_function[lang])


def code_to_ast_string(item):
    code_string, lang = item
    tree = PARSERS[lang][0].parse(bytes(code_string, "utf8"))
    root_node = tree.root_node

    def ast2string(root):
        seq = ""
        name = root.type
        if len(root.children) == 0:

            seq += name
        else:
            seq += name + "{"
            for child in root.children:
                seq += ast2string(child) + '|'
            seq += name + "}"
        return seq

    return ast2string(root_node)


def is_constant(element) -> bool:
    if element.startswith("\"") and element.endswith("\""):
        return True
    elif element.startswith("\'") and element.endswith("\'"):
        return True
    else:
        try:
            float(element)
            return True
        except ValueError:
            return False


def code_to_dataflow_string(item):
    code_string, lang = item
    parser = PARSERS[lang]
    dfg, code_tokens = get_data_flow(code_string, parser, lang)
    var_order = {}
    id2var = {}
    for t in dfg:
        id2var[t[1]] = t[0]
        if is_constant(t[0]):
            continue
        else:
            if t[0] in var_order.keys():
                if t[1] not in var_order[t[0]].keys():
                    count = len(var_order[t[0]].keys())
                    var_order[t[0]][t[1]] = count
            else:
                var_order[t[0]] = {}
                var_order[t[0]][t[1]] = 0
    dataflow = ""
    for t in dfg:
        if is_constant(t[0]):
            continue
        else:
            dataflow += t[0] + str(var_order[t[0]][t[1]]) + ' '
            if t[2] == 'comesFrom':
                dataflow += 'come '
            else:
                dataflow += 'compute '
            for i in range(len(t[4])):
                if t[4][i] not in id2var.keys():
                    dataflow += code_tokens[t[4][i]] + ' '
                elif is_constant(id2var[t[4][i]]):
                    dataflow += id2var[t[4][i]] + ' '
                else:
                    dataflow += id2var[t[4][i]] + str(var_order[id2var[t[4][i]]][t[4][i]]) + ' '
            dataflow = dataflow[:-1] + ','
    return dataflow


def count_tokens(item):
    s, tokenizer = item
    s = s.replace('</s>', '<unk>')
    s_ids = tokenizer.encode(s)
    return len(s_ids)


def code2ast(code_dict):
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    count = {}
    for k, code_list in code_dict.items():
        task = k.split('_')[0]
        subtask = k.split('_')[1]
        if task == 'summarize':
            lang = subtask
        elif task == 'translate':
            lang = subtask.split('-')[0]
            if lang == 'cs':
                lang = 'c_sharp'
        elif task == 'defect':
            lang = 'c'
        else:
            lang = 'java'
        items = [(code, lang) for code in code_list]
        ast_list = pool.map(code_to_ast_string, tqdm(items, total=len(items), desc='parsing ' + task + '_' + subtask))
        with open('ast/' + '{}_{}'.format(task, subtask) + '.code', 'w') as f1, open('ast/' + '{}_{}'.format(task, subtask) + '.ast', 'w') as f2:
            f1.writelines(code_list)
            f2.writelines(ast_list)
        t5_ast_items = [(ast, t5_tokenizer) for ast in ast_list]
        codet5_ast_items = [(ast, codet5_tokenizer) for ast in ast_list]
        t5_code_items = [(code, t5_tokenizer) for code in code_list]
        codet5_code_items = [(code, codet5_tokenizer) for code in code_list]
        t5_code_count = pool.map(count_tokens, tqdm(t5_code_items, total=len(t5_code_items), desc='t5 code'))
        codet5_code_count = pool.map(count_tokens,
                                     tqdm(codet5_code_items, total=len(codet5_code_items), desc='codet5 code'))
        t5_ast_count = pool.map(count_tokens, tqdm(t5_ast_items, total=len(t5_ast_items), desc='t5 ast'))
        codet5_ast_count = pool.map(count_tokens,
                                     tqdm(codet5_ast_items, total=len(codet5_ast_items), desc='codet5 ast'))
        count[k] = (sum(t5_code_count)/len(t5_code_count),
                    sum(codet5_code_count) / len(codet5_code_count),
                    sum(t5_ast_count) / len(t5_ast_count),
                    sum(codet5_ast_count) / len(codet5_ast_count))
    print(count)
    torch.save(count, 'token_count')
    pool.close()


def code2df(code_dict, args):
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    count = {}
    for k, code_list in code_dict.items():
        task = k.split('_')[0]
        subtask = k.split('_')[1]
        if task == 'summarize':
            lang = subtask
        elif task == 'translate':
            lang = subtask.split('-')[0]
            if lang == 'cs':
                lang = 'c_sharp'
        elif task == 'defect':
            lang = 'c'
        else:
            lang = 'java'
        items = [(code, lang) for code in code_list]
        df_list = pool.map(code_to_dataflow_string, tqdm(items, total=len(items), desc='parsing ' + task + '_' + subtask))
        all_data = [{'code': code, 'dataflow': df} for code, df in zip(code_list, df_list)]
        with open(args.save_dir + '/{}_{}'.format(task, subtask) + '.jsonl', 'w') as f:
            for d in all_data:
                json.dump(d, f)
                f.write('\n')
        t5_df_items = [(df, t5_tokenizer) for df in df_list]
        codet5_df_items = [(df, codet5_tokenizer) for df in df_list]
        t5_code_items = [(code, t5_tokenizer) for code in code_list]
        codet5_code_items = [(code, codet5_tokenizer) for code in code_list]
        t5_code_count = pool.map(count_tokens, tqdm(t5_code_items, total=len(t5_code_items), desc='t5 code'))
        codet5_code_count = pool.map(count_tokens,
                                     tqdm(codet5_code_items, total=len(codet5_code_items), desc='codet5 code'))
        t5_df_count = pool.map(count_tokens, tqdm(t5_df_items, total=len(t5_df_items), desc='t5 dataflow'))
        codet5_df_count = pool.map(count_tokens,
                                    tqdm(codet5_df_items, total=len(codet5_df_items), desc='codet5 dataflow'))
        count[k] = (sum(t5_code_count) / len(t5_code_count),
                    sum(codet5_code_count) / len(codet5_code_count),
                    sum(t5_df_count) / len(t5_df_count),
                    sum(codet5_df_count) / len(codet5_df_count))
        filter_code = []
        filter_df = []
        for i in range(len(t5_code_count)):
            if t5_df_count[i] <= 500 and t5_code_count[i] <= 500 and \
                    codet5_df_count[i] <= 500 and codet5_code_count[i] < 500:
                filter_code.append(code_list[i])
                filter_df.append(df_list[i])
        filtered_data = [{'code': code, 'dataflow': df} for code, df in zip(filter_code, filter_df)]
        with open(args.save_dir + '/{}_{}'.format(task, subtask) + '.filtered.jsonl', 'w') as f:
            for d in filtered_data:
                json.dump(d, f)
                f.write('\n')

    print(count)
    torch.save(count, 'token_count_df')
    pool.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--cache_file", type=str, default='code_cache')
    parser.add_argument("--save_dir", type=str, default='df_10')
    parser.add_argument("--data_root", type=str, default='data')
    args = parser.parse_args()

    os.mkdir(args.save_dir)
    random.seed(1234)
    cache_fn = root_dir + '/' + args.cache_file
    if os.path.exists(cache_fn):
        code_dict = torch.load(cache_fn)
    else:
        code_dict = {}
        tasks = ['summarize', 'refine', 'translate']
        for cur_task in tasks:
            if cur_task == 'summarize':
                subtasks = ['python', 'java', 'javascript', 'php', 'go', 'ruby']
            elif cur_task == 'translate':
                subtasks = ['cs-java', 'java-cs']
            elif cur_task == 'refine':
                subtasks = ['small', 'medium']
            else:
                subtasks = ['none']
            for sub in subtasks:
                code = load_code(data_root=args.data_root, task=cur_task, subtask=sub)
                code_dict['{}_{}'.format(cur_task, sub)] = code

        torch.save(code_dict, cache_fn)

    code2df(code_dict, args)


if __name__ == '__main__':
    main()
    # code_string = "public void addAll(BlockList<T> src) {if (src.size == 0)return;int srcDirIdx = 0;for (; srcDirIdx < src.tailDirIdx; srcDirIdx++)addAll(src.directory[srcDirIdx], 0, BLOCK_SIZE);if (src.tailBlkIdx != 0)addAll(src.tailBlock, 0, src.tailBlkIdx);}"
    # parser = PARSERS['java'][0]
    # tree = parser.parse(bytes(code_string, "utf8"))
    # code_lines = code_string.split('\n')
    # root_node = tree.root_node
    # identifiers = find_identifiers(code_string, code_lines, root_node)
    # print(identifiers)
    # print(code_string[])



