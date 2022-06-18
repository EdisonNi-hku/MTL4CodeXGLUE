from tree_sitter import Language, Parser
import os
import json
import random
import math
from collections import OrderedDict
from utils import get_filenames
from evaluator.CodeBLEU.parser import remove_comments_and_docstrings
root_dir = os.path.dirname(__file__)


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
        for idx, line in enumerate(f):
            if idx >= 1:
                break
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
    tgt_code_strings = []
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            try:
                src = remove_comments_and_docstrings(src, src_lang)
                trg = remove_comments_and_docstrings(trg, tgt_lang)
            except:
                pass
            src_code_strings.append(src)
            tgt_code_strings.append(trg)
    return src_code_strings, tgt_code_strings


def load_refine_code(data_root, subtask):
    filename = get_filenames(data_root, 'refine', subtask, 'train')
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    src_code_strings = []
    tgt_code_strings = []
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            try:
                src = remove_comments_and_docstrings(src, 'java')
                trg = remove_comments_and_docstrings(trg, 'java')
            except:
                pass
        src_code_strings.append(src)
        tgt_code_strings.append(trg)
    return src_code_strings, tgt_code_strings


def load_defect_code(data_root, subtask):
    filename = get_filenames(data_root, 'defect', subtask, 'train')
    code_strings = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
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
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = js['func']
            try:
                code = remove_comments_and_docstrings(code, 'java')
            except:
                pass
            code_strings.append(code)
    return code_strings


def code_to_ast_string(code_string, parser):
    tree = parser.parse(bytes(code_string, "utf8"))
    root_node = tree.root_node

    def ast2string(root):
        seq = ""
        name = root.type
        if len(root.children) == 0:
            seq += name + ' '
        else:
            seq += name + "|left "
            for child in root.children:
                seq += ast2string(child) + ' '
            seq += name + "|right "
        return seq

    return ast2string(root_node)


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


def mask_identifiers(code_string, parser):
    tree = parser.parse(bytes(code_string, "utf8"))
    code_lines = code_string.split('\n')
    root_node = tree.root_node
    identifiers = find_identifiers(code_string, code_lines, root_node)
    names = set([t[0] for t in identifiers])
    to_mask = random.sample(names, math.ceil(0.3 * len(names)))
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


def get_parser(lang_dir, lang):
    lang = Language(lang_dir, lang)
    parser = Parser().set_language(lang)
    return parser


if __name__ == '__main__':
    languages = ['python', 'java', 'php', 'c', 'c_sharp', 'javascript', 'php', 'ruby']
    lang_dir = root_dir + '/evaluator/CodeBLEU/parser/my-languages.so'
    parsers = {}
    for lang in languages:
        parsers[lang] = get_parser(lang_dir, lang)


    # python_code = load_summarization_code('data', 'python')
    # java_code = load_summarization_code('data', 'java')
    # print(python_code[0])
    # print(code_to_ast_string(python_code[0], python_parser))
    # #print(mask_identifiers(java_code[0], java_parser))
    # masked_string, tgt = mask_identifiers(java_code[0], java_parser)
    # print(java_code[0])
    # print(masked_string)
    # print(tgt)



