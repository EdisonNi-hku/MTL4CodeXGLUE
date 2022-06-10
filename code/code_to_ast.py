from tree_sitter import Language, Parser
import os
import json

from utils import get_filenames

root_dir = os.path.dirname(__file__)


def load_summarization_code(data_root, subtask):
    filename = get_filenames(data_root, 'summarize', subtask, 'train')
    code_strings = []
    with open(filename, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())


def code_to_ast_string(code, lang):
    LANGUAGE = Language(root_dir + 'evaluator/CodeBLEU/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code))

    root_node = tree.root_node


if __name__ == '__main__':
    LANGUAGE = Language('code/evaluator/CodeBLEU/parser/my-languages.so', 'python')
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes("""
    def foo():
        if bar:
            baz()
    """, "utf8"))
    root_node = tree.root_node
    print(root_node.sexp())
    tree1 = parser.parse(bytes("def foo():    if bar:        baz()", "utf8"))
    print(tree1.root_node.sexp())

