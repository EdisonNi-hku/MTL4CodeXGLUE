from tree_sitter import Language, Parser
from evaluator.CodeBLEU.parser import remove_comments_and_docstrings
import os

root_dir = os.path.dirname(__file__)


def code_to_ast_string(code, lang):
    LANGUAGE = Language(root_dir + 'evaluator/CodeBLEU/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)

