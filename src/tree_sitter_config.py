"""Tree-sitter Python parser used to build the semantic block tree.
"""

import tree_sitter_python
from tree_sitter import Language, Parser

PYTHON_LANGUAGE = Language(tree_sitter_python.language())

py_parser = Parser()
py_parser.language = PYTHON_LANGUAGE
