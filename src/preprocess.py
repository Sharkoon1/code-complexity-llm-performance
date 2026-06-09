"""Source-code normalization shared by all complexity metrics.
"""

import ast
import logging
import os
import subprocess
import tempfile
import uuid
from textwrap import dedent

logger = logging.getLogger(__name__)


def _remove_comments_and_docstrings(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # leave un-parseable code to black (which will skip it)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
                if not body and not isinstance(node, ast.Module):
                    body.append(ast.Pass())

    return ast.unparse(ast.fix_missing_locations(tree))


def _format_python_code(code: str):
    # strip decorators
    if "@" in code:
        code = '\n'.join(line for line in code.split('\n') if not (line.strip().startswith("@")))
    # normalize indentation
    code = dedent(code)
    random_filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.py")
    try:
        with open(random_filename, 'w') as f:
            f.write(code)
        # run black formatting
        try:
            subprocess.run(
                ["black", "--quiet", "--fast", random_filename],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            try:
                subprocess.run(
                    ["black", "--quiet", random_filename],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                logger.warning("black could not format the code: %s", e.stderr)
                return None
        with open(random_filename, 'r') as f:
            formatted_code = f.read()
        return formatted_code
    except Exception as e:
        logger.error("Unexpected error while formatting code: %s", e)
        return None
    finally:
        os.remove(random_filename)


def normalize(code: str):
    """Shared preprocessing: strip comments/docstrings, then black-format.

    Returns the normalized source.
    """
    code = _remove_comments_and_docstrings(code)
    code = _format_python_code(code)
    if not code or not code.strip():
        return None
    return code
