"""Source-code normalization shared by all complexity metrics.
"""

import logging
import os
import re
import subprocess
import tempfile
import uuid
from textwrap import dedent

logger = logging.getLogger(__name__)


def _remove_comments_and_docstrings(code: str) -> str:
    string_placeholders = []
    string_pattern = r'''(
        \"\"\" (?:\\. | "(?!\"\"") | [^\\"])*? \"\"\" |
        \''' (?:\\. | '(?!'') | [^\\'])*? \'''     |
        " (?: \\. | [^"\\] )* "                    |
        ' (?: \\. | [^'\\] )* '
    )'''

    def replace_string(match):
        string_placeholders.append(match.group(1))
        return f'__STRING_{len(string_placeholders)-1}__'

    code = re.sub(string_pattern, replace_string, code, flags=re.DOTALL | re.VERBOSE)
    code = re.sub(r'#.*', '', code)
    for idx, content in enumerate(string_placeholders):
        code = code.replace(f'__STRING_{idx}__', content)

    code = re.sub(r'^(?:\s*)(["\']{3}.*?["\']{3})(?:\s*)$', '', code, flags=re.DOTALL | re.MULTILINE)
    patterns = [
        r'(def\s+\w+\(.*?\):\s*?\n)(\s*["\']{3}.*?["\']{3}\n)',
        r'(class\s+\w+.*?:\s*?\n)(\s*["\']{3}.*?["\']{3}\n)',
        r'(async\s+def\s+\w+\(.*?\):\s*?\n)(\s*["\']{3}.*?["\']{3}\n)'
    ]
    for pattern in patterns:
        code = re.sub(pattern, r'\1', code, flags=re.DOTALL)
    lines = [line for line in code.splitlines() if line.strip()]
    return '\n'.join(lines)


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
