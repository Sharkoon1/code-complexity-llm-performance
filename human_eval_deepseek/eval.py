"""
DeepSeek-V3 IER prompt construction for the LM-CC replication.

Mirrors CodeMind's `create_prompt_gpt_codeqwen` template exactly (same
instruction, fixed one-shot example, format_requirement, dataset branching,
and the `$?$` quirk). The whole template is sent as a SINGLE user message,
which is the closest faithful port of CodeMind's single-string prompt to the
DeepSeek chat API.

Extension of CodeMind's create_prompt_gpt_codeqwen to validate LM-CC scores
against the original paper.
"""

import ast
import os
import re

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


example_python_function = '''
def sum_of_integer(N, A, B):
    sum_1 = 0
    for i in range(1,N+1):
        sum_order = 0
        i_str = str(i)
        n = len(i_str)
        for j in range(0,n):
            sum_order += int(i_str[j])
        if A <= sum_order <= B:
            sum_1 += i
    return sum_1
'''

example_python = '''
N, A, B = map(int, input().split())
sum_1 = 0
for i in range(1,N+1):
    sum_order = 0
    i_str = str(i)
    n = len(i_str)
    for j in range(0,n):
        sum_order += int(i_str[j])
    if A <= sum_order <= B:
        sum_1 += i
print(sum_1)
'''

example_python_cruxeval = '''
def f(s):
    return s + "a"
'''

format_requirement = """
First analyze step by step about how the code processes the input to generate the output. 
Then print the output of the code based on your analysis.

Follow the following format:
<<<Analysis>>>
{YOUR ANALYSIS}
<<<Output>>>
{OUTPUT}
[END-OF-RESPONSE]
"""

example_reasoning_python = """
<<<Analysis>>>
The variable N, variable A and variable B are initialized to 20, 2 and 5, respectively.
variable sum_1 is initialized to 0, which will be used to accumulate the sum of numbers meeting the condition.
The code then enters a loop that iterates from 1 to N (inclusive), meaning it will consider numbers from 1 to 20.
For each number i in this range, it calculates the sum of its digits and stores it in sum_order.
The code checks if sum_order is within the range [A, B], which is [2, 5] in this case. If it is, it adds the current number i to sum_1. The condition is met when i is 2,3,4,5,11,12,13,14 and 20.
After the loop finishes, the code prints the final value of sum_1, which is 84.
<<<Output>>>
84
[END-OF-RESPONSE]
"""

example_reasoning_python_cruxeval = """
The function f takes a string s as input and returns the concatenation of s with the string "a".
To determine the output of executing the function f on the input "hi", we need to concatenate "hi" with "a".
Therefore, the output of executing the function f on the input "hi" is "hia".
<<<Output>>>
'hia'
[END-OF-RESPONSE]
"""

example_input = "20 2 5"
example_input_function = "sum_of_integer(20, 2, 5)"
example_input_cruxeval = 'f("hi")'

question_print_output = "What would be the output of code execution given the following input:\n`"
question_return_value = "What would be the return value of "

instruction = """I want you to act as a {language} code executor. I will give you a piece of {language} code and its input. You need to think step by step and then print the output of code execution."""


def create_prompt_deepseek(code, code_input, dataset, pl):
    """Build the IER prompt string for DeepSeek-V3.

    Args:
        code:       the program source (string)
        code_input: the input expression. For humaneval/mbpp/cruxeval this is a
                    call expression like "func(arg1, arg2)"; for CodeNet/Avatar
                    it is the raw stdin string.
        dataset:    one of {'humaneval','mbpp','cruxeval','CodeNet','Avatar'}
        pl:         'Python' or 'Java'
    Returns:
        A single prompt string to send as one user message.
    """
    template = (
        "<Instruction> " + "$PROMPT_INSTRUCTION$\n" + "</Instruction>\n"
        + "Below is an example:\n<Example>\nConsider the following code\n" + "$EXAMPLE_CODE$"
        + "\n[Question]\n" + "$QUESTION$" + "```" + "$EXAMPLE_INPUT$" + "```$?$" + format_requirement
        + "[Answer]\n" + "$EXAMPLE_REASONING$" + "\n</Example>\n" + "Consider the following code\n" + code
        + "\n" + "[Question]\n" + "$QUESTION$" + "```" + code_input + "```$?$\n" + format_requirement
        + "\n[Answer]\n"
    )

    prompt = template  # fallback

    if pl == 'Python':
        prompt_instruction = instruction.format(language='Python')
        if dataset in ['CodeNet', 'Avatar']:
            prompt = template.replace("$PROMPT_INSTRUCTION$", prompt_instruction) \
                .replace("$EXAMPLE_CODE$", example_python) \
                .replace("$QUESTION$", question_print_output) \
                .replace("$EXAMPLE_INPUT$", example_input) \
                .replace("$EXAMPLE_REASONING$", example_reasoning_python) \
                .replace("$?$", '')
        if dataset in ["mbpp", "humaneval"]:
            prompt = template.replace("$PROMPT_INSTRUCTION$", prompt_instruction) \
                .replace("$EXAMPLE_CODE$", example_python_function) \
                .replace("$QUESTION$", question_return_value) \
                .replace("$EXAMPLE_INPUT$", example_input_function) \
                .replace("$EXAMPLE_REASONING$", example_reasoning_python) \
                .replace("$?$", "?")
        if dataset in ["cruxeval"]:
            prompt = template.replace("$PROMPT_INSTRUCTION$", prompt_instruction) \
                .replace("$EXAMPLE_CODE$", example_python_cruxeval) \
                .replace("$QUESTION$", question_return_value) \
                .replace("$EXAMPLE_INPUT$", example_input_cruxeval) \
                .replace("$EXAMPLE_REASONING$", example_reasoning_python_cruxeval) \
                .replace("$?$", "?")
    return prompt

# Sentinel for expected values that couldn't be reduced to a Python literal
# (e.g. `tuple(sort_third([1, 2, 3]))` on the RHS of an assert).
class _NonLiteralExpected(str):
    """A string subclass tagging unresolved expected expressions."""
    __slots__ = ()


def humaneval_to_triple(problem):
    """Turn one HumanEval problem record into an IER triple.

    Expects a dict with keys 'prompt', 'canonical_solution', 'entry_point',
    'test' (the standard human_eval read_problems() format).

    Strategy:
      - code  = prompt + canonical_solution, with comments/docstrings removed
                (LM-CC normalizes by stripping comments/docstrings).
      - code_input = "<entry_point>(<args>)" pulled from the FIRST assert in
                     the test harness.
      - expected_output = the actual Python value the first assert compares
                          against (e.g. the string "010010", not "'010010'"),
                          or a `_NonLiteralExpected` if the RHS is itself an
                          expression that can't be literal-eval'd.

    Returns (code, code_input, expected_output) or None if no assert was found.
    """
    full_code = problem["prompt"] + problem["canonical_solution"]
    code = _strip_comments_and_docstrings(full_code)

    entry = problem["entry_point"]
    call, expected = _first_assert_call(problem["test"], entry)
    if call is None:
        return None
    return code, call, expected


def _strip_comments_and_docstrings(src):
    """Remove docstrings and # comments while keeping code runnable-looking."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src  # leave as-is if it won't parse

    # Drop docstrings (first stmt that is a bare string) from module/func/class.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and \
               isinstance(getattr(body[0], "value", None), ast.Constant) and \
               isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    try:
        out = ast.unparse(tree)  # Python 3.9+
    except Exception:
        out = src
    # ast.unparse already drops # comments. Tidy stray blank lines.
    return re.sub(r"\n\s*\n\s*\n", "\n\n", out).strip()


def _first_assert_call(test_src, entry_point):
    """Extract (call_expr_string, expected_value) from the first assert.

    expected_value is the actual Python value (str, int, list, ...) when the
    RHS of the assert is a literal. If the RHS is an expression that can't be
    parsed as a literal, we return its source as a `_NonLiteralExpected` so
    the caller can detect and skip / handle it.
    """
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return None, None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and \
           isinstance(test.ops[0], ast.Eq):
            left, right = test.left, test.comparators[0]
            if isinstance(left, ast.Call):
                call_str = ast.unparse(left)
                # normalize the callee name to the real entry_point
                call_str = re.sub(r"^\s*candidate\b", entry_point, call_str)
                try:
                    expected_value = ast.literal_eval(right)
                except (ValueError, SyntaxError):
                    expected_value = _NonLiteralExpected(ast.unparse(right))
                return call_str, expected_value
    return None, None


def query_deepseek(prompt, model="deepseek-chat", temperature=0.0,
                   stop=("[END-OF-RESPONSE]",), max_tokens=2048):
    """Send the single-string prompt as one user message to DeepSeek.

    Requires DEEPSEEK_API_KEY in env. `deepseek-chat` is the V3.x chat model;
    verify the current alias against DeepSeek's docs before a final run.
    Temperature 0.0 matches CodeMind's determinism choice for non-GPT models.
    """
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=list(stop) if stop else None,
    )
    return resp.choices[0].message.content


def parse_output(response):
    """Pull the predicted output between <<<Output>>> and [END-OF-RESPONSE]."""
    m = re.search(r"<<<Output>>>\s*(.*?)\s*(?:\[END-OF-RESPONSE\]|$)",
                  response, re.DOTALL)
    return m.group(1).strip() if m else None


def _coerce_predicted(predicted_str, expected_value):
    """Parse model output into a Python value comparable to expected_value.

    Strategy:
      1. Try `ast.literal_eval` (handles numbers, strings, bools, None,
         lists, tuples, dicts, sets of literals).
      2. If that fails and the expected value is a string, treat the raw
         output as a string (the model often omits quotes around plain
         tokens like `010010` or `db0db`).
      3. Otherwise return the raw stripped string so the caller still gets
         a definite-but-mismatched value rather than a crash.

    Never uses `eval` on model output.
    """
    if predicted_str is None:
        return None
    s = predicted_str.strip()
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    if isinstance(expected_value, str) and not isinstance(expected_value, _NonLiteralExpected):
        # Strip any stray surrounding quotes the model may have added.
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            return s[1:-1]
        return s
    return s


def compare_predicted(predicted_str, expected_value):
    """Return (ok, predicted_value_or_str, reason).

    `reason` is one of: 'value', 'string', 'unresolved', 'no_output'.
    """
    if predicted_str is None:
        return False, None, 'no_output'
    if isinstance(expected_value, _NonLiteralExpected):
        # Can't safely compare — RHS of the assert was an expression, not a
        # literal. Fall back to a string-level match against its source.
        return predicted_str.strip() == str(expected_value).strip(), predicted_str, 'unresolved'
    predicted_value = _coerce_predicted(predicted_str, expected_value)
    return predicted_value == expected_value, predicted_value, 'value'