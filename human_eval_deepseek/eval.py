"""
DeepSeek-V3 IER prompt construction for the LM-CC replication.

Mirrors CodeMind's `create_prompt_gpt_codeqwen` template exactly (same
instruction, fixed one-shot example, format_requirement, dataset branching,
and the `$?$` quirk). The whole template is sent as a SINGLE user message,
which is the closest faithful port of CodeMind's single-string prompt to the
DeepSeek chat API.

Did not write this myself, its to validate the lm cc and compare it to the original paper.
"""

# ---------------------------------------------------------------------------
# Shared prompt components (verbatim from CodeMind's create_prompt_ier.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DeepSeek-V3 prompt builder (GPT/CodeQwen-style base, single string)
# ---------------------------------------------------------------------------

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

    if pl == 'Java':
        prompt_instruction = instruction.format(language='Java')
        if dataset in ['CodeNet', 'Avatar']:
            prompt = template.replace("$PROMPT_INSTRUCTION$", prompt_instruction) \
                .replace("$EXAMPLE_CODE$", example_java) \
                .replace("$QUESTION$", question_print_output) \
                .replace("$EXAMPLE_INPUT$", example_input) \
                .replace("$EXAMPLE_REASONING$", example_reasoning_java) \
                .replace("$?$", '')

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


# ---------------------------------------------------------------------------
# HumanEval -> (code, code_input, expected_output) helper
# ---------------------------------------------------------------------------

import ast
import re


def humaneval_to_triple(problem):
    """Turn one HumanEval problem record into an IER triple.

    Expects a dict with keys 'prompt', 'canonical_solution', 'entry_point',
    'test' (the standard human_eval read_problems() format).

    Strategy:
      - code  = prompt + canonical_solution, with comments/docstrings removed
                (LM-CC normalizes by stripping comments/docstrings).
      - code_input = "<entry_point>(<args>)" pulled from the FIRST assert in
                     the test harness.
      - expected_output = the literal the first assert compares against.

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
    """Extract (call_expr_string, expected_literal_string) from first assert.

    Handles the common HumanEval shapes:
        assert candidate(<args>) == <value>
        assert func(<args>) == <value>
    where the test wraps the function as `candidate = <entry_point>`.
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
            # left should be a call to candidate/entry_point
            if isinstance(left, ast.Call):
                call_str = ast.unparse(left)
                # normalize the callee name to the real entry_point
                call_str = re.sub(r"^\s*candidate\b", entry_point, call_str)
                expected_str = ast.unparse(right)
                return call_str, expected_str
    return None, None


# ---------------------------------------------------------------------------
# DeepSeek API call wrapper
# ---------------------------------------------------------------------------

def query_deepseek(prompt, model="deepseek-chat", temperature=0.0,
                   stop=("[END-OF-RESPONSE]",), max_tokens=2048):
    """Send the single-string prompt as one user message to DeepSeek.

    Requires DEEPSEEK_API_KEY in env. `deepseek-chat` is the V3.x chat model;
    verify the current alias against DeepSeek's docs before a final run.
    Temperature 0.0 matches CodeMind's determinism choice for non-GPT models.
    """
    import os
    from openai import OpenAI

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


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from human_eval.data import read_problems  # pip install human-eval

    problems = read_problems()
    correct = 0
    total = 0
    for task_id, problem in problems.items():
        triple = humaneval_to_triple(problem)
        if triple is None:
            continue
        code, code_input, expected = triple
        prompt = create_prompt_deepseek(code, code_input,
                                        dataset="humaneval", pl="Python")
        response = query_deepseek(prompt)         # temp=0.0 by default
        predicted = parse_output(response)
        # Compare by normalized literal; eval both sides for value equality.
        try:
            ok = eval(predicted) == eval(expected)
        except Exception:
            ok = (predicted == expected)
        correct += int(ok)
        total += 1
        print(f"{task_id}: pred={predicted!r} exp={expected!r} {'OK' if ok else 'X'}")

    print(f"\npass@1 (RIER) = {correct}/{total} = {correct/total:.4f}")