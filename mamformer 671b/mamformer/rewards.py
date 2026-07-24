"""
GRPO Reward Functions
=====================
Rule-based reward calculators for Group Relative Policy Optimization.

Each reward function scores a model response on a specific dimension,
returning a float in [0, 1]. These are used during GRPO training to
compute group-relative advantages without needing a separate critic model.

Supported reward types:
  - "math": Extract and compare boxed answers (DeepSeek-R1 style)
  - "format": Check for think/answer XML tag structure
  - "code": Execute extracted code against test cases
  - "length": Penalize excessively long responses

Reference:
  "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL"
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Math Reward
# ═══════════════════════════════════════════════════════════════════════

def _extract_boxed(text: str) -> list[str]:
    """Extract content from \\boxed{...} with nested brace support."""
    results = []
    for m in re.finditer(r'\\boxed\{', text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i-1])
    return results


def _extract_numeric(s: str) -> Optional[float]:
    """
    Extract numeric value from string.

    Handles: integers, decimals, fractions (3/4), negative numbers,
    scientific notation, and LaTeX formatting.
    """
    if not s:
        return None

    # Remove LaTeX formatting
    s = s.strip()
    s = s.replace(r"\text", "").replace(r"\mathrm", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("$", "").replace("\\", "")
    s = s.replace(",", "")  # Remove thousands separators
    s = s.replace(" ", "")

    # Try fraction first: "3/4" or "-1/2"
    frac_match = re.match(r'^(-?\d+)\s*/\s*(-?\d+)$', s)
    if frac_match:
        num, den = frac_match.groups()
        try:
            return float(num) / float(den)
        except ZeroDivisionError:
            return None

    # Try scientific notation: 1e-5, 3.14e2
    sci_match = re.match(r'^(-?\d+\.?\d*)[eE](-?\d+)$', s)
    if sci_match:
        try:
            return float(s)
        except ValueError:
            pass

    # Try regular number
    num_match = re.search(r'-?\d+\.?\d*', s)
    if num_match:
        try:
            return float(num_match.group())
        except ValueError:
            pass

    return None


def math_reward(response: str, ground_truth: str) -> float:
    """
    Score math response by comparing extracted answer to ground truth.

    Extracts the LAST \\boxed{...} from the response and compares it
    to the ground truth answer. Supports both exact string match and
    numeric comparison (with tolerance for floating point).

    Response format expected:
        <think>...reasoning chain...</think>
        <answer>\\boxed{42}</answer>

    Scoring:
        1.0 — Answer matches exactly (numeric or string)
        0.3 — Correct format (has think + answer + boxed) but wrong answer
        0.0 — No boxed answer found or completely wrong

    Args:
        response: Full model response text
        ground_truth: Expected answer string

    Returns:
        Score in [0, 1]
    """
    # Extract boxed answer(s) with nested brace support
    matches = _extract_boxed(response)
    if not matches:
        return 0.0

    # Use the last boxed answer (final answer)
    extracted = matches[-1].strip()

    # Normalize
    ext_norm = extracted.strip()
    gt_norm = ground_truth.strip()

    # Try numeric comparison
    ext_num = _extract_numeric(ext_norm)
    gt_num = _extract_numeric(gt_norm)

    if ext_num is not None and gt_num is not None:
        # Relative tolerance for large numbers, absolute for small
        if gt_num == 0:
            if abs(ext_num) < 1e-6:
                return 1.0
        else:
            rel_error = abs(ext_num - gt_num) / max(abs(gt_num), 1e-8)
            if rel_error < 1e-4 or abs(ext_num - gt_num) < 1e-6:
                return 1.0

    # String comparison (case-insensitive, whitespace-normalized)
    ext_str = ext_norm.lower().replace(" ", "")
    gt_str = gt_norm.lower().replace(" ", "")
    if ext_str == gt_str:
        return 1.0

    # Partial credit: correct format but wrong answer
    has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))
    if has_think and has_answer:
        return 0.3

    return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Format Reward
# ═══════════════════════════════════════════════════════════════════════

def format_reward(response: str) -> float:
    """
    Score response format compliance.

    Checks that the response follows the expected structure:
      <think>...reasoning process...</think>
      <answer>...final answer...</answer>

    Scoring:
        1.0 — Both think and answer tags present with content
        0.5 — Only one of the two tag pairs present
        0.0 — Neither tag pair present

    Additional checks:
        - think must come before answer (order matters)
        - Both sections must have non-whitespace content

    Args:
        response: Full model response text

    Returns:
        Score in [0, 1]
    """
    # Check for both tag pairs
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

    has_think = think_match is not None and think_match.group(1).strip() != ""
    has_answer = answer_match is not None and answer_match.group(1).strip() != ""

    if has_think and has_answer:
        # Verify think comes before answer
        think_end = think_match.end()
        answer_start = answer_match.start()
        if think_end <= answer_start:
            return 1.0
        # Think and answer present but order reversed
        return 0.7

    if has_think or has_answer:
        return 0.5

    return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Code Reward
# ═══════════════════════════════════════════════════════════════════════

def code_reward(
    response: str,
    test_cases: list[dict],
    function_name: str = "solution",
    timeout: int = 30,
) -> float:
    """
    Score code response by executing against test cases.

    Extracts Python code from the response (last ```python block or
    last ``` block), wraps it in a test harness, and executes it.

    Scoring:
        Fraction of test cases passed (e.g., 3/5 → 0.6).
        If no code block found, returns 0.0.
        If execution fails (syntax error, runtime error, timeout), returns 0.0.

    Args:
        response: Full model response text
        test_cases: List of test case dicts, each with:
                    - "input": Input string (optional)
                    - "expected_output": Expected output string
                    - "function": Function name to call (default: function_name arg)
        function_name: Default function name to call
        timeout: Max seconds for code execution

    Returns:
        Score in [0, 1] (fraction of tests passed)
    """
    if not test_cases:
        return 0.0

    # Extract code block from response
    code = _extract_code(response)
    if code is None:
        return 0.0

    # Default function name
    func_name = function_name

    # Build test harness and run each test case
    passed = 0
    for tc in test_cases:
        tc_func = tc.get("function", func_name)
        tc_input = tc.get("input", "")
        tc_expected = tc.get("expected_output", "")

        if _run_single_test(code, tc_func, tc_input, tc_expected, timeout):
            passed += 1

    return passed / len(test_cases)


def _extract_code(response: str) -> Optional[str]:
    """Extract Python code from a response's code block."""
    # Try ```python ... ``` first
    py_pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(py_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Fallback: any ``` ... ```
    any_pattern = r'```(?:\w*)?\s*\n(.*?)```'
    matches = re.findall(any_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Last resort: try to find the function definition directly
    func_pattern = r'(def\s+\w+\s*\([^)]*\).*?)(?:\n\n|\Z)'
    matches = re.findall(func_pattern, response, re.DOTALL)
    if matches:
        return "\n".join(matches)

    return None


def _run_single_test(
    code: str,
    function_name: str,
    test_input: str,
    expected_output: str,
    timeout: int = 30,
) -> bool:
    """
    Run a single test case against the extracted code.

    Returns True if the function output matches expected_output.

    Security: uses a temporary file + restricted subprocess (no shell).
    """
    import os as _os
    import json as _json

    # Validate test_input — only allow safe characters (no code injection)
    if not _is_safe_test_input(test_input):
        return False

    # Validate function_name: only allow valid Python identifiers
    import re as _re
    if not _re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', function_name):
        return False

    # Build test harness
    harness_lines = [
        "import sys, json",
        "# User code",
        code,
        "",
        "# Test harness: use json.loads to safely parse input",
        "_input = json.loads(sys.argv[1])",
        "_expected = json.loads(sys.argv[2])",
        "try:",
        f"    _result = str({function_name}(*_input)) if isinstance(_input, list) else str({function_name}(_input))",
        "    if _result.strip() == str(_expected).strip():",
        "        print('PASS')",
        "    else:",
        "        print('FAIL')",
        "except Exception as e:",
        "    print('FAIL')",
    ]
    test_script = "\n".join(harness_lines)

    # Parse input safely: try as JSON, fallback to simple types
    try:
        input_val = _json.loads(test_input) if test_input.strip() else ""
    except (_json.JSONDecodeError, ValueError):
        input_val = test_input
    try:
        expected_val = _json.loads(expected_output) if expected_output.strip() else expected_output
    except (_json.JSONDecodeError, ValueError):
        expected_val = expected_output

    try:
        result = subprocess.run(
            ["python", "-c", test_script,
             _json.dumps(input_val), _json.dumps(expected_val)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        return "PASS" in result.stdout and "FAIL" not in result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return False


def _is_safe_test_input(s: str) -> bool:
    """Check that test_input doesn't contain code-injection patterns."""
    dangerous = ["__import__", "os.system", "subprocess", "eval(", "exec(",
                 "open(", "rm -", "shutil", "import ", "compile("]
    return not any(d in s.lower() for d in dangerous)


# ═══════════════════════════════════════════════════════════════════════
# Length Penalty
# ═══════════════════════════════════════════════════════════════════════

def length_penalty(
    response: str,
    min_tokens: int = 50,
    max_tokens: int = 4096,
    target_tokens: int = 512,
) -> float:
    """
    Penalize responses that are too short or too long.

    Uses a Gaussian-like penalty centered on target_tokens:
        - Responses near target_tokens get score ~1.0
        - Very short responses (< min_tokens) get score ~0.0
        - Very long responses (> max_tokens) get linearly decreasing score

    Args:
        response: Full model response text
        min_tokens: Minimum acceptable length (tokens)
        max_tokens: Maximum acceptable length (tokens)
        target_tokens: Ideal response length (tokens)

    Returns:
        Score in [0, 1]
    """
    # Rough token count: split on whitespace
    tokens = len(response.split())

    if tokens <= min_tokens:
        return tokens / max(min_tokens, 1)
    elif tokens <= target_tokens:
        # Linear ramp from min to target
        return min_tokens / target_tokens + (1.0 - min_tokens / target_tokens) * (
            (tokens - min_tokens) / (target_tokens - min_tokens)
        )
    elif tokens <= max_tokens:
        # Linear decay from target to max
        return 1.0 - 0.5 * ((tokens - target_tokens) / (max_tokens - target_tokens))
    else:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════
# Reward Calculator (Dispatch)
# ═══════════════════════════════════════════════════════════════════════

class RewardCalculator:
    """
    Dispatch to appropriate reward function based on reward type.

    Usage:
        calc = RewardCalculator()
        score = calc.compute(
            response="<think>...</think><answer>42</answer>",
            reward_type="math",
            ground_truth="42",
        )

    Supported reward types:
        - "math": Math answer correctness
        - "format": Response format compliance
        - "code": Code correctness (requires test_cases)
        - "length": Response length penalty
        - "combined": Weighted combination (requires reward_weights dict)
    """

    _DISPATCH = {
        "math": math_reward,
        "format": format_reward,
        "code": code_reward,
        "length": length_penalty,
    }

    def compute(
        self,
        response: str,
        reward_type: str = "format",
        **kwargs,
    ) -> float:
        """
        Compute reward for a response.

        Args:
            response: Full model response text
            reward_type: One of "math", "format", "code", "length", "combined"
            **kwargs: Forwarded to the specific reward function:
                      - math: ground_truth (str)
                      - code: test_cases (list[dict]), function_name (str)
                      - length: min_tokens, max_tokens, target_tokens (int)
                      - combined: reward_weights (dict[str, float]),
                                  plus kwargs for each sub-reward

        Returns:
            Reward score in [0, 1]
        """
        if reward_type == "combined":
            return self._combined_reward(response, **kwargs)

        reward_fn = self._DISPATCH.get(reward_type)
        if reward_fn is None:
            raise ValueError(
                f"Unknown reward_type '{reward_type}'. "
                f"Available: {list(self._DISPATCH.keys())} + 'combined'"
            )

        return reward_fn(response, **kwargs)

    def _combined_reward(self, response: str, **kwargs) -> float:
        """
        Compute weighted combination of multiple reward types.

        Args:
            response: Full model response text
            reward_weights: dict like {"math": 0.7, "format": 0.3}
            **kwargs: Forwarded to each sub-reward (keyed by type)

        Example:
            calc.compute(
                response=text,
                reward_type="combined",
                reward_weights={"math": 0.6, "format": 0.3, "length": 0.1},
                math={"ground_truth": "42"},
            )
        """
        weights = kwargs.pop("reward_weights", None)
        if weights is None:
            raise ValueError("'combined' reward type requires 'reward_weights' dict")

        total = 0.0
        total_weight = 0.0

        for rtype, weight in weights.items():
            rfn = self._DISPATCH.get(rtype)
            if rfn is None:
                continue

            # Get type-specific kwargs
            type_kwargs = kwargs.pop(rtype, {})
            if not isinstance(type_kwargs, dict):
                raise TypeError(
                    f"Expected dict for reward type '{rtype}' kwargs, "
                    f"got {type(type_kwargs).__name__}"
                )
            score = rfn(response, **type_kwargs)
            total += weight * score
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total / total_weight
