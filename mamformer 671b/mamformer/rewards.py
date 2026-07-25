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


def _is_safe_test_input(s: str) -> bool:
    """Check that test_input doesn't contain code-injection patterns.
    This is a pre-filter; the main protection is the restricted builtins."""
    dangerous = ["__import__", "os.system", "subprocess", "eval(", "exec(",
                 "open(", "rm -", "shutil", "import ", "compile("]
    return not any(d in s.lower() for d in dangerous)


def _scan_user_code(code: str) -> Optional[str]:
    """
    Scan user code for sandbox bypass attempts. Returns None if safe,
    or a string describing the violation.
    """
    import re as _re

    # Check for attempts to access unrestricted builtins
    bypass_patterns = [
        (r'__import__', 'direct __import__ call'),
        (r'\bopen\s*\(', 'open() call'),
        (r'\beval\s*\(', 'eval() call'),
        (r'\bexec\s*\(', 'exec() call'),
        (r'\bcompile\s*\(', 'compile() call'),
        (r'\bglobals\s*\(\s*\)', 'globals() access'),
        (r'\blocals\s*\(\s*\)', 'locals() access'),
        (r'\bgetattr\s*\(', 'getattr() — potential builtins bypass'),
        (r'__builtins__', 'direct __builtins__ access'),
        (r'__class__\s*\.', 'dunder class traversal'),
        (r'__bases__', 'dunder bases traversal'),
        (r'__subclasses__\s*\(\s*\)', 'subclass enumeration'),
        (r'__globals__', 'dunder globals access'),
        (r'\b(?:import|from)\s+\w+', 'import statement'),
        (r'__import__', 'import via dunder'),
        (r'breakpoint\s*\(', 'breakpoint() call'),
        (r'copyright|credits|license', 'interactive-help builtin'),
    ]
    for pattern, desc in bypass_patterns:
        if _re.search(pattern, code):
            return f"Forbidden pattern in user code: {desc}"
    return None


def _run_single_test(
    code: str,
    function_name: str,
    test_input: str,
    expected_output: str,
    timeout: int = 30,
) -> bool:
    """Run a single test case in a restricted-builtins sandbox.
    Dangerous builtins (__import__, open, eval, exec, etc.) are
    deleted before user code executes."""
    import os as _os
    import json as _json
    import re as _re

    if not _re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', function_name):
        return False

    violation = _scan_user_code(code)
    if violation is not None:
        return False

    if not _is_safe_test_input(test_input):
        return False

    harness_lines = [
        "import sys, json",
        "",
        "# Delete dangerous builtins from __dict__",
        "_DANGEROUS = [",
        '    "__import__", "open", "eval", "exec", "compile",',
        '    "globals", "locals", "vars", "dir",',
        '    "getattr", "setattr", "delattr", "hasattr",',
        '    "breakpoint", "input",',
        '    "copyright", "credits", "license", "help",',
        "]",
        "import builtins",
        "_bd = builtins.__dict__",
        "for _name in _DANGEROUS:",
        "    _bd.pop(_name, None)",
        "if isinstance(__builtins__, dict):",
        "    for _name in _DANGEROUS:",
        "        __builtins__.pop(_name, None)",
        "",
        "# User code",
        code,
        "",
        "# Test harness",
        "_input = json.loads(sys.argv[1])",
        "_expected = json.loads(sys.argv[2])",
        "try:",
        f"    _result = str({function_name}(*_input)) if isinstance(_input, list) else str({function_name}(_input))",
        '    if _result.strip() == str(_expected).strip():',
        '        print("PASS")',
        '    else:',
        '        print("FAIL")',
        'except Exception:',
        '    print("FAIL")',
    ]
    test_script = "\n".join(harness_lines)

    try:
        input_val = _json.loads(test_input) if test_input.strip() else ""
    except (_json.JSONDecodeError, ValueError):
        input_val = test_input
    try:
        expected_val = _json.loads(expected_output) if expected_output.strip() else expected_output
    except (_json.JSONDecodeError, ValueError):
        expected_val = expected_output

    try:
        preexec_fn = None
        try:
            import resource
            def _limit():
                resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
            preexec_fn = _limit
        except (ImportError, ValueError):
            pass

        result = subprocess.run(
            ["python", "-c", test_script,
             _json.dumps(input_val), _json.dumps(expected_val)],
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(), preexec_fn=preexec_fn,
        )
        return "PASS" in result.stdout and "FAIL" not in result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return False



# Length Penalty
# ═══════════════════════════════════════════════════════════════════════

def length_penalty(
    response: str,
    min_words: int = 50,
    max_words: int = 4096,
    target_words: int = 512,
    # ── Deprecated aliases (kept for backward compatibility) ──
    min_tokens: int = None,
    max_tokens: int = None,
    target_tokens: int = None,
) -> float:
    """
    Penalize responses that are too short or too long.

    Uses word count (whitespace split) as a fast approximation for length.
    NOTE: This counts WORDS, not tokens. A rough heuristic: tokens ≈ 1.3× words.
    For precise token-level control, pass an actual tokenizer.

    Args:
        response: Full model response text
        min_words: Minimum acceptable length in words
        max_words: Maximum acceptable length in words
        target_words: Ideal response length in words
        min_tokens: Deprecated — use min_words
        max_tokens: Deprecated — use max_words
        target_tokens: Deprecated — use target_words

    Returns:
        Score in [0, 1]
    """
    # Backward compat: old parameter names
    if min_tokens is not None:
        min_words = min_tokens
    if max_tokens is not None:
        max_words = max_tokens
    if target_tokens is not None:
        target_words = target_tokens

    # Rough word count: split on whitespace
    n_words = len(response.split())

    if n_words <= min_words:
        return n_words / max(min_words, 1)
    elif n_words <= target_words:
        # Linear ramp from min to target
        return min_words / target_words + (1.0 - min_words / target_words) * (
            (n_words - min_words) / (target_words - min_words)
        )
    elif n_words <= max_words:
        # Linear decay from target to max
        return 1.0 - 0.5 * ((n_words - target_words) / (max_words - target_words))
    else:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════
# Thinking Quality Reward
# ═══════════════════════════════════════════════════════════════════════

def thinking_quality_reward(response: str) -> float:
    """
    Score the quality of thinking in a response that uses thinking tokens.

    Evaluates:
      1. Structure: does the response contain thinking markers?
      2. Substance: is there meaningful content in the thinking sections?
      3. Separation: is thinking separated from the final answer?

    Scoring:
      - 1.0: Well-structured thinking with markers + substantive reasoning
      - 0.7: Has markers but thinking content is shallow (< 20 chars)
      - 0.4: Has some markers but incomplete structure
      - 0.0: No thinking markers at all

    Args:
        response: Full model response text

    Returns:
        Score in [0, 1]
    """
    score = 0.0

    # Check for thinking control tokens (both text and token forms)
    has_think_start = bool(re.search(
        r'<\|think_start\|>|<think_start>|<think>|<thinking>',
        response, re.IGNORECASE,
    ))
    has_think_end = bool(re.search(
        r'<\|think_end\|>|<think_end>|</think>|</thinking>',
        response, re.IGNORECASE,
    ))
    has_answer_start = bool(re.search(
        r'<\|answer_start\|>|<answer_start>|<answer>',
        response, re.IGNORECASE,
    ))

    # Structure score
    if has_think_start and has_think_end and has_answer_start:
        score += 0.4
    elif has_think_start and has_answer_start:
        score += 0.25
    elif has_think_start:
        score += 0.1

    # Substance: extract thinking content and check depth
    think_patterns = [
        r'<\|think_start\|>(.*?)<\|think_end\|>',
        r'<think_start>(.*?)<think_end>',
        r'<think>(.*?)</think>',
        r'<thinking>(.*?)</thinking>',
    ]
    for pat in think_patterns:
        m = re.search(pat, response, re.DOTALL | re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            if len(content) > 100:
                score += 0.3  # Substantive reasoning
            elif len(content) > 20:
                score += 0.2  # Some reasoning
            else:
                score += 0.05  # Token thinking
            break

    # Separation: thinking should come before answer
    if has_think_start and has_answer_start:
        think_pos = response.lower().find('<think')
        answer_pos = response.lower().find('<answer')
        if think_pos >= 0 and answer_pos >= 0 and think_pos < answer_pos:
            score += 0.2  # Correct order
        elif think_pos >= 0 and answer_pos >= 0:
            score += 0.1  # Wrong order but both present

    # Diversity bonus: multiple distinct thinking paths
    path_markers = len(re.findall(
        r'<\|think_start\|>|<think_start>|<think>',
        response, re.IGNORECASE,
    ))
    if path_markers > 1:
        score += 0.1  # Multi-path bonus

    return min(score, 1.0)


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
        "thinking": thinking_quality_reward,
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
