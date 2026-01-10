"""
RLVR Verification Functions for OLMo INIS/Bamboo Fine-tuning.

Provides binary reward signals (0 or 1) for reinforcement learning
with verifiable rewards.
"""

import re
import json
import math
from typing import Any
from dataclasses import dataclass

import ibis


@dataclass
class VerificationResult:
    """Result of verifying a model output."""
    reward: float  # 0.0 or 1.0
    extracted_answer: Any | None
    expected_answer: Any
    match: bool
    error: str | None = None


def extract_numeric_answer(text: str) -> float | None:
    """
    Extract the final numeric answer from model output.

    Looks for patterns like:
    - **Answer:** 42
    - The answer is 42
    - = 42
    - Final result: 42
    """
    # Try common answer patterns
    patterns = [
        r"\*\*Answer:\*\*\s*([\d,.-]+)",
        r"[Aa]nswer[:\s]+(?:is\s+)?([\d,.-]+)",
        r"[Rr]esult[:\s]+(?:is\s+)?([\d,.-]+)",
        r"=\s*([\d,.-]+)\s*$",
        r"total[:\s]+(?:is\s+)?([\d,.-]+)",
        r"([\d,.-]+)\s*(?:$|\n)",  # Last number in output
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            # Take the last match (usually the final answer)
            num_str = matches[-1].replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                continue

    # Fallback: find any number in the text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def extract_code_and_execute(
    text: str,
    test_data: dict[str, list[dict]] | None = None
) -> Any | None:
    """
    Extract Ibis code from model output and execute it.

    Args:
        text: Model output containing ```python code blocks
        test_data: Dict mapping table names to list of row dicts

    Returns:
        Execution result or None if failed
    """
    # Extract code blocks
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    code = matches[0]

    try:
        ibis.set_backend("duckdb")

        # Create execution environment
        exec_globals = {
            "ibis": ibis,
            "__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "sum": sum,
                "None": None,
                "True": True,
                "False": False,
            }
        }

        # Load test data as in-memory tables
        if test_data:
            for name, rows in test_data.items():
                exec_globals[name] = ibis.memtable(rows)

        exec_locals = {}
        exec(code, exec_globals, exec_locals)

        # Try to find the result
        if "result" in exec_locals:
            result = exec_locals["result"]
            # If it's an Ibis expression, execute it
            if hasattr(result, "execute"):
                return result.execute()
            return result

        # Return last assigned value
        if exec_locals:
            last_val = list(exec_locals.values())[-1]
            if hasattr(last_val, "execute"):
                return last_val.execute()
            return last_val

    except Exception as e:
        return None

    return None


def verify_numeric(
    model_output: str,
    ground_truth: str | float,
    tolerance: float = 0.01,
    test_data: dict | None = None
) -> VerificationResult:
    """
    Verify a numeric answer.

    Args:
        model_output: The model's full response
        ground_truth: Expected numeric answer
        tolerance: Relative tolerance for comparison (0.01 = 1%)
        test_data: Optional test data for Ibis execution

    Returns:
        VerificationResult with reward 1.0 if correct, 0.0 otherwise
    """
    expected = float(ground_truth)

    # First try to extract from stated answer
    extracted = extract_numeric_answer(model_output)

    # If we have test data, also try executing the code
    if test_data and extracted is None:
        exec_result = extract_code_and_execute(model_output, test_data)
        if exec_result is not None:
            # Handle DataFrame results
            if hasattr(exec_result, "iloc"):
                # Get first numeric value from DataFrame
                try:
                    extracted = float(exec_result.iloc[0, -1])
                except (IndexError, ValueError):
                    pass
            elif isinstance(exec_result, (int, float)):
                extracted = float(exec_result)

    if extracted is None:
        return VerificationResult(
            reward=0.0,
            extracted_answer=None,
            expected_answer=expected,
            match=False,
            error="Could not extract numeric answer"
        )

    # Compare with tolerance
    if expected == 0:
        match = abs(extracted) < tolerance
    else:
        match = abs(extracted - expected) / abs(expected) <= tolerance

    return VerificationResult(
        reward=1.0 if match else 0.0,
        extracted_answer=extracted,
        expected_answer=expected,
        match=match,
        error=None
    )


def verify_exact_string(
    model_output: str,
    ground_truth: str,
    case_sensitive: bool = False
) -> VerificationResult:
    """
    Verify an exact string match.

    Args:
        model_output: The model's full response
        ground_truth: Expected string answer
        case_sensitive: Whether comparison is case-sensitive

    Returns:
        VerificationResult with reward 1.0 if correct, 0.0 otherwise
    """
    # Extract answer portion
    answer_pattern = r"\*\*Answer:\*\*\s*(.*?)(?:\n|$)"
    match = re.search(answer_pattern, model_output)

    if match:
        extracted = match.group(1).strip()
    else:
        # Use last line as answer
        lines = model_output.strip().split("\n")
        extracted = lines[-1].strip()

    if case_sensitive:
        is_match = extracted == ground_truth
    else:
        is_match = extracted.lower() == ground_truth.lower()

    return VerificationResult(
        reward=1.0 if is_match else 0.0,
        extracted_answer=extracted,
        expected_answer=ground_truth,
        match=is_match,
        error=None
    )


def verify_json_field(
    model_output: str,
    field_path: str,
    expected_value: Any
) -> VerificationResult:
    """
    Verify a specific field in JSON output.

    Args:
        model_output: The model's response containing JSON
        field_path: Dot-separated path to field (e.g., "results.temperature")
        expected_value: Expected value at that path

    Returns:
        VerificationResult with reward 1.0 if correct, 0.0 otherwise
    """
    # Extract JSON from code blocks
    json_pattern = r"```json\n(.*?)```"
    match = re.search(json_pattern, model_output, re.DOTALL)

    if not match:
        return VerificationResult(
            reward=0.0,
            extracted_answer=None,
            expected_answer=expected_value,
            match=False,
            error="No JSON block found"
        )

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        return VerificationResult(
            reward=0.0,
            extracted_answer=None,
            expected_answer=expected_value,
            match=False,
            error=f"JSON parse error: {e}"
        )

    # Navigate to field
    current = data
    for key in field_path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return VerificationResult(
                    reward=0.0,
                    extracted_answer=None,
                    expected_answer=expected_value,
                    match=False,
                    error=f"Index {idx} out of range"
                )
        else:
            return VerificationResult(
                reward=0.0,
                extracted_answer=None,
                expected_answer=expected_value,
                match=False,
                error=f"Field '{key}' not found"
            )

    # Compare values
    is_match = current == expected_value

    return VerificationResult(
        reward=1.0 if is_match else 0.0,
        extracted_answer=current,
        expected_answer=expected_value,
        match=is_match,
        error=None
    )


def create_verifier(verification_type: str):
    """
    Factory function to create appropriate verifier.

    Args:
        verification_type: One of "numeric", "string", "json_field"

    Returns:
        Verification function
    """
    verifiers = {
        "numeric": verify_numeric,
        "string": verify_exact_string,
        "json_field": verify_json_field,
    }
    return verifiers.get(verification_type, verify_numeric)


def compute_reward(
    model_output: str,
    example: dict
) -> float:
    """
    Compute RLVR reward for a single example.

    Args:
        model_output: The model's response
        example: Dict with ground_truth, verification_type, and optionally test_data

    Returns:
        Reward value (0.0 or 1.0)
    """
    verification_type = example.get("verification_type", "numeric")
    ground_truth = example.get("ground_truth")
    test_data = example.get("test_data")

    if verification_type == "numeric":
        result = verify_numeric(model_output, ground_truth, test_data=test_data)
    elif verification_type == "string":
        result = verify_exact_string(model_output, ground_truth)
    elif verification_type == "json_field":
        field_path = example.get("field_path", "")
        result = verify_json_field(model_output, field_path, ground_truth)
    else:
        result = verify_numeric(model_output, ground_truth, test_data=test_data)

    return result.reward


if __name__ == "__main__":
    # Test the verifier
    test_output = """<think>
I need to calculate the total revenue for the North region.

```python
import ibis
sales = ibis.memtable([
    {"region": "North", "product": "Widget", "revenue": 1000},
    {"region": "North", "product": "Gadget", "revenue": 1500},
])
result = sales.filter(sales.region == "North").revenue.sum().execute()
```
</think>

**Answer:** 2500
"""

    result = verify_numeric(test_output, "2500")
    print(f"Reward: {result.reward}")
    print(f"Extracted: {result.extracted_answer}")
    print(f"Expected: {result.expected_answer}")
    print(f"Match: {result.match}")
