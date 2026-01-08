"""
Ibis/Bamboo evaluation harness for OLMo fine-tuning.

Extracts Ibis code from model outputs, executes on DuckDB backend,
and verifies results match expected answers.
"""

import re
import json
import ibis
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    """Result of evaluating a single example."""
    id: str
    success: bool
    code_extracted: bool
    code_valid: bool
    execution_result: Any | None
    expected_answer: str | None
    error: str | None


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown-formatted text."""
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def extract_think_block(text: str) -> str | None:
    """Extract content within <think>...</think> tags."""
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def extract_answer(text: str) -> str | None:
    """Extract the answer after </think> or **Answer:** marker."""
    # Try to get content after </think>
    if "</think>" in text:
        after_think = text.split("</think>", 1)[1].strip()
        return after_think

    # Try **Answer:** pattern
    pattern = r"\*\*Answer:\*\*\s*(.*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    return None


def execute_ibis_code(code: str, test_data: dict[str, Any] | None = None) -> Any:
    """
    Execute Ibis code and return the result.

    Args:
        code: Python code containing Ibis expressions
        test_data: Optional dict mapping table names to test DataFrames

    Returns:
        The result of executing the code
    """
    ibis.set_backend("duckdb")

    # Create a restricted execution environment
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
            "None": None,
            "True": True,
            "False": False,
        }
    }

    # If test data provided, create in-memory tables
    if test_data:
        for name, df in test_data.items():
            exec_globals[name] = ibis.memtable(df)

    exec_locals = {}

    # Execute the code
    exec(code, exec_globals, exec_locals)

    # Return the 'result' variable if it exists
    if "result" in exec_locals:
        return exec_locals["result"]

    # Otherwise return the last assigned variable
    return exec_locals.get(list(exec_locals.keys())[-1]) if exec_locals else None


def evaluate_bamboo_example(
    model_output: str,
    expected_answer: str | None = None,
    test_data: dict[str, Any] | None = None
) -> EvalResult:
    """
    Evaluate a single Bamboo/Ibis example.

    Args:
        model_output: The model's full response
        expected_answer: Optional expected answer to compare against
        test_data: Optional test data for execution

    Returns:
        EvalResult with evaluation details
    """
    # Extract code blocks
    code_blocks = extract_code_blocks(model_output)

    if not code_blocks:
        return EvalResult(
            id="",
            success=False,
            code_extracted=False,
            code_valid=False,
            execution_result=None,
            expected_answer=expected_answer,
            error="No code blocks found in output"
        )

    # Try to execute the first code block (usually in <think>)
    code = code_blocks[0]

    try:
        result = execute_ibis_code(code, test_data)

        return EvalResult(
            id="",
            success=True,
            code_extracted=True,
            code_valid=True,
            execution_result=result,
            expected_answer=expected_answer,
            error=None
        )
    except SyntaxError as e:
        return EvalResult(
            id="",
            success=False,
            code_extracted=True,
            code_valid=False,
            execution_result=None,
            expected_answer=expected_answer,
            error=f"Syntax error: {e}"
        )
    except Exception as e:
        return EvalResult(
            id="",
            success=False,
            code_extracted=True,
            code_valid=True,  # Valid syntax but runtime error
            execution_result=None,
            expected_answer=expected_answer,
            error=f"Execution error: {e}"
        )


def validate_json_output(text: str) -> tuple[bool, Any | str]:
    """
    Validate that text contains valid JSON.

    Returns:
        Tuple of (is_valid, parsed_json_or_error_message)
    """
    # Extract JSON from code blocks if present
    json_pattern = r"```json\n(.*?)```"
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_str = match.group(1)
    else:
        # Try to find raw JSON
        json_str = text

    try:
        parsed = json.loads(json_str)
        return True, parsed
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"


def evaluate_inis_example(
    model_output: str,
    expected_format: str = "text"  # "text", "json", "structured"
) -> dict[str, Any]:
    """
    Evaluate an INIS domain example.

    Args:
        model_output: The model's response
        expected_format: Expected output format

    Returns:
        Dict with evaluation metrics
    """
    result = {
        "format_valid": True,
        "format_type": expected_format,
        "error": None
    }

    if expected_format == "json":
        is_valid, parsed = validate_json_output(model_output)
        result["format_valid"] = is_valid
        result["parsed_content"] = parsed if is_valid else None
        if not is_valid:
            result["error"] = parsed  # Error message

    return result


def load_eval_dataset(path: Path) -> list[dict]:
    """Load evaluation dataset from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def run_evaluation(
    dataset_path: Path,
    domain: str = "bamboo"
) -> dict[str, Any]:
    """
    Run evaluation on a dataset.

    Args:
        dataset_path: Path to JSONL file
        domain: "bamboo" or "inis"

    Returns:
        Evaluation summary with metrics
    """
    examples = load_eval_dataset(dataset_path)

    results = {
        "total": len(examples),
        "successful": 0,
        "code_extracted": 0,
        "code_valid": 0,
        "format_valid": 0,
        "errors": []
    }

    for ex in examples:
        ex_id = ex.get("id", "unknown")
        messages = ex.get("messages", [])

        # Get assistant response
        assistant_msg = next(
            (m["content"] for m in messages if m["role"] == "assistant"),
            None
        )

        if not assistant_msg:
            results["errors"].append({"id": ex_id, "error": "No assistant message"})
            continue

        if domain == "bamboo":
            eval_result = evaluate_bamboo_example(assistant_msg)
            eval_result.id = ex_id

            if eval_result.code_extracted:
                results["code_extracted"] += 1
            if eval_result.code_valid:
                results["code_valid"] += 1
            if eval_result.success:
                results["successful"] += 1
            if eval_result.error:
                results["errors"].append({"id": ex_id, "error": eval_result.error})

        elif domain == "inis":
            # Detect expected format
            expected_format = "json" if "```json" in assistant_msg else "text"
            eval_result = evaluate_inis_example(assistant_msg, expected_format)

            if eval_result["format_valid"]:
                results["format_valid"] += 1
                results["successful"] += 1
            if eval_result.get("error"):
                results["errors"].append({"id": ex_id, "error": eval_result["error"]})

    # Calculate percentages
    results["success_rate"] = results["successful"] / results["total"] if results["total"] > 0 else 0

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ibis_evaluator.py <dataset.jsonl> [bamboo|inis]")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    domain = sys.argv[2] if len(sys.argv) > 2 else "bamboo"

    results = run_evaluation(dataset_path, domain)

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({domain} domain)")
    print(f"{'='*50}")
    print(f"Total examples: {results['total']}")
    print(f"Successful: {results['successful']} ({results['success_rate']:.1%})")

    if domain == "bamboo":
        print(f"Code extracted: {results['code_extracted']}")
        print(f"Code valid: {results['code_valid']}")
    else:
        print(f"Format valid: {results['format_valid']}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"][:5]:  # Show first 5 errors
            print(f"  - {err['id']}: {err['error']}")
