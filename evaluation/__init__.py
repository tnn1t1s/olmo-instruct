"""Evaluation harness for OLMo INIS/Bamboo fine-tuning."""

from .ibis_evaluator import (
    EvalResult,
    evaluate_bamboo_example,
    evaluate_inis_example,
    extract_code_blocks,
    extract_think_block,
    run_evaluation,
)

from .verifier import (
    VerificationResult,
    verify_numeric,
    verify_exact_string,
    verify_json_field,
    compute_reward,
    create_verifier,
)

__all__ = [
    # Evaluation
    "EvalResult",
    "evaluate_bamboo_example",
    "evaluate_inis_example",
    "extract_code_blocks",
    "extract_think_block",
    "run_evaluation",
    # RLVR Verification
    "VerificationResult",
    "verify_numeric",
    "verify_exact_string",
    "verify_json_field",
    "compute_reward",
    "create_verifier",
]
