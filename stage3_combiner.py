#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Stage 3: Combiner
Equal-weight combines forecasts from all methods.
Uses OUTPUT_B evidence packet for consequence narratives.
"""

import os
import json
from typing import Dict, List, Optional
from utils import (
    log, log_progress, save_response, call_llm,
    extract_json_blocks, is_retryable_error,
    normalize_leg_results, MAX_RETRIES, RETRY_DELAYS
)
import time

# Get model and prompt from environment
MODEL_COMBINER = os.getenv("MODEL_FC", "").strip()  # Use forecast model for combiner
PROMPT_FINAL_COMBINER = os.getenv("PROMPT_FINAL_COMBINER", "").strip()

# Validate
if not MODEL_COMBINER:
    log("[FATAL] MODEL_FC must be set in environment (used for combiner)")
    raise ValueError("MODEL_FC not set")

if not PROMPT_FINAL_COMBINER:
    log("[FATAL] PROMPT_FINAL_COMBINER must be set in environment")
    raise ValueError("PROMPT_FINAL_COMBINER not set")

# ========================================
# EXTRACT BRANCH VALUES
# ========================================

def extract_forecast_value(leg_results: dict, leg_name: str) -> Optional[float]:
    """
    Extract forecast value from a leg's results.
    Handles binary (P_YES), numeric (MEAN), MC (average of probs), date (mean of dates).
    Also handles Panshul's direct forecast values.
    Returns a single float value for equal-weighting.
    """
    if not leg_results or not isinstance(leg_results, dict):
        return None
    
    result_key = f"{leg_name}_RESULTS"
    if result_key not in leg_results:
        return None
    
    result_data = leg_results[result_key]
    if not isinstance(result_data, dict):
        return None
    
    # SPECIAL CASE: Panshul results (no branches, direct forecast)
    if leg_name == "PANSHUL" and "forecast" in result_data:
        forecast = result_data["forecast"]
        
        # Binary: single float
        if isinstance(forecast, (int, float)):
            return float(forecast)
        
        # Multiple choice: dict of probabilities
        if isinstance(forecast, dict):
            # For MC, return the highest probability (most likely outcome)
            # This is a simplification but allows equal-weighting
            probs = [float(v) for v in forecast.values() if isinstance(v, (int, float))]
            return max(probs) if probs else None
        
        # Numeric: list (CDF)
        if isinstance(forecast, list):
            # CDF list - skip for now, too complex for simple averaging
            log(f"[COMBINER] Panshul numeric CDF detected - skipping for simple average")
            return None
        
        return None
    
    # NORMAL CASE: Method results with branches (A/B/C/D)
    # Find the branch (A, B, C, or D)
    branch = None
    for key in ['A', 'B', 'C', 'D']:
        if key in result_data:
            branch = result_data[key]
            break
    
    if not branch or not isinstance(branch, dict):
        return None
    
    # Extract value based on type
    # Binary: P_YES
    if "P_YES" in branch:
        return float(branch["P_YES"])
    
    # Numeric: MEAN
    if "MEAN" in branch:
        return float(branch["MEAN"])
    
    # Multiple choice: highest probability
    if "CANDIDATE_PROBS" in branch:
        probs = branch["CANDIDATE_PROBS"]
        if isinstance(probs, list):
            # Return highest probability
            values = [float(item.get("p", 0)) for item in probs if isinstance(item, dict)]
            return max(values) if values else None
    
    # Date: not easily reduced to single value, skip for now
    if "DATE_PROBS" in branch:
        log(f"[COMBINER] Date questions not yet supported for simple averaging")
        return None
    
    return None

# ========================================
# EQUAL-WEIGHT COMBINING
# ========================================

def combine_forecasts(
    qobj: dict,
    panshul_result: Optional[dict],
    method_results: Dict[str, dict],
    evidence_packet: dict
) -> Optional[dict]:
    """
    Combine all method forecasts with equal weighting.
    
    Inputs:
    - qobj: Question object
    - panshul_result: Raw Panshul result {"forecast": ..., "comment": ...} or None
    - method_results: {"KM": {...}, "BD": {...}, "EX": {...}} or None
    - evidence_packet: Evidence for THIS question from OUTPUT_B
    
    Returns: Final combined forecast or None if failed
    """
    log_progress(f"🎯 COMBINING FORECASTS FOR Q {qobj.get('question_id')}")
    
    # Build consolidated results for prompt
    consolidated = {}
    
    # Add Panshul - wrap it for the prompt
    if panshul_result:
        consolidated["PANSHUL_RESULTS"] = panshul_result
    
    # Add method results
    for method_name, result in method_results.items():
        if method_name == "EX":
            # EX has special structure with MK + PL
            consolidated["EX_RESULTS"] = result.get("EX_RESULTS", {})
            consolidated["MK_RESULTS"] = result.get("MK_RESULTS", {})
            consolidated["PL_RESULTS"] = result.get("PL_RESULTS", {})
        else:
            result_key = f"{method_name}_RESULTS"
            consolidated[result_key] = result.get(result_key, {})
    
    log(f"[COMBINER] Consolidated results from: {', '.join(consolidated.keys())}")
    
    # Build prompt payload
    user_lines = [
        "QUESTION_OBJECT:",
        json.dumps(qobj, indent=2),
        "",
        "ALL_RESULTS:",
        json.dumps(consolidated, indent=2),
        "",
        "EVIDENCE_PACKET:",
        json.dumps(evidence_packet, indent=2)
    ]
    user_payload = "\n".join(user_lines)
    
    # Call LLM
    final_payload = None
    response = ""
    
    for attempt in range(1, MAX_RETRIES + 1):
        response = call_llm(
            MODEL_COMBINER,
            PROMPT_FINAL_COMBINER,
            user_payload,
            max_tokens=24000,
            timeout=300
        )
        log(f"[COMBINER] 📥 Received {len(response)} chars from model (attempt {attempt}/{MAX_RETRIES})")
        
        # Extract JSON
        blocks = extract_json_blocks(response, "COMBINER")
        
        if blocks:
            # Take the longest block (most likely to be complete)
            block = max(blocks, key=len)
            try:
                final_payload = json.loads(block)
                log(f"[COMBINER] ✅ Valid final payload extracted ({len(block)} chars)")
                break
            except Exception as e:
                log(f"[COMBINER] ⚠️ JSON parse error: {e}")
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[COMBINER] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    if final_payload is None:
        log(f"[COMBINER] ❌ FAILED after {MAX_RETRIES} attempts")
        save_response("combiner_full.txt", response)
        return None
    
    # Save combiner response
    save_response("combiner_full.txt", response)
    
    # Calculate simple equal-weight average for validation
    values = []
    
    if panshul_result:
        val = extract_forecast_value({"PANSHUL_RESULTS": panshul_result}, "PANSHUL")
        if val is not None:
            values.append(("PANSHUL", val))
    
    for method_name in ["KM", "BD"]:
        if method_name in method_results:
            val = extract_forecast_value(method_results[method_name], method_name)
            if val is not None:
                values.append((method_name, val))
    
    # EX special case: extract MK and PL separately
    if "EX" in method_results:
        ex_result = method_results["EX"]
        mk_val = extract_forecast_value({"MK_RESULTS": ex_result.get("MK_RESULTS", {})}, "MK")
        pl_val = extract_forecast_value({"PL_RESULTS": ex_result.get("PL_RESULTS", {})}, "PL")
        if mk_val is not None:
            values.append(("MK", mk_val))
        if pl_val is not None:
            values.append(("PL", pl_val))
    
    # Log individual values
    if values:
        log(f"[COMBINER] Individual forecast values:")
        for name, val in values:
            log(f"  {name}: {val:.4f}")
        
        avg = sum(v for _, v in values) / len(values)
        log(f"[COMBINER] Simple average: {avg:.4f} (from {len(values)} methods)")
    
    # Add metadata
    final_payload["metadata"] = {
        "question_id": qobj.get("question_id"),
        "num_methods": len(values),
        "methods_used": [name for name, _ in values],
        "simple_average": sum(v for _, v in values) / len(values) if values else None
    }
    
    # CRITICAL: Inject all method results into final_payload for artifact downloads
    if "all_method_results" not in final_payload:
        final_payload["all_method_results"] = {}
    
    # Inject Panshul - wrap it properly for artifacts
    if panshul_result:
        final_payload["all_method_results"]["PANSHUL_RESULTS"] = panshul_result
        log(f"[COMBINER] ✅ Injected PANSHUL_RESULTS into final_payload")
    
    # Inject KM, BD
    for method_name in ["KM", "BD"]:
        if method_name in method_results:
            result_key = f"{method_name}_RESULTS"
            if result_key in method_results[method_name]:
                final_payload["all_method_results"][result_key] = method_results[method_name][result_key]
                log(f"[COMBINER] ✅ Injected {result_key} into final_payload")
    
    # Inject EX (special structure with MK + PL)
    if "EX" in method_results:
        ex_result = method_results["EX"]
        if "MK_RESULTS" in ex_result:
            final_payload["all_method_results"]["MK_RESULTS"] = ex_result["MK_RESULTS"]
            log(f"[COMBINER] ✅ Injected MK_RESULTS into final_payload")
        if "PL_RESULTS" in ex_result:
            final_payload["all_method_results"]["PL_RESULTS"] = ex_result["PL_RESULTS"]
            log(f"[COMBINER] ✅ Injected PL_RESULTS into final_payload")
    
    # Save final payload with all method results
    save_response("final_payload.json", json.dumps(final_payload, indent=2))
    
    log_progress(f"✅ COMBINER COMPLETE FOR Q {qobj.get('question_id')}")
    
    return final_payload

# ========================================
# RUN COMBINER FOR A QUESTION
# ========================================

def run_stage3(
    qobj: dict,
    panshul_result: Optional[dict],
    method_results: Dict[str, dict],
    evidence_packet_full: dict
) -> Optional[dict]:
    """
    Run Stage 3 combiner for a single question.
    
    Inputs:
    - qobj: Question object
    - panshul_result: Panshul's forecast or None
    - method_results: {"KM": {...}, "BD": {...}, "EX": {...}}
    - evidence_packet_full: Full OUTPUT_B with all questions
    
    Returns: Final combined forecast or None
    """
    # Extract evidence for THIS question only
    q_id = qobj.get("question_id")
    evidence_for_this_q = evidence_packet_full.get(q_id, {})
    
    if not evidence_for_this_q:
        log(f"[STAGE3] ⚠️ No evidence found for Q {q_id} in OUTPUT_B")
        # Create empty evidence
        evidence_for_this_q = {
            "question_text": qobj.get("question_text", qobj.get("title", "")),
            "consequences": {
                "near_term": "No evidence available",
                "knock_ons": "No evidence available"
            }
        }
    
    # Combine
    result = combine_forecasts(qobj, panshul_result, method_results, evidence_for_this_q)
    
    return result

# ========================================
# END OF STAGE3_COMBINER
# ========================================
