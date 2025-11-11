#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Method Runner
Runs KM/BD/EX legs using 2-phase PT1→PT2 pattern.
Handles EX leg's special dual-output (MK_RESULTS + PL_RESULTS).
"""

import os
import json
import time
from typing import Tuple, Optional, Dict
from utils import (
    log, log_progress, save_response, call_llm,
    extract_json_blocks, is_retryable_error, is_not_applicable_status,
    is_valid_pt1_envelope, validate_pt2_result, extract_pt2_results,
    normalize_leg_results, augment_question_object,
    MAX_RETRIES, RETRY_DELAYS, TIMEOUT_PT1_RESEARCH, TIMEOUT_PT2_FORECAST,
    MAX_TOTAL_TIME_PT1, MAX_TOTAL_TIME_PT2
)

# Get models from environment (NO HARDCODED FALLBACKS)
MODEL_RESEARCH = os.getenv("MODEL_RS", "").strip()
MODEL_FORECAST = os.getenv("MODEL_FC", "").strip()
MODEL_BACKUP = os.getenv("MODEL_BK", "").strip()

# Validate required models
if not MODEL_RESEARCH or not MODEL_FORECAST:
    log("[FATAL] MODEL_RS and MODEL_FC must be set in environment")
    raise ValueError("Required model environment variables not set")

# ========================================
# PT1: RESEARCH PHASE
# ========================================

def run_pt1(
    leg_name: str,
    prompt: str,
    qobj: dict,
    model: str,
    max_retries: int = MAX_RETRIES
) -> Tuple[str, Optional[dict]]:
    """
    Run PT1 (research phase) for a leg.
    Returns: (full_response_text, findings_json_object or None)
    """
    log(f"[{leg_name} PT.1] Starting research on {model}")
    
    # Augment question object
    q_for_leg = augment_question_object(qobj)
    user_payload = "QUESTION_OBJECT:\n" + json.dumps(q_for_leg, ensure_ascii=False, indent=2)
    
    envelope = None
    response = ""
    start_time = time.time()
    
    for attempt in range(1, max_retries + 1):
        elapsed = time.time() - start_time
        if elapsed > MAX_TOTAL_TIME_PT1:
            log(f"[{leg_name} PT.1] ⏱️ TIMEOUT - exceeded {MAX_TOTAL_TIME_PT1}s")
            return response, None
        
        remaining = MAX_TOTAL_TIME_PT1 - elapsed
        current_model = model
        
        # Use backup model on final retry if available and needed
        if attempt == max_retries and MODEL_BACKUP and envelope is None:
            current_model = MODEL_BACKUP
            log(f"[{leg_name} PT.1] 🔄 Switching to backup model: {MODEL_BACKUP}")
        
        # Call LLM
        response = call_llm(
            current_model,
            prompt,
            user_payload,
            max_tokens=48000,
            timeout=min(TIMEOUT_PT1_RESEARCH, int(remaining))
        )
        log(f"[{leg_name} PT.1] 📥 Received {len(response)} chars from model (attempt {attempt}/{max_retries})")
        
        # Extract JSON blocks
        blocks = extract_json_blocks(response, leg_name)
        
        # Try to find valid PT1 envelope
        for raw in blocks:
            try:
                obj = json.loads(raw)
                if is_valid_pt1_envelope(obj, raw, leg_name):
                    envelope = obj
                    log(f"[{leg_name} PT.1] ✅ Valid JSON extracted ({len(raw)} chars)")
                    break
            except Exception:
                continue
        
        # Success - save and return
        if envelope is not None:
            save_response(f"{leg_name.lower()}_pt1_full.txt", response)
            return response, envelope
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < max_retries:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[{leg_name} PT.1] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    # Failed after all retries
    log(f"[{leg_name} PT.1] ❌ FAILED after {max_retries} attempts")
    save_response(f"{leg_name.lower()}_pt1_full.txt", response)
    return response, envelope

# ========================================
# PT2: FORECAST PHASE
# ========================================

def run_pt2(
    leg_name: str,
    prompt: str,
    qobj: dict,
    pt1_output: dict,
    model: str,
    result_key: str,
    max_retries: int = MAX_RETRIES
) -> Tuple[str, Optional[dict]]:
    """
    Run PT2 (forecast phase) for a leg.
    Returns: (full_response_text, results_json_object or None)
    """
    log(f"[{leg_name} PT.2] Starting forecast on {model}")
    
    # Augment question object
    q_for_leg = augment_question_object(qobj)
    
    # Build user payload with question + PT1 findings
    user_lines = [
        "QUESTION_OBJECT:",
        json.dumps(q_for_leg, ensure_ascii=False, indent=2),
        "",
        f"{leg_name}_FINDINGS:",
        json.dumps(pt1_output, ensure_ascii=False, indent=2)
    ]
    user_payload = "\n".join(user_lines)
    
    results = None
    response = ""
    start_time = time.time()
    
    for attempt in range(1, max_retries + 1):
        elapsed = time.time() - start_time
        if elapsed > MAX_TOTAL_TIME_PT2:
            log(f"[{leg_name} PT.2] ⏱️ TIMEOUT - exceeded {MAX_TOTAL_TIME_PT2}s")
            return response, None
        
        remaining = MAX_TOTAL_TIME_PT2 - elapsed
        
        # Call LLM
        response = call_llm(
            model,
            prompt,
            user_payload,
            max_tokens=48000,
            timeout=min(TIMEOUT_PT2_FORECAST, int(remaining))
        )
        log(f"[{leg_name} PT.2] 📥 Received {len(response)} chars from model (attempt {attempt}/{max_retries})")
        
        # Extract JSON blocks
        blocks = extract_json_blocks(response, leg_name)
        result_envelope = extract_pt2_results(blocks, result_key, leg_name)
        
        # Validate
        if result_envelope and validate_pt2_result(result_envelope, result_key, leg_name):
            results = result_envelope
            log(f"[{leg_name} PT.2] ✅ Valid forecast extracted")
            break
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < max_retries:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[{leg_name} PT.2] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    # Save response
    if results is None:
        log(f"[{leg_name} PT.2] ❌ FAILED after {max_retries} attempts")
    
    save_response(f"{leg_name.lower()}_pt2_full.txt", response)
    return response, results

# ========================================
# LEG RUNNER (PT1 → PT2)
# ========================================

def run_leg(
    leg_name: str,
    prompt_pt1: Optional[str],
    prompt_pt2: Optional[str],
    qobj: dict,
    model_research: str,
    model_forecast: str
) -> Optional[dict]:
    """
    Run a complete leg: PT1 (research) → PT2 (forecast).
    Returns normalized results or None if leg failed.
    """
    # Check if prompts are configured
    if not prompt_pt1 or not prompt_pt2:
        log(f"[{leg_name}] ⏭️ SKIPPED (prompts not configured)")
        return None
    
    log_progress(f"🔬 STARTING {leg_name} LEG")
    
    # PT1: Research
    pt1_response, pt1_output = run_pt1(leg_name, prompt_pt1, qobj, model_research)
    
    if pt1_output is None:
        log(f"[{leg_name}] ❌ PT.1 FAILED - aborting leg")
        return None
    
    # Check if N/A
    if is_not_applicable_status(pt1_output.get("status", "")):
        log(f"[{leg_name}] ℹ️ PT.1 returned NOT_APPLICABLE - skipping PT.2")
        return None
    
    # PT2: Forecast
    result_key = f"{leg_name}_RESULTS"
    _, pt2_results = run_pt2(leg_name, prompt_pt2, qobj, pt1_output, model_forecast, result_key)
    
    if pt2_results is None:
        log(f"[{leg_name}] ❌ PT.2 FAILED - leg incomplete")
        return None
    
    # Normalize results
    normalized = normalize_leg_results(pt2_results, leg_name)
    
    log(f"[{leg_name}] ✅ LEG COMPLETE")
    return normalized

# ========================================
# EX LEG RUNNER (SPECIAL CASE)
# ========================================

def run_ex_leg(
    qobj: dict,
    model_research: str,
    model_forecast: str
) -> Optional[dict]:
    """
    Run EX leg with special handling for dual output (MK_RESULTS + PL_RESULTS).
    
    The EX leg is unique:
    - PT1 outputs findings (like other legs)
    - PT2 outputs BOTH MK_RESULTS and PL_RESULTS in a single response
    
    Returns: {
        "EX_RESULTS": {...},  # Main results envelope
        "MK_RESULTS": {...},  # Extracted from PT2
        "PL_RESULTS": {...}   # Extracted from PT2
    }
    """
    leg_name = "EX"
    
    # Get prompts
    prompt_pt1 = os.getenv("PROMPT_EX_PT1", "").strip()
    prompt_pt2 = os.getenv("PROMPT_EX_PT2", "").strip()
    
    if not prompt_pt1 or not prompt_pt2:
        log(f"[{leg_name}] ⏭️ SKIPPED (prompts not configured)")
        return None
    
    log_progress(f"🔬 STARTING {leg_name} LEG (DUAL OUTPUT)")
    
    # PT1: Research
    pt1_response, pt1_output = run_pt1(leg_name, prompt_pt1, qobj, model_research)
    
    if pt1_output is None:
        log(f"[{leg_name}] ❌ PT.1 FAILED - aborting leg")
        return None
    
    # Check if N/A
    if is_not_applicable_status(pt1_output.get("status", "")):
        log(f"[{leg_name}] ℹ️ PT.1 returned NOT_APPLICABLE - skipping PT.2")
        return None
    
    # PT2: Forecast (expecting dual output)
    log(f"[{leg_name}] PT.2 will output BOTH MK_RESULTS and PL_RESULTS")
    
    q_for_leg = augment_question_object(qobj)
    user_lines = [
        "QUESTION_OBJECT:",
        json.dumps(q_for_leg, ensure_ascii=False, indent=2),
        "",
        f"{leg_name}_FINDINGS:",
        json.dumps(pt1_output, ensure_ascii=False, indent=2)
    ]
    user_payload = "\n".join(user_lines)
    
    response = ""
    mk_results = None
    pl_results = None
    start_time = time.time()
    
    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = time.time() - start_time
        if elapsed > MAX_TOTAL_TIME_PT2:
            log(f"[{leg_name} PT.2] ⏱️ TIMEOUT - exceeded {MAX_TOTAL_TIME_PT2}s")
            break
        
        remaining = MAX_TOTAL_TIME_PT2 - elapsed
        
        # Call LLM
        response = call_llm(
            model_forecast,
            prompt_pt2,
            user_payload,
            max_tokens=48000,
            timeout=min(TIMEOUT_PT2_FORECAST, int(remaining))
        )
        log(f"[{leg_name} PT.2] 📥 Received {len(response)} chars from model (attempt {attempt}/{MAX_RETRIES})")
        
        # Extract JSON blocks
        blocks = extract_json_blocks(response, leg_name)
        
        # Look for a block containing both MK_RESULTS and PL_RESULTS
        for raw in blocks:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    # Check for both keys
                    has_mk = "MK_RESULTS" in obj
                    has_pl = "PL_RESULTS" in obj
                    
                    if has_mk and has_pl:
                        mk_results = {"MK_RESULTS": obj["MK_RESULTS"]}
                        pl_results = {"PL_RESULTS": obj["PL_RESULTS"]}
                        log(f"[{leg_name} PT.2] ✅ Found both MK_RESULTS and PL_RESULTS")
                        break
                    elif has_mk:
                        log(f"[{leg_name} PT.2] ⚠️ Found MK_RESULTS but missing PL_RESULTS")
                    elif has_pl:
                        log(f"[{leg_name} PT.2] ⚠️ Found PL_RESULTS but missing MK_RESULTS")
            except Exception as e:
                log(f"[{leg_name} PT.2] ⚠️ JSON parse error: {e}")
                continue
        
        # Success
        if mk_results and pl_results:
            break
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[{leg_name} PT.2] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    # Save response
    save_response(f"{leg_name.lower()}_pt2_full.txt", response)
    
    # Check results
    if not mk_results or not pl_results:
        log(f"[{leg_name}] ❌ PT.2 FAILED - did not get both MK_RESULTS and PL_RESULTS")
        return None
    
    # Normalize both results
    mk_normalized = normalize_leg_results(mk_results, "MK")
    pl_normalized = normalize_leg_results(pl_results, "PL")
    
    # Combine into final structure
    final_results = {
        "EX_RESULTS": {},  # Empty envelope (not used but keeps structure consistent)
        "MK_RESULTS": mk_normalized.get("MK_RESULTS", {}),
        "PL_RESULTS": pl_normalized.get("PL_RESULTS", {})
    }
    
    log(f"[{leg_name}] ✅ LEG COMPLETE (dual output)")
    return final_results

# ========================================
# RUN ALL METHODS FOR A QUESTION
# ========================================

def run_all_methods(qobj: dict) -> Dict[str, dict]:
    """
    Run all forecasting methods (KM, BD, EX) for a single question.
    
    Returns: {
        "KM": {...} or None,
        "BD": {...} or None,
        "EX": {...} or None  # Contains MK_RESULTS + PL_RESULTS
    }
    """
    log_progress(f"🎯 RUNNING ALL METHODS FOR Q {qobj.get('question_id')}")
    
    log(f"[CONFIG] Research model: {MODEL_RESEARCH}")
    log(f"[CONFIG] Forecast model: {MODEL_FORECAST}")
    if MODEL_BACKUP:
        log(f"[CONFIG] Backup model: {MODEL_BACKUP}")
    
    # Get prompts from environment
    prompt_km_pt1 = os.getenv("PROMPT_KM_PT1", "").strip()
    prompt_km_pt2 = os.getenv("PROMPT_KM_PT2", "").strip()
    prompt_bd_pt1 = os.getenv("PROMPT_BD_PT1", "").strip()
    prompt_bd_pt2 = os.getenv("PROMPT_BD_PT2", "").strip()
    
    results = {}
    
    # Run KM leg
    km_result = run_leg("KM", prompt_km_pt1, prompt_km_pt2, qobj, MODEL_RESEARCH, MODEL_FORECAST)
    if km_result:
        results["KM"] = km_result
    
    # Run BD leg
    bd_result = run_leg("BD", prompt_bd_pt1, prompt_bd_pt2, qobj, MODEL_RESEARCH, MODEL_FORECAST)
    if bd_result:
        results["BD"] = bd_result
    
    # Run EX leg (special case)
    ex_result = run_ex_leg(qobj, MODEL_RESEARCH, MODEL_FORECAST)
    if ex_result:
        results["EX"] = ex_result
    
    # Log summary
    num_methods = len(results)
    log_progress(f"📊 METHOD SUMMARY: {num_methods}/3 completed")
    log(f"[INFO] Successful methods: {', '.join(results.keys())}")
    
    return results

# ========================================
# END OF METHOD_RUNNER
# ========================================
