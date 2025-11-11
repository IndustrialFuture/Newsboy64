#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Stage 1: Question Generation
Runs RS1 → RS2 → QG chain to generate forecasting questions.
Outputs:
  - OUTPUT_A_MINIMAL_API_ARRAY: List of 3 questions
  - OUTPUT_B_EVIDENCE_PACKET: Evidence/consequences for each question
"""

import os
import json
from typing import Tuple, Optional, Dict, List
from utils import (
    log, log_progress, save_response, call_llm,
    extract_json_blocks, is_retryable_error,
    MAX_RETRIES, RETRY_DELAYS
)
import time

# Get models and prompts from environment
MODEL_QG = os.getenv("MODEL_QG", "").strip()
PROMPT_RS1 = os.getenv("PROMPT_RS1", "").strip()
PROMPT_RS2 = os.getenv("PROMPT_RS2", "").strip()
PROMPT_QG = os.getenv("PROMPT_QG", "").strip()

# Validate
if not MODEL_QG:
    log("[FATAL] MODEL_QG must be set in environment")
    raise ValueError("MODEL_QG not set")

# ========================================
# RS1: RESEARCH DISCOVERY
# ========================================

def run_rs1(topic: str, max_retries: int = MAX_RETRIES) -> Tuple[str, Optional[dict]]:
    """
    Run RS1 (Research Discovery).
    Returns: (full_response, discovery_json or None)
    """
    if not PROMPT_RS1:
        log("[RS1] ⏭️ SKIPPED (PROMPT_RS1 not configured)")
        return "", None
    
    log_progress("🔍 RUNNING RS1: RESEARCH DISCOVERY")
    
    user_payload = f"Topic: {topic}"
    response = ""
    discovery_json = None
    
    for attempt in range(1, max_retries + 1):
        response = call_llm(
            MODEL_QG,
            PROMPT_RS1,
            user_payload,
            max_tokens=16000,
            timeout=180
        )
        log(f"[RS1] 📥 Received {len(response)} chars (attempt {attempt}/{max_retries})")
        
        # Extract JSON
        blocks = extract_json_blocks(response, "RS1")
        
        for raw in blocks:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and len(obj) > 0:
                    discovery_json = obj
                    log(f"[RS1] ✅ Valid discovery JSON extracted ({len(raw)} chars)")
                    break
            except Exception:
                continue
        
        if discovery_json:
            break
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < max_retries:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[RS1] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    if discovery_json is None:
        log(f"[RS1] ❌ FAILED after {max_retries} attempts")
    
    save_response("rs1_discovery_full.txt", response)
    if discovery_json:
        save_response("rs1_discovery.json", json.dumps(discovery_json, indent=2))
    
    return response, discovery_json

# ========================================
# RS2: RESEARCH SYNTHESIS
# ========================================

def run_rs2(discovery_json: dict, max_retries: int = MAX_RETRIES) -> Tuple[str, Optional[dict]]:
    """
    Run RS2 (Research Synthesis).
    Takes discovery_json from RS1, produces synthesis_json.
    Returns: (full_response, synthesis_json or None)
    """
    if not PROMPT_RS2:
        log("[RS2] ⏭️ SKIPPED (PROMPT_RS2 not configured)")
        return "", None
    
    log_progress("🔬 RUNNING RS2: RESEARCH SYNTHESIS")
    
    user_payload = "DISCOVERY_JSON:\n" + json.dumps(discovery_json, indent=2)
    response = ""
    synthesis_json = None
    
    for attempt in range(1, max_retries + 1):
        response = call_llm(
            MODEL_QG,
            PROMPT_RS2,
            user_payload,
            max_tokens=24000,
            timeout=240
        )
        log(f"[RS2] 📥 Received {len(response)} chars (attempt {attempt}/{max_retries})")
        
        # Extract JSON
        blocks = extract_json_blocks(response, "RS2")
        
        for raw in blocks:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and len(obj) > 0:
                    synthesis_json = obj
                    log(f"[RS2] ✅ Valid synthesis JSON extracted ({len(raw)} chars)")
                    break
            except Exception:
                continue
        
        if synthesis_json:
            break
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < max_retries:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[RS2] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    if synthesis_json is None:
        log(f"[RS2] ❌ FAILED after {max_retries} attempts")
    
    save_response("rs2_synthesis_full.txt", response)
    if synthesis_json:
        save_response("rs2_synthesis.json", json.dumps(synthesis_json, indent=2))
    
    return response, synthesis_json

# ========================================
# QG: QUESTION GENERATOR
# ========================================

def run_qg(
    discovery_json: dict,
    synthesis_json: dict,
    max_questions: int = 3,
    max_retries: int = MAX_RETRIES
) -> Tuple[str, Optional[List[dict]], Optional[dict]]:
    """
    Run QG (Question Generator).
    Takes discovery + synthesis, produces:
      - OUTPUT_A_MINIMAL_API_ARRAY: list of questions
      - OUTPUT_B_EVIDENCE_PACKET: evidence/consequences
    
    Returns: (full_response, output_a or None, output_b or None)
    """
    if not PROMPT_QG:
        log("[QG] ⏭️ SKIPPED (PROMPT_QG not configured)")
        return "", None, None
    
    log_progress("❓ RUNNING QG: QUESTION GENERATION")
    
    user_lines = [
        "DISCOVERY_JSON:",
        json.dumps(discovery_json, indent=2),
        "",
        "SYNTHESIS_JSON:",
        json.dumps(synthesis_json, indent=2),
        "",
        f"Generate exactly {max_questions} forecasting questions."
    ]
    user_payload = "\n".join(user_lines)
    
    response = ""
    output_a = None
    output_b = None
    
    for attempt in range(1, max_retries + 1):
        response = call_llm(
            MODEL_QG,
            PROMPT_QG,
            user_payload,
            max_tokens=24000,
            timeout=240
        )
        log(f"[QG] 📥 Received {len(response)} chars (attempt {attempt}/{max_retries})")
        
        # Extract JSON blocks
        blocks = extract_json_blocks(response, "QG")
        
        # Look for OUTPUT_A (array) and OUTPUT_B (object)
        for raw in blocks:
            try:
                obj = json.loads(raw)
                
                # Check if it's an array (OUTPUT_A)
                if isinstance(obj, list) and len(obj) > 0:
                    # Validate it looks like questions
                    if all(isinstance(q, dict) and "question_id" in q for q in obj):
                        output_a = obj[:max_questions]  # Limit to max_questions
                        log(f"[QG] ✅ Found OUTPUT_A: {len(output_a)} questions")
                
                # Check if it's an object (OUTPUT_B)
                elif isinstance(obj, dict):
                    # Look for question IDs as keys
                    keys = list(obj.keys())
                    if keys and any(k.startswith("Q") for k in keys):
                        output_b = obj
                        log(f"[QG] ✅ Found OUTPUT_B: evidence for {len(obj)} questions")
            
            except Exception as e:
                log(f"[QG] ⚠️ JSON parse error: {e}")
                continue
        
        # Success if we have both
        if output_a and output_b:
            break
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < max_retries:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[QG] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    if output_a is None or output_b is None:
        log(f"[QG] ❌ FAILED after {max_retries} attempts")
        if output_a:
            log(f"[QG]   - OUTPUT_A: ✅ Found")
        else:
            log(f"[QG]   - OUTPUT_A: ❌ Missing")
        if output_b:
            log(f"[QG]   - OUTPUT_B: ✅ Found")
        else:
            log(f"[QG]   - OUTPUT_B: ❌ Missing")
    
    save_response("qg_full.txt", response)
    if output_a:
        save_response("qg_output_a.json", json.dumps(output_a, indent=2))
    if output_b:
        save_response("qg_output_b.json", json.dumps(output_b, indent=2))
    
    return response, output_a, output_b

# ========================================
# STAGE 1 ORCHESTRATOR
# ========================================

def run_stage1(topic: str, max_questions: int = 3) -> Optional[dict]:
    """
    Run complete Stage 1: RS1 → RS2 → QG
    
    Returns: {
        "discovery_json": {...},
        "synthesis_json": {...},
        "output_a_minimal_api_array": [...],
        "output_b_evidence_packet": {...}
    } or None if failed
    """
    log_progress(f"🚀 STARTING STAGE 1: QUESTION GENERATION")
    log(f"[STAGE1] Topic: {topic}")
    log(f"[STAGE1] Max questions: {max_questions}")
    
    # RS1: Discovery
    _, discovery_json = run_rs1(topic)
    if discovery_json is None:
        log("[STAGE1] ❌ FAILED at RS1")
        return None
    
    # RS2: Synthesis
    _, synthesis_json = run_rs2(discovery_json)
    if synthesis_json is None:
        log("[STAGE1] ❌ FAILED at RS2")
        return None
    
    # QG: Question Generation
    _, output_a, output_b = run_qg(discovery_json, synthesis_json, max_questions)
    if output_a is None or output_b is None:
        log("[STAGE1] ❌ FAILED at QG")
        return None
    
    # Package results
    result = {
        "discovery_json": discovery_json,
        "synthesis_json": synthesis_json,
        "output_a_minimal_api_array": output_a,
        "output_b_evidence_packet": output_b
    }
    
    # Save complete stage1 output
    save_response("stage1_complete.json", json.dumps(result, indent=2))
    
    log_progress(f"✅ STAGE 1 COMPLETE: Generated {len(output_a)} questions")
    
    return result

# ========================================
# END OF STAGE1_QUESTION_GEN
# ========================================
