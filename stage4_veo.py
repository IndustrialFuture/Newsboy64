#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Stage 4: Veo Script Generation
Generates video scripts from forecasts using PROMPT_REPORTER.
"""

import os
import json
import time
from typing import Dict, List, Optional
from utils import (
    log, log_progress, save_response, call_llm,
    extract_json_blocks, is_retryable_error,
    MAX_RETRIES, RETRY_DELAYS
)

# Get model and prompt from environment
MODEL_VEO = os.getenv("MODEL_FC", "").strip()  # Use forecast model for Veo
PROMPT_REPORTER = os.getenv("PROMPT_REPORTER", "").strip()

# ========================================
# GENERATE VEO SCRIPT FOR A FORECAST
# ========================================

def generate_veo_script(forecast: dict) -> Optional[dict]:
    """
    Generate a Veo video script from a forecast.
    
    Input:
    - forecast: Final combined forecast with metadata
    
    Returns: Veo script JSON or None if failed
    """
    if not PROMPT_REPORTER:
        log("[VEO] ⏭️ SKIPPED (PROMPT_REPORTER not configured)")
        return None
    
    if not MODEL_VEO:
        log("[VEO] ⏭️ SKIPPED (MODEL_FC not configured)")
        return None
    
    q_id = forecast.get("metadata", {}).get("question_id", "unknown")
    log(f"[VEO] Generating script for Q {q_id}")
    
    # Build user payload
    user_payload = json.dumps(forecast, indent=2)
    
    script = None
    response = ""
    
    for attempt in range(1, MAX_RETRIES + 1):
        response = call_llm(
            MODEL_VEO,
            PROMPT_REPORTER,
            user_payload,
            max_tokens=16000,
            timeout=180
        )
        log(f"[VEO] 📥 Received {len(response)} chars from model (attempt {attempt}/{MAX_RETRIES})")
        
        # Extract JSON
        blocks = extract_json_blocks(response, "VEO")
        
        if blocks:
            # Take the longest block
            block = max(blocks, key=len)
            try:
                script = json.loads(block)
                log(f"[VEO] ✅ Valid script extracted ({len(block)} chars)")
                break
            except Exception as e:
                log(f"[VEO] ⚠️ JSON parse error: {e}")
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[VEO] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    if script is None:
        log(f"[VEO] ❌ FAILED after {MAX_RETRIES} attempts")
        save_response(f"veo_script_{q_id}_full.txt", response)
        return None
    
    # Save script
    save_response(f"veo_script_{q_id}.json", json.dumps(script, indent=2))
    save_response(f"veo_script_{q_id}_full.txt", response)
    
    return script

# ========================================
# RUN VEO GENERATION FOR ALL FORECASTS
# ========================================

def run_stage4(forecasts: List[dict]) -> Optional[Dict[str, dict]]:
    """
    Run Stage 4: Generate Veo scripts for all forecasts.
    
    Input:
    - forecasts: List of final combined forecasts
    
    Returns: Dict mapping question IDs to Veo scripts, or None if failed
    """
    log_progress("🎬 STARTING STAGE 4: VEO SCRIPT GENERATION")
    
    if not PROMPT_REPORTER:
        log("[VEO] ⏭️ SKIPPED (PROMPT_REPORTER not configured)")
        return {"skipped": "PROMPT_REPORTER not configured"}
    
    if not forecasts:
        log("[VEO] ⚠️ No forecasts provided")
        return None
    
    scripts = {}
    
    for forecast in forecasts:
        q_id = forecast.get("metadata", {}).get("question_id", "unknown")
        
        script = generate_veo_script(forecast)
        
        if script:
            scripts[q_id] = script
        else:
            log(f"[VEO] ⚠️ Failed to generate script for Q {q_id}")
    
    log_progress(f"✅ STAGE 4 COMPLETE: {len(scripts)} scripts generated")
    
    return scripts if scripts else None

# ========================================
# END OF STAGE4_VEO
# ========================================
