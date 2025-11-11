#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Stage 4: Veo Script Generation (STUB)
Generates video scripts from forecasts.
Veo API integration will be added in Phase 2.
"""

import os
import json
from typing import List, Dict, Optional
from utils import (
    log, log_progress, save_response, call_llm,
    extract_json_blocks, is_retryable_error,
    MAX_RETRIES, RETRY_DELAYS
)
import time

# Get model and prompt from environment
MODEL_VEO = os.getenv("MODEL_FC", "").strip()  # Use forecast model for now
PROMPT_REPORTER = os.getenv("PROMPT_REPORTER", "").strip()

# ========================================
# GENERATE VEO SCRIPTS
# ========================================

def generate_veo_scripts(forecasts: List[dict]) -> Optional[dict]:
    """
    Generate Veo video scripts from forecasts.
    
    Input: List of final forecast objects
    Output: {"Q1_script": "...", "Q2_script": "...", "Q3_script": "..."}
    """
    if not PROMPT_REPORTER:
        log("[VEO] ⏭️ SKIPPED (PROMPT_REPORTER not configured)")
        # Create stub scripts
        stub_scripts = {}
        for forecast in forecasts:
            q_id = forecast.get("metadata", {}).get("question_id", "unknown")
            stub_scripts[f"{q_id}_script"] = f"[STUB] Video script for {q_id} would go here."
        return stub_scripts
    
    log_progress("🎬 GENERATING VEO SCRIPTS")
    
    if not MODEL_VEO:
        log("[VEO] ⚠️ MODEL_FC not set, using stub mode")
        stub_scripts = {}
        for forecast in forecasts:
            q_id = forecast.get("metadata", {}).get("question_id", "unknown")
            stub_scripts[f"{q_id}_script"] = f"[STUB] Video script for {q_id} would go here."
        return stub_scripts
    
    # Build prompt payload
    user_payload = "FORECASTS:\n" + json.dumps(forecasts, indent=2)
    
    scripts = None
    
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
            block = max(blocks, key=len)
            try:
                scripts = json.loads(block)
                log(f"[VEO] ✅ Valid scripts extracted ({len(block)} chars)")
                break
            except Exception as e:
                log(f"[VEO] ⚠️ JSON parse error: {e}")
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[VEO] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    if scripts is None:
        log(f"[VEO] ❌ FAILED after {MAX_RETRIES} attempts - using stub")
        # Fallback to stub
        scripts = {}
        for forecast in forecasts:
            q_id = forecast.get("metadata", {}).get("question_id", "unknown")
            scripts[f"{q_id}_script"] = f"[STUB] Video script for {q_id} would go here."
    
    # Save
    save_response("veo_scripts_full.txt", response if 'response' in locals() else "")
    save_response("veo_scripts.json", json.dumps(scripts, indent=2))
    
    return scripts

# ========================================
# RUN STAGE 4
# ========================================

def run_stage4(forecasts: List[dict]) -> Optional[dict]:
    """
    Run Stage 4: Generate Veo scripts from all forecasts.
    
    Input: List of final forecast objects [forecast1, forecast2, forecast3]
    Output: {"Q1_script": "...", "Q2_script": "...", "Q3_script": "..."}
    """
    log_progress("🎬 STARTING STAGE 4: VEO SCRIPT GENERATION")
    
    scripts = generate_veo_scripts(forecasts)
    
    if scripts:
        log_progress(f"✅ STAGE 4 COMPLETE: Generated {len(scripts)} scripts")
    else:
        log("[STAGE4] ⚠️ No scripts generated")
    
    return scripts

# ========================================
# STUB: VEO API INTEGRATION (Phase 2)
# ========================================

def call_veo_api(script: str) -> Optional[str]:
    """
    STUB: Call Veo API to generate video from script.
    
    This will be implemented in Phase 2.
    For now, just returns a placeholder.
    """
    log("[VEO API] 🚧 STUB - Veo API integration not yet implemented")
    return None

def generate_videos(scripts: dict) -> Dict[str, str]:
    """
    STUB: Generate videos from scripts using Veo API.
    
    This will be implemented in Phase 2.
    For now, just returns placeholders.
    """
    log("[VEO API] 🚧 STUB - Video generation not yet implemented")
    
    videos = {}
    for script_id, script_text in scripts.items():
        video_id = script_id.replace("_script", "_video")
        videos[video_id] = "[PLACEHOLDER] Video would be generated here"
    
    return videos

# ========================================
# END OF STAGE4_VEO
# ========================================
