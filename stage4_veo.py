#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Stage 4: Veo Video Generation
Generates 5-shot news satire videos from forecasts.

Flow:
1. PROMPT_REPORTER1: Forecast → 5-sentence script
2. PROMPT_REPORTER2: Script → Veo-formatted prompts with visuals
3. Generate each shot with Veo (with retries + alternates)
4. Concatenate shots into final video
"""

import os
import json
import time
import subprocess
from typing import Dict, List, Optional
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image

from utils import (
    log, log_progress, save_response, call_llm,
    extract_json_blocks, is_retryable_error,
    MAX_RETRIES, RETRY_DELAYS, set_current_question
)

# Get configuration from environment
MODEL_REPORTER = os.getenv("MODEL_FC", "").strip()
PROMPT_REPORTER1 = os.getenv("PROMPT_REPORTER1", "").strip()
PROMPT_REPORTER2 = os.getenv("PROMPT_REPORTER2", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Veo model - CORRECT model name from docs
VEO_MODEL = "veo-3.1-generate-preview"

# Reference image path (in repo)
REFERENCE_IMAGE_PATH = "Diane-Medium.png"

# Initialize Gemini client
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# ========================================
# STEP 1: GENERATE SCRIPT (REPORTER1)
# ========================================

def generate_script(forecast: dict) -> Optional[dict]:
    """
    Use PROMPT_REPORTER1 to convert forecast into 5-sentence news script.
    
    Input: forecast (final_payload.json)
    Output: {
        "question_id": "Q1",
        "sentences": [
            {"number": 1, "text": "...", "shot_type": "anchor_lede"},
            {"number": 2, "text": "...", "shot_type": "anchor_vo"},
            {"number": 3, "text": "...", "shot_type": "anchor_vo"},
            {"number": 4, "text": "...", "shot_type": "broll"},
            {"number": 5, "text": "...", "shot_type": "anchor_final"}
        ]
    }
    """
    if not PROMPT_REPORTER1:
        log("[REPORTER1] ⏭️ SKIPPED (PROMPT_REPORTER1 not configured)")
        return None
    
    q_id = forecast.get("metadata", {}).get("question_id", "unknown")
    log(f"[REPORTER1] Generating script for Q {q_id}")
    
    user_payload = json.dumps(forecast, indent=2)
    
    script = None
    response = ""
    
    for attempt in range(1, MAX_RETRIES + 1):
        response = call_llm(
            MODEL_REPORTER,
            PROMPT_REPORTER1,
            user_payload,
            max_tokens=8000,
            timeout=120
        )
        log(f"[REPORTER1] 📥 Received {len(response)} chars (attempt {attempt}/{MAX_RETRIES})")
        
        blocks = extract_json_blocks(response, "REPORTER1")
        
        if blocks:
            block = max(blocks, key=len)
            try:
                script = json.loads(block)
                log(f"[REPORTER1] ✅ Valid script extracted")
                break
            except Exception as e:
                log(f"[REPORTER1] ⚠️ JSON parse error: {e}")
        
        if is_retryable_error(response) and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[REPORTER1] ⚠️ Retrying in {delay}s...")
            time.sleep(delay)
            continue
    
    if script is None:
        log(f"[REPORTER1] ❌ FAILED after {MAX_RETRIES} attempts")
        output_path = os.path.join("out", f"reporter1_q{q_id}_full.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)
        return None
    
    # Save script
    output_path = os.path.join("out", f"script_q{q_id}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(script, f, indent=2)
    
    log(f"[REPORTER1] 💾 Saved to {output_path}")
    
    return script

# ========================================
# STEP 2: GENERATE VEO PROMPTS (REPORTER2)
# ========================================

def generate_veo_prompts(script: dict) -> Optional[dict]:
    """
    Use PROMPT_REPORTER2 to convert script into Veo-formatted prompts.
    
    Input: script from REPORTER1
    Output: {
        "question_id": "Q1",
        "shots": [
            {
                "shot": 1,
                "vo_text": "Lede sentence 15-20 words",
                "use_reference_image": true,
                "visual_prompts": [
                    {
                        "version": "primary",
                        "prompt": "Full Veo prompt with CONTINUITY LOCK, etc."
                    },
                    {
                        "version": "alternate_1",
                        "prompt": "Sanitized version with 'White House official' instead of 'Trump'"
                    }
                ]
            },
            # ... shots 2-5
        ]
    }
    """
    if not PROMPT_REPORTER2:
        log("[REPORTER2] ⏭️ SKIPPED (PROMPT_REPORTER2 not configured)")
        return None
    
    q_id = script.get("question_id", "unknown")
    log(f"[REPORTER2] Generating Veo prompts for Q {q_id}")
    
    user_payload = json.dumps(script, indent=2)
    
    veo_prompts = None
    response = ""
    
    for attempt in range(1, MAX_RETRIES + 1):
        response = call_llm(
            MODEL_REPORTER,
            PROMPT_REPORTER2,
            user_payload,
            max_tokens=16000,
            timeout=180
        )
        log(f"[REPORTER2] 📥 Received {len(response)} chars (attempt {attempt}/{MAX_RETRIES})")
        
        blocks = extract_json_blocks(response, "REPORTER2")
        
        if blocks:
            block = max(blocks, key=len)
            try:
                veo_prompts = json.loads(block)
                log(f"[REPORTER2] ✅ Valid Veo prompts extracted")
                break
            except Exception as e:
                log(f"[REPORTER2] ⚠️ JSON parse error: {e}")
        
        if is_retryable_error(response) and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[REPORTER2] ⚠️ Retrying in {delay}s...")
            time.sleep(delay)
            continue
    
    if veo_prompts is None:
        log(f"[REPORTER2] ❌ FAILED after {MAX_RETRIES} attempts")
        output_path = os.path.join("out", f"reporter2_q{q_id}_full.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)
        return None
    
    # Save Veo prompts
    output_path = os.path.join("out", f"veo_prompts_q{q_id}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(veo_prompts, f, indent=2)
    
    log(f"[REPORTER2] 💾 Saved to {output_path}")
    
    return veo_prompts

# ========================================
# STEP 3: LOAD REFERENCE IMAGE
# ========================================

def load_reference_image() -> Optional[types.VideoGenerationReferenceImage]:
    """
    Load reference image as VideoGenerationReferenceImage.
    Returns: VideoGenerationReferenceImage object for consistent character appearance
    """
    if not os.path.exists(REFERENCE_IMAGE_PATH):
        log(f"[VEO] ❌ Reference image not found: {REFERENCE_IMAGE_PATH}")
        return None
    
    try:
        log(f"[VEO] Loading reference image: {REFERENCE_IMAGE_PATH}")
        pil_image = Image.open(REFERENCE_IMAGE_PATH)
        log(f"[VEO] ✅ Loaded image: {pil_image.size} {pil_image.mode}")
        
        # Create VideoGenerationReferenceImage for character consistency
        anchor_reference = types.VideoGenerationReferenceImage(
            image=pil_image,
            reference_type="asset"  # Preserves the person's appearance
        )
        log(f"[VEO] ✅ Created reference image object")
        
        return anchor_reference
    except Exception as e:
        log(f"[VEO] ❌ Failed to load image: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================
# STEP 4: GENERATE SINGLE SHOT WITH RETRIES
# ========================================

def generate_shot_with_retries(shot_data: dict, anchor_reference: Optional[types.VideoGenerationReferenceImage], q_id: str) -> Optional[str]:
    """
    Generate a single shot with retry logic and alternate prompts.
    
    Returns: Path to generated MP4 file, or None if all attempts failed
    """
    shot_num = shot_data.get("shot", 0)
    visual_prompts = shot_data.get("visual_prompts", [])
    use_reference = shot_data.get("use_reference_image", False)
    
    if not visual_prompts:
        log(f"[VEO] ⚠️ Shot {shot_num} has no visual prompts - skipping")
        return None
    
    if not client:
        log(f"[VEO] ❌ Gemini client not initialized")
        return None
    
    log(f"[VEO] Generating shot {shot_num} for Q {q_id}")
    
    # Try each prompt version
    for prompt_data in visual_prompts:
        version = prompt_data.get("version", "unknown")
        prompt_text = prompt_data.get("prompt", "")
        
        if not prompt_text:
            continue
        
        log(f"[VEO] Trying shot {shot_num} - version '{version}'")
        
        # Try this version up to 2 times
        for attempt in range(1, 3):
            try:
                log(f"[VEO] Submitting generation request for shot {shot_num}...")
                
                # Use reference image if requested AND available
                if use_reference and anchor_reference:
                    log(f"[VEO] Using anchor reference image for shot {shot_num}")
                    operation = client.models.generate_videos(
                        model=VEO_MODEL,
                        prompt=prompt_text,
                        config=types.GenerateVideosConfig(
                            reference_images=[anchor_reference],
                        ),
                    )
                else:
                    # Text-to-video only
                    operation = client.models.generate_videos(
                        model=VEO_MODEL,
                        prompt=prompt_text,
                    )
                
                # Poll for completion (from docs)
                log(f"[VEO] ⏳ Waiting for shot {shot_num} to generate (2-5 minutes)...")
                
                max_wait_time = 600  # 10 minutes max
                elapsed = 0
                poll_interval = 10  # Check every 10 seconds
                
                while not operation.done:
                    if elapsed >= max_wait_time:
                        log(f"[VEO] ⏱️ Shot {shot_num} timed out after {max_wait_time}s")
                        break
                    
                    time.sleep(poll_interval)
                    elapsed += poll_interval
                    log(f"[VEO] ⏳ Still waiting... ({elapsed}s elapsed)")
                    
                    # Refresh operation status
                    operation = client.operations.get(operation)
                
                if not operation.done:
                    log(f"[VEO] ⏱️ Shot {shot_num} timed out")
                    continue
                
                # Download the generated video
                generated_video = operation.response.generated_videos[0]
                
                output_filename = f"shot_{shot_num}_q{q_id}.mp4"
                output_path = os.path.join("out", output_filename)
                
                # Download and save
                client.files.download(file=generated_video.video)
                generated_video.video.save(output_path)
                
                log(f"[VEO] ✅ Shot {shot_num} saved to {output_path}")
                return output_path
                
            except Exception as e:
                log(f"[VEO] ⚠️ Shot {shot_num} attempt {attempt}/2 failed: {e}")
                import traceback
                traceback.print_exc()
                if attempt < 2:
                    time.sleep(5)
                    continue
        
        log(f"[VEO] ⚠️ Shot {shot_num} version '{version}' failed - trying next version")
    
    log(f"[VEO] ❌ Shot {shot_num} - all versions failed")
    return None

# ========================================
# STEP 5: CONCATENATE VIDEOS
# ========================================

def concatenate_videos(shot_files: List[str], output_filename: str) -> Optional[str]:
    """
    Concatenate multiple MP4 files into one video using ffmpeg.
    
    Returns: Path to final video or None if failed
    """
    if not shot_files:
        log("[VEO] ❌ No shots to concatenate")
        return None
    
    log(f"[VEO] Concatenating {len(shot_files)} shots...")
    
    try:
        # Create concat list file
        concat_list_path = os.path.join("out", "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for shot_file in shot_files:
                # Use absolute path or relative from out/ directory
                f.write(f"file '{os.path.basename(shot_file)}'\n")
        
        # Run ffmpeg
        output_path = os.path.join("out", output_filename)
        result = subprocess.run([
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', 'concat_list.txt',
            '-c', 'copy',
            output_filename
        ], cwd='out', capture_output=True, text=True)
        
        if result.returncode != 0:
            log(f"[VEO] ❌ ffmpeg failed: {result.stderr}")
            return None
        
        log(f"[VEO] ✅ Final video saved to {output_path}")
        
        # Clean up concat list
        os.remove(concat_list_path)
        
        return output_path
    
    except Exception as e:
        log(f"[VEO] ❌ Concatenation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================
# MAIN: RUN COMPLETE STAGE 4 FOR ONE QUESTION
# ========================================

def run_stage4_for_question(forecast: dict) -> Optional[str]:
    """
    Complete Veo video generation pipeline for one question.
    
    Returns: Path to final video MP4, or None if failed
    """
    q_id = forecast.get("metadata", {}).get("question_id", "unknown")
    
    log_progress(f"🎬 STARTING STAGE 4 FOR Q {q_id}")
    
    # Step 1: Generate script
    script = generate_script(forecast)
    if not script:
        return None
    
    # Step 2: Generate Veo prompts
    veo_prompts = generate_veo_prompts(script)
    if not veo_prompts:
        return None
    
    # Step 3: Load reference image (once per question)
    anchor_reference = load_reference_image()
    if not anchor_reference:
        log("[VEO] ⚠️ Proceeding without reference image - anchor may be inconsistent")
    
    # Step 4: Generate each shot
    shots = veo_prompts.get("shots", [])
    generated_shots = []
    
    for shot_data in shots:
        shot_file = generate_shot_with_retries(shot_data, anchor_reference, q_id)
        if shot_file:
            generated_shots.append(shot_file)
        else:
            log(f"[VEO] ⚠️ Skipping failed shot {shot_data.get('shot')}")
    
    if not generated_shots:
        log(f"[VEO] ❌ No shots generated for Q {q_id}")
        return None
    
    # Step 5: Concatenate shots
    final_video = concatenate_videos(generated_shots, f"video_q{q_id}.mp4")
    
    if final_video:
        log_progress(f"✅ STAGE 4 COMPLETE FOR Q {q_id}: {final_video}")
    
    return final_video

# ========================================
# RUN STAGE 4 FOR ALL FORECASTS
# ========================================

def run_stage4(forecasts: List[dict]) -> Optional[Dict[str, str]]:
    """
    Run Stage 4 for all forecasts sequentially.
    
    Returns: Dict mapping question IDs to video paths
    """
    log_progress("🎬 STARTING STAGE 4: VEO VIDEO GENERATION")
    
    # Reset to root directory
    set_current_question(None)
    
    if not PROMPT_REPORTER1 or not PROMPT_REPORTER2:
        log("[VEO] ⏭️ SKIPPED (PROMPT_REPORTER1 or PROMPT_REPORTER2 not configured)")
        return {"skipped": "Prompts not configured"}
    
    if not GEMINI_API_KEY:
        log("[VEO] ⏭️ SKIPPED (GEMINI_API_KEY not configured)")
        return {"skipped": "Gemini API key not configured"}
    
    if not forecasts:
        log("[VEO] ⚠️ No forecasts provided")
        return None
    
    videos = {}
    
    for forecast in forecasts:
        q_id = forecast.get("metadata", {}).get("question_id", "unknown")
        
        video_path = run_stage4_for_question(forecast)
        
        if video_path:
            videos[q_id] = video_path
        else:
            log(f"[VEO] ⚠️ Failed to generate video for Q {q_id}")
    
    log_progress(f"✅ STAGE 4 COMPLETE: {len(videos)} videos generated")
    
    return videos if videos else None

# ========================================
# END OF STAGE4_VEO
# ========================================

Pillow
