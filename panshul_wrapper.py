#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Panshul Bot Wrapper
Wraps Panshul's bot with API fallbacks and output translation.
Handles missing optional API keys gracefully (Serper, AskNews, Gemini).
"""

import os
import sys
import json
import asyncio
from typing import Optional, Dict
from datetime import datetime, timezone

# Add Bot folder to path so we can import Panshul's modules
BOT_DIR = os.path.join(os.path.dirname(__file__), "Bot")
sys.path.insert(0, BOT_DIR)

from utils import log, log_progress, save_response, augment_question_object

# Try to import Panshul's bot modules
try:
    from Bot.forecaster import binary_forecast, multiple_choice_forecast, numeric_forecast
    from Bot.binary import BinaryQuestion
    from Bot.multiple_choice import MultipleChoiceQuestion
    from Bot.numeric import NumericQuestion
    PANSHUL_BOT_AVAILABLE = True
except ImportError as e:
    log(f"[ERROR] Failed to import Panshul's bot: {e}")
    PANSHUL_BOT_AVAILABLE = False

# ========================================
# API KEY CHECKING
# ========================================

def check_panshul_apis() -> Dict[str, bool]:
    """
    Check which optional APIs are available for Panshul's bot.
    Returns dict of API availability.
    """
    available = {
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "serper": bool(os.getenv("SERPER_API_KEY")),
        "asknews": bool(os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")),
        "gemini": bool(os.getenv("GEMINI_API_KEY"))
    }
    
    log("[PANSHUL] API availability check:")
    log(f"  OpenRouter: {'✅' if available['openrouter'] else '❌'}")
    log(f"  Serper (Google Search): {'✅' if available['serper'] else '❌ (limited search)'}")
    log(f"  AskNews: {'✅' if available['asknews'] else '❌ (skipping news API)'}")
    log(f"  Gemini: {'✅' if available['gemini'] else '❌'}")
    
    if not available['openrouter']:
        log("[FATAL] OpenRouter API key required for Panshul bot")
        raise ValueError("OPENROUTER_API_KEY not set")
    
    if not available['serper']:
        log("[WARN] Serper API key missing - Panshul bot will have limited search capability")
    
    if not available['asknews']:
        log("[INFO] AskNews API not configured - skipping (expensive API, not required)")
    
    return available

# ========================================
# QUESTION CONVERSION
# ========================================

def convert_to_panshul_question(qobj: dict):
    """
    Convert our question format to Panshul's question objects.
    Returns appropriate question object for Panshul's bot.
    """
    q_type = qobj.get("question_type", "binary").lower()
    
    if q_type == "binary":
        return BinaryQuestion(
            question_text=qobj.get("question_text", ""),
            id_of_post=str(qobj.get("question_id", "unknown")),
            id_of_question=str(qobj.get("question_id", "unknown")),
            page_url="",  # Not needed for forecasting
            api_json={}
        )
    
    elif q_type == "multiple_choice":
        return MultipleChoiceQuestion(
            question_text=qobj.get("question_text", ""),
            options=qobj.get("options", []),
            id_of_post=str(qobj.get("question_id", "unknown")),
            id_of_question=str(qobj.get("question_id", "unknown")),
            page_url="",
            api_json={}
        )
    
    elif q_type in ("numeric", "discrete"):
        qrange = qobj.get("range") or qobj.get("numeric_range") or {}
        return NumericQuestion(
            question_text=qobj.get("question_text", ""),
            lower_bound=qrange.get("min", 0),
            upper_bound=qrange.get("max", 100),
            id_of_post=str(qobj.get("question_id", "unknown")),
            id_of_question=str(qobj.get("question_id", "unknown")),
            page_url="",
            api_json={}
        )
    
    else:
        log(f"[PANSHUL] Unsupported question type: {q_type}")
        raise ValueError(f"Unsupported question type for Panshul bot: {q_type}")

# ========================================
# OUTPUT TRANSLATION
# ========================================

def translate_panshul_binary_output(panshul_result, qobj: dict) -> dict:
    """
    Translate Panshul's binary output to our unified format.
    
    Panshul returns: ReasonedPrediction with prediction_value (float)
    We need: {"PANSHUL_RESULTS": {"A": {"P_YES": x, "P_NO": y}}}
    """
    try:
        # Extract probability
        p_yes = float(panshul_result.prediction_value)
        p_yes = max(0.01, min(0.99, p_yes))  # Clamp to [0.01, 0.99]
        p_no = 1.0 - p_yes
        
        reasoning = getattr(panshul_result, 'reasoning', 'Multi-model ensemble forecast')
        
        return {
            "PANSHUL_RESULTS": {
                "A": {
                    "P_YES": p_yes,
                    "P_NO": p_no
                }
            },
            "metadata": {
                "reasoning": reasoning,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_id": qobj.get("question_id"),
                "method": "panshul_bot"
            }
        }
    except Exception as e:
        log(f"[PANSHUL] Error translating binary output: {e}")
        raise

def translate_panshul_mc_output(panshul_result, qobj: dict) -> dict:
    """
    Translate Panshul's multiple choice output to our unified format.
    
    Panshul returns: ReasonedPrediction with PredictedOptionList
    We need: {"PANSHUL_RESULTS": {"C": {"CANDIDATE_PROBS": [...]}}}
    """
    try:
        options = qobj.get("options", [])
        predicted_options = panshul_result.prediction_value.predicted_options
        
        # Build candidate probs in order
        candidate_probs = []
        for i, opt in enumerate(options):
            # Find matching predicted option
            prob = 0.0
            for pred_opt in predicted_options:
                if pred_opt.option_name == opt:
                    prob = float(pred_opt.probability)
                    break
            candidate_probs.append({"label": opt, "p": prob})
        
        # Normalize to sum to 1.0
        total = sum(item["p"] for item in candidate_probs)
        if total > 0:
            for item in candidate_probs:
                item["p"] /= total
        
        reasoning = getattr(panshul_result, 'reasoning', 'Multi-model ensemble forecast')
        
        return {
            "PANSHUL_RESULTS": {
                "C": {
                    "CANDIDATE_PROBS": candidate_probs
                }
            },
            "metadata": {
                "reasoning": reasoning,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_id": qobj.get("question_id"),
                "method": "panshul_bot"
            }
        }
    except Exception as e:
        log(f"[PANSHUL] Error translating MC output: {e}")
        raise

def translate_panshul_numeric_output(panshul_result, qobj: dict) -> dict:
    """
    Translate Panshul's numeric output to our unified format.
    
    Panshul returns: ReasonedPrediction with NumericDistribution
    We need: {"PANSHUL_RESULTS": {"B": {"MEAN": x, "SD": y, "INTERVAL": [lo, hi]}}}
    """
    try:
        dist = panshul_result.prediction_value
        
        # Extract percentiles from distribution
        # NumericDistribution has .percentiles attribute
        percentiles = dist.percentiles if hasattr(dist, 'percentiles') else []
        
        # Calculate mean from percentiles (approximate as median)
        p50 = None
        p10 = None
        p90 = None
        
        for p in percentiles:
            if abs(p.percentile - 0.5) < 0.01:
                p50 = p.value
            elif abs(p.percentile - 0.1) < 0.01:
                p10 = p.value
            elif abs(p.percentile - 0.9) < 0.01:
                p90 = p.value
        
        # Use median as mean approximation
        mean = p50 if p50 is not None else (qobj.get("range", {}).get("min", 0) + qobj.get("range", {}).get("max", 100)) / 2
        
        # Estimate SD from P10-P90 range
        if p10 is not None and p90 is not None:
            # Approximate: P10-P90 range ≈ 2.56 * SD
            sd = (p90 - p10) / 2.56
            interval = [p10, p90]
        else:
            # Fallback
            qrange = qobj.get("range") or {}
            rmin = qrange.get("min", 0)
            rmax = qrange.get("max", 100)
            sd = (rmax - rmin) / 6  # Rough estimate
            interval = [mean - sd, mean + sd]
        
        reasoning = getattr(panshul_result, 'reasoning', 'Multi-model ensemble forecast')
        
        return {
            "PANSHUL_RESULTS": {
                "B": {
                    "MEAN": mean,
                    "SD": sd,
                    "INTERVAL": interval
                }
            },
            "metadata": {
                "reasoning": reasoning,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_id": qobj.get("question_id"),
                "method": "panshul_bot"
            }
        }
    except Exception as e:
        log(f"[PANSHUL] Error translating numeric output: {e}")
        raise

# ========================================
# ASYNC WRAPPER
# ========================================

async def run_panshul_async(qobj: dict) -> dict:
    """
    Run Panshul's bot asynchronously.
    Returns translated results in unified format.
    """
    q_type = qobj.get("question_type", "binary").lower()
    
    log(f"[PANSHUL] Running forecast for Q {qobj.get('question_id')} (type: {q_type})")
    
    # Convert question
    panshul_q = convert_to_panshul_question(qobj)
    
    # Run appropriate forecast function
    if q_type == "binary":
        result = await binary_forecast(panshul_q)
        translated = translate_panshul_binary_output(result, qobj)
    
    elif q_type == "multiple_choice":
        result = await multiple_choice_forecast(panshul_q)
        translated = translate_panshul_mc_output(result, qobj)
    
    elif q_type in ("numeric", "discrete"):
        result = await numeric_forecast(panshul_q)
        translated = translate_panshul_numeric_output(result, qobj)
    
    else:
        raise ValueError(f"Unsupported question type: {q_type}")
    
    log(f"[PANSHUL] ✅ Forecast complete")
    return translated

# ========================================
# SYNCHRONOUS WRAPPER
# ========================================

def run_panshul(qobj: dict) -> Optional[dict]:
    """
    Run Panshul's bot synchronously (wraps async function).
    Returns translated results or None if failed.
    """
    if not PANSHUL_BOT_AVAILABLE:
        log("[PANSHUL] ❌ Bot not available - skipping")
        return None
    
    log_progress("🤖 RUNNING PANSHUL BOT")
    
    # Check API availability
    try:
        apis = check_panshul_apis()
    except ValueError as e:
        log(f"[PANSHUL] ❌ API check failed: {e}")
        return None
    
    # Augment question
    qobj = augment_question_object(qobj)
    
    # Run async function
    try:
        result = asyncio.run(run_panshul_async(qobj))
        
        # Save output
        save_response("panshul_full.json", json.dumps(result, indent=2))
        
        log(f"[PANSHUL] ✅ COMPLETE")
        return result
    
    except Exception as e:
        log(f"[PANSHUL] ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================
# END OF PANSHUL_WRAPPER
# ========================================
