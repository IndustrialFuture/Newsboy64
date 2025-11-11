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
    PANSHUL_BOT_AVAILABLE = True
    log("[PANSHUL] ✅ Bot modules imported successfully")
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
        "serper": bool(os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY")),
        "asknews": bool(os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")),
        "gemini": bool(os.getenv("GEMINI_API_KEY"))
    }
    
    log("[PANSHUL] API availability check:")
    log(f"  OpenRouter: {'✅' if available['openrouter'] else '❌'}")
    log(f"  Serper (Google Search): {'✅' if available['serper'] else '⚠️ Using MODEL_RS fallback'}")
    log(f"  AskNews: {'✅' if available['asknews'] else '⚠️ Using MODEL_RS fallback'}")
    log(f"  Gemini: {'✅' if available['gemini'] else '❌'}")
    
    if not available['openrouter']:
        log("[FATAL] OpenRouter API key required for Panshul bot")
        raise ValueError("OPENROUTER_API_KEY not set")
    
    return available

# ========================================
# QUESTION OBJECT CREATION
# ========================================

def create_question_dict(qobj: dict) -> dict:
    """
    Convert our question format to Panshul's expected dictionary format.
    Panshul's bot expects specific fields like 'title', 'resolution_criteria', etc.
    """
    q_type = qobj.get("question_type", "binary").lower()
    
    # Build the base dict with all fields Panshul expects
    base_dict = {
        "title": qobj.get("question_text", ""),  # Panshul uses 'title' not 'question_text'
        "question_text": qobj.get("question_text", ""),
        "resolution_criteria": qobj.get("resolution_criteria", "Standard resolution."),
        "fine_print": qobj.get("fine_print", ""),
        "description": qobj.get("background_info", qobj.get("description", "")),
        "id_of_post": str(qobj.get("question_id", "unknown")),
        "id_of_question": str(qobj.get("question_id", "unknown")),
        "page_url": "",
        "api_json": {},
        "question_type": q_type,
        "resolution_date": qobj.get("horizon_utc", "")
    }
    
    if q_type == "binary":
        return base_dict
    
    elif q_type == "multiple_choice":
        base_dict["options"] = qobj.get("options", [])
        return base_dict
    
    elif q_type in ("numeric", "discrete"):
        qrange = qobj.get("range") or qobj.get("numeric_range") or {}
        base_dict["lower_bound"] = qrange.get("min", 0)
        base_dict["upper_bound"] = qrange.get("max", 100)
        base_dict["units"] = qrange.get("units", "")
        return base_dict
    
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
        # Extract probability - handle both object and dict formats
        if hasattr(panshul_result, 'prediction_value'):
            p_yes = float(panshul_result.prediction_value)
        elif isinstance(panshul_result, dict):
            p_yes = float(panshul_result.get('prediction_value', 0.5))
        else:
            p_yes = float(panshul_result)
        
        p_yes = max(0.01, min(0.99, p_yes))  # Clamp to [0.01, 0.99]
        p_no = 1.0 - p_yes
        
        # Get reasoning if available
        if hasattr(panshul_result, 'reasoning'):
            reasoning = panshul_result.reasoning
        elif isinstance(panshul_result, dict):
            reasoning = panshul_result.get('reasoning', 'Multi-model ensemble forecast')
        else:
            reasoning = 'Multi-model ensemble forecast'
        
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
    """
    try:
        options = qobj.get("options", [])
        
        # Extract predicted options
        if hasattr(panshul_result, 'prediction_value'):
            pred_value = panshul_result.prediction_value
        elif isinstance(panshul_result, dict):
            pred_value = panshul_result.get('prediction_value', {})
        else:
            pred_value = panshul_result
        
        # Handle different formats
        if hasattr(pred_value, 'predicted_options'):
            predicted_options = pred_value.predicted_options
        elif isinstance(pred_value, dict) and 'predicted_options' in pred_value:
            predicted_options = pred_value['predicted_options']
        else:
            # Fallback: uniform distribution
            predicted_options = [{"option_name": opt, "probability": 1.0/len(options)} for opt in options]
        
        # Build candidate probs
        candidate_probs = []
        for opt in options:
            prob = 0.0
            for pred_opt in predicted_options:
                opt_name = pred_opt.get('option_name') if isinstance(pred_opt, dict) else pred_opt.option_name
                opt_prob = pred_opt.get('probability') if isinstance(pred_opt, dict) else pred_opt.probability
                if opt_name == opt:
                    prob = float(opt_prob)
                    break
            candidate_probs.append({"label": opt, "p": prob})
        
        # Normalize
        total = sum(item["p"] for item in candidate_probs)
        if total > 0:
            for item in candidate_probs:
                item["p"] /= total
        
        # Get reasoning
        if hasattr(panshul_result, 'reasoning'):
            reasoning = panshul_result.reasoning
        elif isinstance(panshul_result, dict):
            reasoning = panshul_result.get('reasoning', 'Multi-model ensemble forecast')
        else:
            reasoning = 'Multi-model ensemble forecast'
        
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
    """
    try:
        # Extract distribution
        if hasattr(panshul_result, 'prediction_value'):
            dist = panshul_result.prediction_value
        elif isinstance(panshul_result, dict):
            dist = panshul_result.get('prediction_value', {})
        else:
            dist = panshul_result
        
        # Extract percentiles
        if hasattr(dist, 'percentiles'):
            percentiles = dist.percentiles
        elif isinstance(dist, dict) and 'percentiles' in dist:
            percentiles = dist['percentiles']
        else:
            percentiles = []
        
        # Calculate mean from percentiles
        p50 = None
        p10 = None
        p90 = None
        
        for p in percentiles:
            p_val = p.get('percentile') if isinstance(p, dict) else p.percentile
            val = p.get('value') if isinstance(p, dict) else p.value
            
            if abs(p_val - 0.5) < 0.01:
                p50 = val
            elif abs(p_val - 0.1) < 0.01:
                p10 = val
            elif abs(p_val - 0.9) < 0.01:
                p90 = val
        
        # Use median as mean approximation
        qrange = qobj.get("range") or {}
        rmin = qrange.get("min", 0)
        rmax = qrange.get("max", 100)
        mean = p50 if p50 is not None else (rmin + rmax) / 2
        
        # Estimate SD
        if p10 is not None and p90 is not None:
            sd = (p90 - p10) / 2.56
            interval = [p10, p90]
        else:
            sd = (rmax - rmin) / 6
            interval = [mean - sd, mean + sd]
        
        # Get reasoning
        if hasattr(panshul_result, 'reasoning'):
            reasoning = panshul_result.reasoning
        elif isinstance(panshul_result, dict):
            reasoning = panshul_result.get('reasoning', 'Multi-model ensemble forecast')
        else:
            reasoning = 'Multi-model ensemble forecast'
        
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
    
    # Create question dict (not a class instance)
    question_dict = create_question_dict(qobj)
    
    # Run appropriate forecast function
    if q_type == "binary":
        result = await binary_forecast(question_dict)
        translated = translate_panshul_binary_output(result, qobj)
    
    elif q_type == "multiple_choice":
        result = await multiple_choice_forecast(question_dict)
        translated = translate_panshul_mc_output(result, qobj)
    
    elif q_type in ("numeric", "discrete"):
        result = await numeric_forecast(question_dict)
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
