#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polymarket API Search Module
Searches Polymarket for relevant prediction markets and scores them using Tversky similarity.
"""

import os
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime, timezone

# Import from your existing utils
from utils import log, call_llm

# ========================================
# CONFIGURATION
# ========================================

POLYMARKET_API_BASE = "https://gamma-api.polymarket.com"
POLYMARKET_TAGS = [
    "politics",
    "sports", 
    "crypto",
    "pop-culture",
    "science",
    "economics",
    "business",
    "weather",
    "international"
]

# Model for LLM calls (Claude 3.5 Sonnet via OpenRouter)
QUERY_GEN_MODEL = "anthropic/claude-3.5-sonnet"
SCORER_MODEL = "anthropic/claude-3.5-sonnet"

# Load prompts from files
def load_prompt(filename: str) -> str:
    """Load prompt from prompts/ directory"""
    try:
        filepath = os.path.join("prompts", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        log(f"[POLYMARKET] ❌ Failed to load prompt {filename}: {e}")
        return ""

PROMPT_QUERY_GEN = load_prompt("polymarket_query_gen.txt")
PROMPT_SCORER = load_prompt("polymarket_scorer.txt")

# ========================================
# STEP 1: GENERATE SEARCH QUERIES (WITH DEBUG LOGGING)
# ========================================

def generate_search_queries(question_obj: dict) -> Optional[Dict[str, any]]:
    """
    Use LLM to generate smart search queries for Polymarket.
    
    Returns: {
        "tags": ["politics", "economics"],
        "queries": ["query1", "query2", ...]
    }
    Or None if failed.
    """
    log("[POLYMARKET] 🔍 Generating search queries with Claude 3.5 Sonnet...")
    
    if not PROMPT_QUERY_GEN:
        log("[POLYMARKET] ❌ Query generation prompt not loaded")
        return None
    
    # Build user message
    user_message = "QUESTION_OBJECT:\n" + json.dumps(question_obj, indent=2, ensure_ascii=False)
    
    # Call LLM (using your existing call_llm from utils.py)
    response = call_llm(
        model=QUERY_GEN_MODEL,
        system_prompt=PROMPT_QUERY_GEN,
        user_payload=user_message,
        temperature=0.3,
        max_tokens=2000,
        timeout=60
    )
    
    if not response or len(response) < 20:
        log("[POLYMARKET] ❌ Empty or invalid response from query generator")
        return None
    
    # DEBUG: Show what we got
    log(f"[POLYMARKET] 📝 Raw response ({len(response)} chars):")
    log(f"[POLYMARKET] First 500 chars: {response[:500]}")
    
    # Parse JSON response
    try:
        # Try to extract JSON from response
        response_clean = response.strip()
        
        # Remove markdown code blocks if present
        if "```json" in response_clean:
            # Extract content between ```json and ```
            start = response_clean.find("```json") + 7
            end = response_clean.find("```", start)
            if end > start:
                response_clean = response_clean[start:end].strip()
                log("[POLYMARKET] 🔧 Extracted from ```json code block")
        elif response_clean.startswith("```"):
            # Generic code block
            lines = response_clean.split("\n")
            response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean
            log("[POLYMARKET] 🔧 Extracted from ``` code block")
        
        # Try to find JSON object if there's extra text
        if not response_clean.startswith("{"):
            # Look for first {
            start_idx = response_clean.find("{")
            if start_idx >= 0:
                response_clean = response_clean[start_idx:]
                log(f"[POLYMARKET] 🔧 Found JSON starting at position {start_idx}")
        
        # Try to find end of JSON object if there's trailing text
        if response_clean.count("{") > 0:
            depth = 0
            end_idx = -1
            for i, char in enumerate(response_clean):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            if end_idx > 0:
                response_clean = response_clean[:end_idx]
                log(f"[POLYMARKET] 🔧 Trimmed to valid JSON ({end_idx} chars)")
        
        log(f"[POLYMARKET] 🔍 Attempting to parse: {response_clean[:200]}...")
        
        result = json.loads(response_clean)
        
        # Validate structure
        if not isinstance(result, dict):
            log(f"[POLYMARKET] ❌ Response is not a dict, got {type(result)}")
            return None
        
        log(f"[POLYMARKET] 📋 Parsed JSON keys: {list(result.keys())}")
        
        if "tags" not in result or "queries" not in result:
            log(f"[POLYMARKET] ❌ Missing 'tags' or 'queries' in response")
            log(f"[POLYMARKET] Got keys: {list(result.keys())}")
            log(f"[POLYMARKET] Full result: {json.dumps(result, indent=2)[:500]}")
            return None
        
        tags = result["tags"]
        queries = result["queries"]
        
        # FIX: Convert strings to lists if needed
        if isinstance(tags, str):
            log("[POLYMARKET] 🔧 Converting tags string to list")
            tags = [tags]
            result["tags"] = tags
        
        if isinstance(queries, str):
            log("[POLYMARKET] 🔧 Converting queries string to list")
            queries = [q.strip() for q in queries.split(",")]
            result["queries"] = queries
        
        if not isinstance(tags, list) or not isinstance(queries, list):
            log(f"[POLYMARKET] ❌ Invalid types - tags: {type(tags)}, queries: {type(queries)}")
            return None
        
        if not queries:
            log("[POLYMARKET] ⚠️ No queries generated")
            return None
        
        log(f"[POLYMARKET] ✅ Generated {len(queries)} queries: {queries[:3]}...")
        log(f"[POLYMARKET] ✅ Selected tags: {tags}")
        
        return result
    
    except json.JSONDecodeError as e:
        log(f"[POLYMARKET] ❌ JSON parse error: {e}")
        log(f"[POLYMARKET] Attempted to parse: {response_clean[:500]}...")
        return None
    except Exception as e:
        log(f"[POLYMARKET] ❌ Error parsing query response: {e}")
        return None

# ========================================
# STEP 2: SEARCH POLYMARKET API
# ========================================

def search_polymarket_api(tags: List[str], queries: List[str], max_markets: int = 400) -> List[dict]:
    """
    Search Polymarket API using tags and filter by keyword queries.
    
    Args:
        tags: List of category tags to search (e.g., ["politics", "economics"])
        queries: List of keyword queries to filter results
        max_markets: Maximum markets to fetch per tag
    
    Returns:
        List of unique market dicts
    """
    log("[POLYMARKET] 🌐 Searching Polymarket API...")
    
    all_markets = []
    seen_ids = set()
    
    # First, get tag IDs from Polymarket
    tag_id_map = get_tag_ids()
    
    for tag in tags:
        tag_id = tag_id_map.get(tag.lower())
        
        if not tag_id:
            log(f"[POLYMARKET] ⚠️ Unknown tag '{tag}', searching without tag filter")
            # Fall back to untagged search
            markets = fetch_markets_untagged(queries, max_markets)
        else:
            log(f"[POLYMARKET] 🔍 Searching tag '{tag}' (ID: {tag_id})...")
            markets = fetch_markets_by_tag(tag_id, max_markets)
        
        log(f"[POLYMARKET] 📊 Retrieved {len(markets)} markets from '{tag}'")
        
        # Filter by keywords
        filtered = filter_markets_by_keywords(markets, queries)
        log(f"[POLYMARKET] 🔍 After keyword filtering: {len(filtered)} markets")
        
        # Deduplicate
        for market in filtered:
            market_id = market.get("condition_id") or market.get("id")
            if market_id and market_id not in seen_ids:
                all_markets.append(market)
                seen_ids.add(market_id)
    
    log(f"[POLYMARKET] ✅ Total unique markets found: {len(all_markets)}")
    return all_markets

def get_tag_ids() -> Dict[str, str]:
    """
    Get Polymarket tag IDs.
    Returns mapping of tag_name -> tag_id
    """
    try:
        url = f"{POLYMARKET_API_BASE}/tags"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            log(f"[POLYMARKET] ⚠️ Failed to fetch tags: HTTP {response.status_code}")
            return {}
        
        tags_data = response.json()
        
        # Build mapping
        tag_map = {}
        for tag in tags_data:
            tag_name = tag.get("label", "").lower()
            tag_id = tag.get("id")
            if tag_name and tag_id:
                tag_map[tag_name] = tag_id
        
        log(f"[POLYMARKET] 📋 Loaded {len(tag_map)} tag mappings")
        return tag_map
    
    except Exception as e:
        log(f"[POLYMARKET] ⚠️ Error fetching tags: {e}")
        return {}

def fetch_markets_by_tag(tag_id: str, max_markets: int = 400) -> List[dict]:
    """Fetch markets filtered by tag"""
    markets = []
    
    for offset in range(0, max_markets, 100):
        try:
            url = f"{POLYMARKET_API_BASE}/markets"
            params = {
                "tag_id": tag_id,
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": offset
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                log(f"[POLYMARKET] ⚠️ HTTP {response.status_code} at offset {offset}")
                break
            
            batch = response.json()
            if not batch:
                break
            
            markets.extend(batch)
            
            # Stop if we got fewer than 100 (last page)
            if len(batch) < 100:
                break
        
        except Exception as e:
            log(f"[POLYMARKET] ⚠️ Error at offset {offset}: {e}")
            break
    
    return markets

def fetch_markets_untagged(queries: List[str], max_markets: int = 400) -> List[dict]:
    """Fetch markets without tag filter (fallback)"""
    markets = []
    
    for offset in range(0, max_markets, 100):
        try:
            url = f"{POLYMARKET_API_BASE}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": offset
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                log(f"[POLYMARKET] ⚠️ HTTP {response.status_code} at offset {offset}")
                break
            
            batch = response.json()
            if not batch:
                break
            
            markets.extend(batch)
            
            if len(batch) < 100:
                break
        
        except Exception as e:
            log(f"[POLYMARKET] ⚠️ Error at offset {offset}: {e}")
            break
    
    return markets

def filter_markets_by_keywords(markets: List[dict], queries: List[str]) -> List[dict]:
    """
    Filter markets by keyword matching with improved word-level matching.
    Now matches individual words, not just full substrings.
    """
    if not queries:
        return markets
    
    filtered = []
    
    # Break queries into individual words
    query_words = set()
    for query in queries:
        words = query.lower().split()
        query_words.update(words)
    
    log(f"[POLYMARKET] 🔍 Matching words: {sorted(query_words)[:10]}...")
    
    for market in markets:
        question = market.get("question", "").lower()
        question_words = set(question.split())
        
        # Check if any query word appears in question words
        matches = query_words & question_words
        if matches:
            filtered.append(market)
            log(f"[POLYMARKET] ✓ Match: '{market.get('question', '')[:60]}...' (words: {matches})")
    
    return filtered

# ========================================
# STEP 3: SCORE MARKETS WITH LLM
# ========================================

def score_markets_with_llm(markets: List[dict], question_obj: dict) -> Optional[dict]:
    """
    Use LLM to score markets using Tversky similarity.
    
    Returns MK_FINDINGS structure or None if failed.
    """
    if not markets:
        log("[POLYMARKET] ℹ️ No markets to score")
        return create_empty_mk_findings(question_obj)
    
    log(f"[POLYMARKET] 🧮 Scoring {len(markets)} markets with Tversky similarity (α=0.7, β=0.3)...")
    
    if not PROMPT_SCORER:
        log("[POLYMARKET] ❌ Scorer prompt not loaded")
        return None
    
    # Build user message with question + markets
    user_message = "QUESTION_OBJECT:\n" + json.dumps(question_obj, indent=2, ensure_ascii=False)
    user_message += "\n\nMARKETS:\n" + json.dumps(markets, indent=2, ensure_ascii=False)
    
    # Call LLM
    response = call_llm(
        model=SCORER_MODEL,
        system_prompt=PROMPT_SCORER,
        user_payload=user_message,
        temperature=0.2,
        max_tokens=8000,
        timeout=120
    )
    
    if not response or len(response) < 50:
        log("[POLYMARKET] ❌ Empty or invalid response from scorer")
        return None
    
    # Parse JSON response
    try:
        response_clean = response.strip()
        
        # Remove markdown code blocks if present
        if response_clean.startswith("```"):
            lines = response_clean.split("\n")
            response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean
        
        mk_findings = json.loads(response_clean)
        
        # Validate structure
        if not isinstance(mk_findings, dict):
            log("[POLYMARKET] ❌ Response is not a dict")
            return None
        
        if "items" not in mk_findings:
            log("[POLYMARKET] ❌ Missing 'items' in response")
            return None
        
        items = mk_findings["items"]
        num_items = len(items) if isinstance(items, list) else 0
        
        # Log results by category
        log_scored_markets(items)
        
        log(f"[POLYMARKET] ✅ MK_FINDINGS complete with {num_items} markets")
        return mk_findings
    
    except json.JSONDecodeError as e:
        log(f"[POLYMARKET] ❌ JSON parse error: {e}")
        log(f"[POLYMARKET] Response was: {response[:300]}...")
        return None
    except Exception as e:
        log(f"[POLYMARKET] ❌ Error parsing scorer response: {e}")
        return None

def log_scored_markets(items: List[dict]) -> None:
    """Log markets grouped by similarity category"""
    if not items:
        return
    
    exact = [m for m in items if m.get("similarity_S", 0) >= 0.85]
    similar = [m for m in items if 0.70 <= m.get("similarity_S", 0) < 0.85]
    adjacent = [m for m in items if 0.35 <= m.get("similarity_S", 0) < 0.70]
    
    if exact:
        log(f"[POLYMARKET] ✅ EXACT/NEAR-EXACT MARKETS (S ≥ 0.85):")
        for m in exact[:3]:  # Show first 3
            title = m.get("title", "Unknown")[:60]
            sim = m.get("similarity_S", 0)
            bridge = m.get("bridge_note", "")
            log(f"  • \"{title}...\" | S={sim:.2f} | {bridge}")
    
    if similar:
        log(f"[POLYMARKET] ✅ SIMILAR MARKETS (0.70 ≤ S < 0.85):")
        for m in similar[:3]:
            title = m.get("title", "Unknown")[:60]
            sim = m.get("similarity_S", 0)
            bridge = m.get("bridge_note", "")
            log(f"  • \"{title}...\" | S={sim:.2f} | {bridge}")
    
    if adjacent:
        log(f"[POLYMARKET] ✅ ADJACENT MARKETS (0.35 ≤ S < 0.70):")
        for m in adjacent[:3]:
            title = m.get("title", "Unknown")[:60]
            sim = m.get("similarity_S", 0)
            bridge = m.get("bridge_note", "")
            log(f"  • \"{title}...\" | S={sim:.2f} | {bridge}")

def create_empty_mk_findings(question_obj: dict) -> dict:
    """Create empty MK_FINDINGS structure"""
    return {
        "question_type": question_obj.get("question_type", "binary"),
        "locked_mc": False,
        "horizon_utc": question_obj.get("horizon_utc"),
        "options_canonical": question_obj.get("mc_options") or question_obj.get("options"),
        "items": [],
        "prisma": {
            "identified": 0,
            "screened": 0,
            "included": 0,
            "excluded": 0,
            "top_exclusion_reasons": ["No markets found"]
        },
        "search_log": {
            "queries": [],
            "venues": ["Polymarket"]
        }
    }

# ========================================
# MAIN ENTRY POINT
# ================================
