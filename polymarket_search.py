#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polymarket API Search Module
Searches Polymarket for relevant prediction markets and scores them using Tversky similarity.
"""

import os
import json
import requests
import time
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
# STEP 1: GENERATE SEARCH QUERIES
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
    
    # Call LLM
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
    
    # Parse JSON response
    try:
        response_clean = response.strip()
        
        # Remove markdown code blocks
        if "```json" in response_clean:
            start = response_clean.find("```json") + 7
            end = response_clean.find("```", start)
            if end > start:
                response_clean = response_clean[start:end].strip()
        elif response_clean.startswith("```"):
            lines = response_clean.split("\n")
            response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean
        
        # Find JSON object
        if not response_clean.startswith("{"):
            start_idx = response_clean.find("{")
            if start_idx >= 0:
                response_clean = response_clean[start_idx:]
        
        # Find end of JSON
        depth = 0
        for i, char in enumerate(response_clean):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    response_clean = response_clean[:i+1]
                    break
        
        result = json.loads(response_clean)
        
        if not isinstance(result, dict):
            log(f"[POLYMARKET] ❌ Response is not a dict")
            return None
        
        if "tags" not in result or "queries" not in result:
            log(f"[POLYMARKET] ❌ Missing 'tags' or 'queries'")
            return None
        
        tags = result["tags"]
        queries = result["queries"]
        
        # Convert strings to lists if needed
        if isinstance(tags, str):
            tags = [tags]
            result["tags"] = tags
        
        if isinstance(queries, str):
            queries = [q.strip() for q in queries.split(",")]
            result["queries"] = queries
        
        if not isinstance(tags, list) or not isinstance(queries, list):
            log(f"[POLYMARKET] ❌ Invalid types")
            return None
        
        log(f"[POLYMARKET] ✅ Generated {len(queries)} queries: {queries[:3]}...")
        log(f"[POLYMARKET] ✅ Selected tags: {tags}")
        
        return result
    
    except Exception as e:
        log(f"[POLYMARKET] ❌ Error: {e}")
        return None

# ========================================
# STEP 2: SEARCH POLYMARKET API
# ========================================

def search_polymarket_api(tags: List[str], queries: List[str], max_markets: int = 400) -> List[dict]:
    """Search Polymarket API and filter by keywords"""
    log("[POLYMARKET] 🌐 Searching Polymarket API...")
    
    all_markets = []
    seen_ids = set()
    
    tag_id_map = get_tag_ids()
    
    for tag in tags:
        tag_id = tag_id_map.get(tag.lower())
        
        if not tag_id:
            log(f"[POLYMARKET] ⚠️ Unknown tag '{tag}'")
            markets = fetch_markets_untagged(queries, max_markets)
        else:
            log(f"[POLYMARKET] 🔍 Searching tag '{tag}'...")
            markets = fetch_markets_by_tag(tag_id, max_markets)
        
        log(f"[POLYMARKET] 📊 Retrieved {len(markets)} markets")
        
        for market in markets:
            market_id = market.get("condition_id") or market.get("id")
            if market_id and market_id not in seen_ids:
                all_markets.append(market)
                seen_ids.add(market_id)
    
    log(f"[POLYMARKET] 📚 Total markets scanned: {len(all_markets)}")
    
    filtered = filter_markets_by_keywords(all_markets, queries, max_results=25)
    log(f"[POLYMARKET] ✅ Filtered to top {len(filtered)} markets")
    
    return filtered

def get_tag_ids() -> Dict[str, str]:
    """Get Polymarket tag IDs"""
    try:
        url = f"{POLYMARKET_API_BASE}/tags"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return {}
        
        tags_data = response.json()
        tag_map = {}
        
        for tag in tags_data:
            tag_name = tag.get("label", "").lower()
            tag_id = tag.get("id")
            if tag_name and tag_id:
                tag_map[tag_name] = tag_id
        
        log(f"[POLYMARKET] 📋 Loaded {len(tag_map)} tags")
        return tag_map
    
    except Exception as e:
        log(f"[POLYMARKET] ⚠️ Error fetching tags: {e}")
        return {}

def fetch_markets_by_tag(tag_id: str, max_markets: int = 400) -> List[dict]:
    """Fetch markets by tag"""
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
                break
            
            batch = response.json()
            if not batch or len(batch) < 100:
                markets.extend(batch if batch else [])
                break
            
            markets.extend(batch)
        
        except Exception as e:
            log(f"[POLYMARKET] ⚠️ Error at offset {offset}: {e}")
            break
    
    return markets

def fetch_markets_untagged(queries: List[str], max_markets: int = 400) -> List[dict]:
    """Fetch markets without tag"""
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
                break
            
            batch = response.json()
            if not batch or len(batch) < 100:
                markets.extend(batch if batch else [])
                break
            
            markets.extend(batch)
        
        except Exception as e:
            log(f"[POLYMARKET] ⚠️ Error: {e}")
            break
    
    return markets

def filter_markets_by_keywords(markets: List[dict], queries: List[str], max_results: int = 25) -> List[dict]:
    """Filter and rank markets by keyword relevance"""
    if not queries:
        return markets[:max_results]
    
    query_words = set()
    for query in queries:
        query_words.update(query.lower().split())
    
    log(f"[POLYMARKET] 🔍 Scanning {len(markets)} markets...")
    
    scored_markets = []
    
    for market in markets:
        question = market.get("question", "").lower()
        question_words = set(question.split())
        matches = query_words & question_words
        
        if matches:
            score = len(matches)
            scored_markets.append((score, market, matches))
    
    scored_markets.sort(key=lambda x: x[0], reverse=True)
    
    log(f"[POLYMARKET] 📊 Found {len(scored_markets)} matches (keeping top {max_results})")
    
    for i, (score, market, matches) in enumerate(scored_markets[:max_results], 1):
        q = market.get('question', '')[:60]
        log(f"[POLYMARKET] #{i} (score={score}): '{q}...'")
    
    return [market for score, market, matches in scored_markets[:max_results]]

# ========================================
# STEP 3: SCORE MARKETS WITH LLM
# ========================================

def score_markets_with_llm(markets: List[dict], question_obj: dict) -> Optional[dict]:
    """Score markets using Tversky similarity"""
    if not markets:
        log("[POLYMARKET] ℹ️ No markets to score")
        return create_empty_mk_findings(question_obj)
    
    log(f"[POLYMARKET] 🧮 Scoring {len(markets)} markets...")
    
    if not PROMPT_SCORER:
        log("[POLYMARKET] ❌ Scorer prompt not loaded")
        return None
    
    user_message = "QUESTION_OBJECT:\n" + json.dumps(question_obj, indent=2, ensure_ascii=False)
    user_message += "\n\nMARKETS:\n" + json.dumps(markets, indent=2, ensure_ascii=False)
    
    response = call_llm(
        model=SCORER_MODEL,
        system_prompt=PROMPT_SCORER,
        user_payload=user_message,
        temperature=0.2,
        max_tokens=16000,
        timeout=180
    )
    
    if not response or len(response) < 50:
        log("[POLYMARKET] ❌ Empty response")
        return None
    
    # Check for rate limit
    if "rate" in response.lower() or "429" in response:
        log("[POLYMARKET] ⚠️ Rate limit detected - waiting 30s...")
        time.sleep(30)
        return None
    
    try:
        response_clean = response.strip()
        
        # Remove markdown
        if "```json" in response_clean:
            start = response_clean.find("```json") + 7
            end = response_clean.find("```", start)
            if end > start:
                response_clean = response_clean[start:end].strip()
        elif response_clean.startswith("```"):
            lines = response_clean.split("\n")
            response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean
        
        # Find JSON
        if not response_clean.startswith("{"):
            start_idx = response_clean.find("{")
            if start_idx >= 0:
                response_clean = response_clean[start_idx:]
        
        # Find end of JSON
        depth = 0
        for i, char in enumerate(response_clean):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    response_clean = response_clean[:i+1]
                    break
        
        mk_findings = json.loads(response_clean)
        
        if not isinstance(mk_findings, dict) or "items" not in mk_findings:
            log("[POLYMARKET] ❌ Invalid structure")
            return None
        
        items = mk_findings["items"]
        log_scored_markets(items)
        
        log(f"[POLYMARKET] ✅ MK_FINDINGS complete with {len(items)} markets")
        return mk_findings
    
    except json.JSONDecodeError as e:
        log(f"[POLYMARKET] ❌ JSON error: {e}")
        log(f"[POLYMARKET] Response: {response[:500]}...")
        return None
    except Exception as e:
        log(f"[POLYMARKET] ❌ Error: {e}")
        return None

def log_scored_markets(items: List[dict]) -> None:
    """Log markets by similarity category"""
    if not items:
        return
    
    exact = [m for m in items if m.get("similarity_S", 0) >= 0.85]
    similar = [m for m in items if 0.70 <= m.get("similarity_S", 0) < 0.85]
    adjacent = [m for m in items if 0.35 <= m.get("similarity_S", 0) < 0.70]
    
    if exact:
        log(f"[POLYMARKET] ✅ EXACT (S ≥ 0.85):")
        for m in exact[:3]:
            title = m.get("title", "")[:60]
            sim = m.get("similarity_S", 0)
            log(f"  • \"{title}...\" | S={sim:.2f}")
    
    if similar:
        log(f"[POLYMARKET] ✅ SIMILAR (0.70-0.85):")
        for m in similar[:3]:
            title = m.get("title", "")[:60]
            sim = m.get("similarity_S", 0)
            log(f"  • \"{title}...\" | S={sim:.2f}")
    
    if adjacent:
        log(f"[POLYMARKET] ✅ ADJACENT (0.35-0.70):")
        for m in adjacent[:3]:
            title = m.get("title", "")[:60]
            sim = m.get("similarity_S", 0)
            log(f"  • \"{title}...\" | S={sim:.2f}")

def create_empty_mk_findings(question_obj: dict) -> dict:
    """Create empty MK_FINDINGS"""
    return {
        "question_type": question_obj.get("question_type", "binary"),
        "locked_mc": False,
        "horizon_utc": question_obj.get("horizon_utc"),
        "options_canonical": question_obj.get("mc_options") or question_obj.get("options"),
        "items": [],
        "prisma": {
            "identified": 0,
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
# ========================================

def get_polymarket_findings(question_obj: dict, max_retries: int = 2) -> Optional[dict]:
    """
    Main entry point: Get MK_FINDINGS from Polymarket API.
    
    Workflow:
    1. Generate search queries (LLM)
    2. Search Polymarket API (Python)
    3. Filter to top 25 (Python)
    4. Score with Tversky (LLM)
    """
    log("[POLYMARKET] 🚀 Starting Polymarket API search...")
    
    # Step 1: Generate queries
    query_result = None
    for attempt in range(1, max_retries + 1):
        query_result = generate_search_queries(question_obj)
        if query_result:
            break
        if attempt < max_retries:
            log(f"[POLYMARKET] 🔄 Retry {attempt}/{max_retries}...")
    
    if not query_result:
        log("[POLYMARKET] ❌ Failed to generate queries")
        return create_empty_mk_findings(question_obj)
    
    tags = query_result.get("tags", [])
    queries = query_result.get("queries", [])
    
    if not tags:
        tags = ["politics"]
    
    if not queries:
        log("[POLYMARKET] ❌ No queries")
        return create_empty_mk_findings(question_obj)
    
    # Step 2: Search API
    try:
        markets = search_polymarket_api(tags, queries, max_markets=400)
    except Exception as e:
        log(f"[POLYMARKET] ❌ API error: {e}")
        return create_empty_mk_findings(question_obj)
    
    if not markets:
        log("[POLYMARKET] ℹ️ No markets found")
        empty = create_empty_mk_findings(question_obj)
        empty["search_log"]["queries"] = queries
        return empty
    
    # Step 3: Score markets
    mk_findings = None
    for attempt in range(1, max_retries + 1):
        mk_findings = score_markets_with_llm(markets, question_obj)
        if mk_findings:
            break
        if attempt < max_retries:
            log(f"[POLYMARKET] 🔄 Retry {attempt}/{max_retries}...")
            time.sleep(5)  # Brief delay between retries
    
    if not mk_findings:
        log("[POLYMARKET] ❌ Failed to score markets")
        return create_empty_mk_findings(question_obj)
    
    return mk_findings

# ========================================
# END OF POLYMARKET_SEARCH MODULE
# ========================================
