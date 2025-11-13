import asyncio
from typing import List, Dict
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FastContentExtractor import FastContentExtractor
from prompts import INITIAL_SEARCH_PROMPT, CONTINUATION_SEARCH_PROMPT
import dateparser
from dotenv import load_dotenv
import json
import os
from aiohttp import ClientSession, ClientTimeout
from asknews_sdk import AskNewsSDK
from prompts import context
from dotenv import load_dotenv
import aiohttp
import re
import random
import time
from openai import OpenAI   
import traceback
from datetime import datetime, timezone
load_dotenv()

# ========================================
# HELPER FUNCTIONS (must be defined first!)
# ========================================

def write(x):
    print(x)

def parse_date(date_str: str) -> str:
    """Parse date string and return formatted date."""
    parsed_date = dateparser.parse(date_str, settings={'STRICT_PARSING': False})
    if parsed_date:
        return parsed_date.strftime("%b %d, %Y")
    return "Unknown"

def validate_time(before_date_str, source_date_str):
    """
    Validate that source date is before the cutoff date.
    Handles timezone-aware and timezone-naive datetimes.
    """
    if source_date_str == "Unknown":
        return False
    
    try:
        before_date = dateparser.parse(before_date_str)
        source_date = dateparser.parse(source_date_str)
        
        if before_date is None or source_date is None:
            return False
        
        # Make both timezone-aware if either is timezone-aware
        if before_date.tzinfo is not None and source_date.tzinfo is None:
            source_date = source_date.replace(tzinfo=timezone.utc)
        elif before_date.tzinfo is None and source_date.tzinfo is not None:
            before_date = before_date.replace(tzinfo=timezone.utc)
        
        return source_date <= before_date
    except Exception as e:
        write(f"[validate_time] Error comparing dates: {e}")
        return False

# ========================================
# API KEYS AND CONFIGURATION
# ========================================

SERPER_KEY = os.getenv("SERPER_KEY") or os.getenv("SERPER_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Use OpenRouter for all LLM calls
if OPENROUTER_API_KEY:
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    write(f"[INIT] ✅ Using OpenRouter for LLM calls")
elif OPENAI_API_KEY:
    # Fallback to direct OpenAI if OpenRouter not available
    client = OpenAI(api_key=OPENAI_API_KEY)
    write(f"[INIT] ✅ Using OpenAI directly")
else:
    client = None
    write(f"[INIT] ⚠️ No OpenAI/OpenRouter API key found")

# Check which APIs are available
HAS_SERPER = bool(SERPER_KEY)
HAS_ASKNEWS = bool(ASKNEWS_CLIENT_ID and ASKNEWS_SECRET)
HAS_NEWSAPI = bool(NEWSAPI_KEY)
HAS_PERPLEXITY = bool(PERPLEXITY_API_KEY)

# Log API availability
write(f"[INIT] API Availability:")
write(f"  Serper (Google): {'✅' if HAS_SERPER else '❌'}")
write(f"  AskNews: {'⚠️ → Will try NewsAPI' if not HAS_ASKNEWS else '✅'}")
write(f"  NewsAPI: {'✅' if HAS_NEWSAPI else '❌'}")
write(f"  Perplexity: {'✅' if HAS_PERPLEXITY else '⚠️ → Will use agentic search fallback'}")

# ========================================
# JINA READER FAILSAFE (NEW - PURELY ADDITIVE)
# ========================================

async def fetch_with_jina(url: str) -> dict:
    """
    FAILSAFE ONLY: Fetch content using Jina Reader API when FastContentExtractor fails.
    Jina bypasses bot protection and returns clean markdown content.
    
    This is a FREE service with no API key required.
    Only called when FastContentExtractor returns 401/empty content.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        dict with 'content', 'title', and 'success' keys
    """
    jina_url = f"https://r.jina.ai/{url}"
    
    try:
        write(f"[JINA FAILSAFE] Attempting to fetch: {url}")
        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ForecastBot/1.0)'
            }
            async with session.get(jina_url, timeout=timeout, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    # Jina returns markdown, clean it up a bit
                    content = content.strip()
                    if len(content) > 50:  # Basic validation
                        write(f"[JINA FAILSAFE] ✅ Success: {len(content)} chars from {url}")
                        return {
                            'content': content,
                            'title': 'Extracted via Jina',
                            'success': True
                        }
                    else:
                        write(f"[JINA FAILSAFE] ⚠️ Content too short ({len(content)} chars) for {url}")
                        return {'content': '', 'success': False}
                else:
                    write(f"[JINA FAILSAFE] ⚠️ Status {response.status} for {url}")
                    return {'content': '', 'success': False}
    except asyncio.TimeoutError:
        write(f"[JINA FAILSAFE] ⏱️ Timeout for {url}")
        return {'content': '', 'success': False}
    except Exception as e:
        write(f"[JINA FAILSAFE] ❌ Error for {url}: {str(e)}")
        return {'content': '', 'success': False}

# ========================================
# ARTICLE SUMMARIZATION
# ========================================

async def summarize_article(article: str, question_details: dict) -> str:
    assistant_prompt = """
You are an assistant to a superforecaster and your task involves high-quality information retrieval to help the forecaster make the most informed forecasts. Forecasting involves parsing through an immense trove of internet articles and web content. To make this easier for the forecaster, you read entire articles and extract the key pieces of the articles relevant to the question. The key pieces generally include:

1. Facts, statistics and other objective measurements described in the article
2. Opinions from reliable and named sources (e.g. if the article writes 'according to a 2023 poll by Gallup' or 'The 2025 presidential approval rating poll by Reuters' etc.)
3. Potentially useful opinions from less reliable/not-named sources (you explicitly document the less reliable origins of these opinions though)

Today, you're focusing on the question:

{title}

Resolution criteria:
{resolution_criteria}

Fine print:
{fine_print}

Background information:
{background}

Article to summarize:
{article}

Note: If the web content extraction is incomplete or you believe the quality of the extracted content isn't the best, feel free to add a disclaimer before your summary.

Please summarize only the article given, not injecting your own knowledge or providing a forecast. Aim to achieve a balance between a superficial summary and an overly verbose account.
"""
    
    prompt = assistant_prompt.format(
        title=question_details["title"],
        resolution_criteria=question_details["resolution_criteria"],
        fine_print=question_details["fine_print"],
        background=question_details["description"],
        article=article
    )
    return await call_gpt(prompt)

# ========================================
# NEWSAPI (fallback for AskNews)
# ========================================

async def call_newsapi(question: str) -> str:
    """
    Use NewsAPI as fallback when AskNews unavailable.
    Multi-tier fallback: AskNews → NewsAPI
    """
    if not HAS_NEWSAPI:
        write(f"[call_newsapi] ⚠️ NewsAPI not configured")
        return f"<Asknews_articles>\nQuery: {question}\nNo news APIs available.\n</Asknews_articles>\n"
    
    try:
        write(f"[call_newsapi] Using NewsAPI for query: {question}")
        
        # Simplify query for NewsAPI - extract keywords only
        keywords = ' '.join([word for word in question.split() if len(word) > 4])[:100]
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": keywords,
            "apiKey": NEWSAPI_KEY,
            "pageSize": 10,
            "sortBy": "relevancy",
            "language": "en"
        }
        
        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.get(url, params=params, timeout=timeout) as response:
                if response.status == 429:
                    write(f"[call_newsapi] ⚠️ Rate limit hit")
                    return f"<Asknews_articles>\nQuery: {question}\nNewsAPI rate limit reached.\n</Asknews_articles>\n"
                
                if response.status != 200:
                    error_text = await response.text()
                    write(f"[call_newsapi] ❌ Error {response.status}: {error_text}")
                    return f"<Asknews_articles>\nQuery: {question}\nNewsAPI error.\n</Asknews_articles>\n"
                
                data = await response.json()
                
                if data.get("status") != "ok":
                    write(f"[call_newsapi] ❌ API returned status: {data.get('status')}")
                    return f"<Asknews_articles>\nQuery: {question}\nNewsAPI unavailable.\n</Asknews_articles>\n"
                
                articles = data.get("articles", [])
                
                if not articles:
                    write(f"[call_newsapi] No articles found")
                    return f"<Asknews_articles>\nQuery: {question}\nNo articles found.\n</Asknews_articles>\n"
                
                write(f"[call_newsapi] ✅ Found {len(articles)} articles")
                
                formatted_articles = "Here are the relevant news articles:\n\n"
                
                for article in articles[:8]:
                    title = article.get("title", "No title")
                    description = article.get("description", "No description available")
                    published_at = article.get("publishedAt", "Unknown date")
                    source_name = article.get("source", {}).get("name", "Unknown source")
                    url_link = article.get("url", "")
                    
                    # Parse and format date
                    try:
                        parsed_date = dateparser.parse(published_at)
                        if parsed_date:
                            formatted_date = parsed_date.strftime("%B %d, %Y %I:%M %p")
                        else:
                            formatted_date = published_at
                    except:
                        formatted_date = published_at
                    
                    formatted_articles += f"**{title}**\n"
                    formatted_articles += f"{description}\n"
                    formatted_articles += f"Publish date: {formatted_date}\n"
                    formatted_articles += f"Source: [{source_name}]({url_link})\n\n"
                
                return formatted_articles
                
    except asyncio.TimeoutError:
        write(f"[call_newsapi] ⚠️ Timeout")
        return f"<Asknews_articles>\nQuery: {question}\nNewsAPI timeout.\n</Asknews_articles>\n"
    except Exception as e:
        write(f"[call_newsapi] ❌ Error: {str(e)}")
        return f"<Asknews_articles>\nQuery: {question}\nNewsAPI error: {str(e)}\n</Asknews_articles>\n"

# ========================================
# ASKNEWS API (with NewsAPI fallback)
# ========================================

async def call_asknews(question: str) -> str:
    """
    Use AskNews API if available, otherwise fall back to NewsAPI.
    Multi-tier fallback: AskNews → NewsAPI
    """
    if not HAS_ASKNEWS:
        write(f"[call_asknews] ⚠️ AskNews API not configured, trying NewsAPI")
        return await call_newsapi(question)
    
    try:
        write(f"[call_asknews] Using AskNews API for query: {question}")
        ask = AskNewsSDK(
            client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
        )

        async with aiohttp.ClientSession() as session:
            # Create tasks for both API calls
            hot_task = asyncio.create_task(asyncio.to_thread(ask.news.search_news,
                query=question,
                n_articles=8,
                return_type="both",
                strategy="latest news"
            ))
            historical_task = asyncio.create_task(asyncio.to_thread(ask.news.search_news,
                query=question,
                n_articles=8,
                return_type="both",
                strategy="news knowledge"
            ))

            # Wait for both tasks to complete
            hot_response, historical_response = await asyncio.gather(hot_task, historical_task)

        hot_articles = hot_response.as_dicts
        historical_articles = historical_response.as_dicts
        formatted_articles = "Here are the relevant news articles:\n\n"

        if hot_articles:
            hot_articles = [article.__dict__ for article in hot_articles]
            hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

            for article in hot_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

        if historical_articles:
            historical_articles = [article.__dict__ for article in historical_articles]
            historical_articles = sorted(
                historical_articles, key=lambda x: x["pub_date"], reverse=True
            )

            for article in historical_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

        if not hot_articles and not historical_articles:
            formatted_articles += "No articles were found.\n\n"
            return formatted_articles

        write(f"[call_asknews] ✅ Successfully retrieved articles")
        return formatted_articles
        
    except Exception as e:
        write(f"[call_asknews] ❌ AskNews API error: {str(e)}, trying NewsAPI")
        return await call_newsapi(question)

# ========================================
# AGENTIC SEARCH (for deep research)
# ========================================

async def agentic_search(query: str) -> str:
    """
    Performs agentic search using o3 to iteratively research and analyze a query.
    Used as fallback when Perplexity API is unavailable.
    """
    write(f"[agentic_search] Starting research for query: {query}")
    
    if not client:
        write(f"[agentic_search] ⚠️ No LLM client available")
        return f"<Agent_report>\nQuery: {query}\nNo LLM client configured.\n</Agent_report>\n"
    
    max_steps = 2  # Reduced for efficiency (was 7 in original)
    current_analysis = ""
    all_search_queries = []
    
    # Cost tracking variables
    total_input_tokens = 0
    total_output_tokens = 0
    
    def estimate_tokens(text: str) -> int:
        """Estimate token count using ~4 characters per token rule"""
        return max(1, len(text) // 4)
    
    def calculate_cost(input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        input_cost = (input_tokens / 1_000_000) * 1.100
        output_cost = (output_tokens / 1_000_000) * 4.400
        return input_cost + output_cost
    
    for step in range(max_steps):
        try:
            # Prepare the prompt
            if step == 0:
                prompt = INITIAL_SEARCH_PROMPT.format(query=query)
            else:
                if current_analysis:
                    previous_section = f"Your previous analysis:\n{current_analysis}\n\nPrevious search queries used: {', '.join(all_search_queries)}\n"
                else:
                    previous_section = f"Previous search queries used: {', '.join(all_search_queries)}\n"
                
                prompt = CONTINUATION_SEARCH_PROMPT.format(
                    query=query,
                    previous_section=previous_section,
                    search_results=search_results
                )
            
            prompt_tokens = estimate_tokens(prompt)
            total_input_tokens += prompt_tokens
            
            write(f"[agentic_search] Step {step + 1}: Calling o3")
            response = await call_gpt(prompt, step)
            
            response_tokens = estimate_tokens(response)
            total_output_tokens += response_tokens
            
            # Parse the response
            analysis_match = re.search(r'Analysis:\s*(.*?)(?=Search queries:|$)', response, re.DOTALL)
            if not analysis_match:
                write(f"[agentic_search] Error: Could not parse analysis from response")
                return f"<Agent_report>\nQuery: {query}\nError parsing analysis.\n</Agent_report>\n"
            
            if step > 0:
                current_analysis = analysis_match.group(1).strip()
                write(f"[agentic_search] Step {step + 1}: Analysis updated ({len(current_analysis)} chars)")
            else:
                write(f"[agentic_search] Step 1: Initial query understanding complete")
            
            # Check for search queries
            search_queries_match = re.search(r'Search queries:\s*(.*)', response, re.DOTALL)
            
            if step == 0 and not search_queries_match:
                write(f"[agentic_search] Error: No search queries in initial response")
                return f"<Agent_report>\nQuery: {query}\nError: No search queries generated.\n</Agent_report>\n"
            
            if not search_queries_match or step == max_steps - 1:
                if step > 0:
                    write(f"[agentic_search] Research complete at step {step + 1}")
                    break
            
            # Extract search queries with sources
            queries_text = search_queries_match.group(1).strip()
            search_queries_with_source = re.findall(r'\d+\.\s*([^(]+?)\s*\((Google|Google News)\)', queries_text)
            
            if not search_queries_with_source:
                if step == 0:
                    write(f"[agentic_search] Error: No valid search queries in initial response")
                    return f"<Agent_report>\nQuery: {query}\nError: Failed to parse search queries.\n</Agent_report>\n"
                else:
                    write(f"[agentic_search] No new search queries, completing research")
                    break
            
            search_queries_with_source = [(q.strip(), source) for q, source in search_queries_with_source[:5]]
            
            write(f"[agentic_search] Step {step + 1}: Found {len(search_queries_with_source)} search queries")
            all_search_queries.extend([q for q, _ in search_queries_with_source])
            
            # Execute searches in parallel
            search_tasks = []
            for sq, source in search_queries_with_source:
                write(f"[agentic_search] Searching: {sq} (Source: {source})")
                search_tasks.append(
                    google_search_agentic(
                        sq,
                        is_news=(source == "Google News")
                    )
                )
            
            search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            search_results = ""
            for (sq, source), result in zip(search_queries_with_source, search_results_list):
                if isinstance(result, Exception):
                    search_results += f"\nSearch query: {sq} (Source: {source})\nError: {str(result)}\n"
                else:
                    search_results += f"\nSearch query: {sq} (Source: {source})\n{result}\n"
            
            write(f"[agentic_search] Step {step + 1}: Search complete, {len(search_results)} chars of results")
            
        except Exception as e:
            write(f"[agentic_search] Error at step {step + 1}: {str(e)}")
            if current_analysis:
                break
            else:
                return f"<Agent_report>\nQuery: {query}\nError: {str(e)}\n</Agent_report>\n"
    
    steps_used = step + 1
    total_cost = calculate_cost(total_input_tokens, total_output_tokens)
    
    print(f"\n🔍 Agentic Search Summary:")
    print(f"   Steps used: {steps_used}")
    print(f"   Total tokens: {total_input_tokens + total_output_tokens:,} ({total_input_tokens:,} input + {total_output_tokens:,} output)")
    print(f"   Estimated cost: ${total_cost:.4f}")
    
    if not current_analysis:
        return f"<Agent_report>\nQuery: {query}\nError: No analysis generated.\n</Agent_report>\n"
    
    return current_analysis

# ========================================
# PERPLEXITY API (with agentic search fallback)
# ========================================

async def call_perplexity(prompt: str) -> str:
    """
    Call Perplexity API if available, otherwise fall back to agentic search.
    Multi-tier fallback: Perplexity → agentic_search (o3 + Google)
    """
    if not HAS_PERPLEXITY:
        write(f"[call_perplexity] ⚠️ Perplexity API not configured, using agentic search fallback")
        return await agentic_search(prompt)
    
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar-deep-research",
        "messages": [
            {
                "role": "system",
                "content": "Be thorough and detailed. Be objective in your analysis, proving documented facts only. Cite all sources with names and dates."
            },
            {
                "role": "user",
                "content": prompt + " Cite all sources with names and dates, compiling a list of sources at the end. Be objective in your analysis, providing documented facts only."
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }

    max_retries = 3
    backoff_base = 3

    for attempt in range(1, max_retries + 1):
        try:
            write(f"[Perplexity API] Attempt {attempt} for query: {prompt[:50]}...")
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=800)
                async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                        write(f"[Perplexity API] ✅ Success on attempt {attempt}")
                        return content.strip()
                    elif response.status == 401:
                        write(f"[Perplexity API] ❌ 401 Unauthorized - API key invalid/expired, using agentic search fallback")
                        return await agentic_search(prompt)
                    else:
                        response_text = await response.text()
                        write(f"[Perplexity API] ❌ Error: HTTP {response.status}: {response_text[:200]}...")
                        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            write(f"[Perplexity API] ⚠️ Attempt {attempt} failed: {e}")
        
        if attempt < max_retries:
            wait_time = backoff_base * attempt
            write(f"[Perplexity API] 🔁 Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
        else:
            write(f"[Perplexity API] ❌ Max retries reached, using agentic search fallback")
            return await agentic_search(prompt)

    return await agentic_search(prompt)

# ========================================
# GOOGLE SEARCH (core function)
# ========================================

async def google_search(query, is_news=False, date_before=None):
    """
    Google search via Serper API.
    Returns list of URLs or empty list if unavailable.
    """
    if not HAS_SERPER:
        write(f"[google_search] ⚠️ Serper API not configured")
        return []
    
    original_query = query
    query = query.replace('"', '').replace("'", '').strip()
    write(f"[google_search] Cleaned query: '{query}' (original: '{original_query}') | is_news={is_news}, date_before={date_before}")
    
    search_type = "news" if is_news else "search"
    url = f"https://google.serper.dev/{search_type}"
    headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "q": query,
        "num": 20
    })
    timeout = ClientTimeout(total=70)

    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, data=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('news' if is_news else 'organic', [])
                    write(f"[google_search] Found {len(items)} raw results")

                    filtered_items = []
                    for item in items:
                        item_url = item.get('link')
                        item_date_str = item.get('date', '')
                        item_date = parse_date(item_date_str)
                        if date_before:
                            if item_date != "Unknown" and validate_time(date_before, item_date):
                                write(f"[google_search] ✅ Keeping: {item_url} (date: {item_date})")
                                filtered_items.append(item)
                            else:
                                write(f"[google_search] ❌ Dropped by date: {item_url} (date: {item_date})")
                        else:
                            write(f"[google_search] ✅ Keeping: {item_url}")
                            filtered_items.append(item)

                        if len(filtered_items) >= 12:
                            break
                    
                    urls = [item['link'] for item in filtered_items]
                    write(f"[google_search] Returning {len(urls)} URLs")
                    return urls
                else:
                    response_text = await response.text()
                    write(f"[google_search] ❌ Error in Serper API: Status {response.status}")
                    write(f"[google_search] Response: {response_text[:500]}...")
                    response.raise_for_status()
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        write(f"[google_search] ❌ {error_type}: {error_msg}")
        
        # Log full traceback for debugging
        import traceback
        tb = traceback.format_exc()
        write(f"[google_search] Traceback:\n{tb}")
        
        return []

# ========================================
# GPT CALL (using o3 via OpenRouter)
# ========================================

async def call_gpt(prompt, step=1):
    """Call o3 via OpenRouter for research tasks"""
    if not client:
        write(f"[call_gpt] ❌ No LLM client available")
        return "Error: No LLM client configured"
    
    try:
        # Use o3 (full, not mini) as per original
        response = client.responses.create(
            model="openai/o3-mini",  # ← CHANGE THIS back to model="openai/o3" when ready!
            input=prompt
        )
        return response.output_text
    except Exception as e:
        write(f"[call_gpt] Error: {str(e)}")
        return f"Error calling LLM API: {str(e)}"

# ========================================
# GOOGLE SEARCH AND SCRAPE (WITH JINA FAILSAFE)
# ========================================

async def google_search_and_scrape(query, is_news, question_details, date_before=None):
    """
    Performs Google search and scrapes/summarizes articles.
    NOW WITH JINA FAILSAFE: If FastContentExtractor fails (401s), tries Jina Reader.
    """
    write(f"[google_search_and_scrape] Called with query='{query}', is_news={is_news}, date_before={date_before}")
    try:
        urls = await google_search(query, is_news, date_before)

        if not urls:
            write(f"[google_search_and_scrape] ❌ No URLs returned for query: '{query}'")
            return f"<Summary query=\"{query}\">No URLs returned from Google.</Summary>\n"

        # STEP 1: Try FastContentExtractor first (original behavior)
        async with FastContentExtractor() as extractor:
            write(f"[google_search_and_scrape] 🔍 Starting content extraction for {len(urls)} URLs")
            results = await extractor.extract_content(urls)
            write(f"[google_search_and_scrape] ✅ Finished content extraction")

        # STEP 2: NEW FAILSAFE - Check if FastContentExtractor failed, use Jina as fallback
        failed_urls = []
        for url in urls:
            if url not in results or not results[url].get('content', '').strip():
                failed_urls.append(url)
        
        if failed_urls:
            write(f"[google_search_and_scrape] 🔄 FastContentExtractor failed for {len(failed_urls)} URLs, trying Jina failsafe")
            # Limit to 3 to avoid overwhelming free Jina API
            jina_tasks = [fetch_with_jina(url) for url in failed_urls[:3]]
            jina_results = await asyncio.gather(*jina_tasks)
            
            for url, jina_result in zip(failed_urls[:3], jina_results):
                if jina_result.get('success'):
                    results[url] = jina_result
                    write(f"[google_search_and_scrape] ✅ Jina failsafe succeeded for {url}")
                else:
                    write(f"[google_search_and_scrape] ⚠️ Jina failsafe also failed for {url}")

        # STEP 3: Continue with original summarization logic
        summarize_tasks = []
        no_results = 3
        valid_urls = []
        for url, data in results.items():
            if len(summarize_tasks) >= no_results:
                break  
            content = (data.get('content') or '').strip()
            if len(content.split()) < 100:
                write(f"[google_search_and_scrape] ⚠️ Skipping low-content article: {url}")
                continue
            if content:
                truncated = content[:8000]
                write(f"[google_search_and_scrape] ✂️ Truncated content for summarization: {len(truncated)} chars from {url}")
                summarize_tasks.append(
                    asyncio.create_task(summarize_article(truncated, question_details))
                )
                valid_urls.append(url)
            else:
                write(f"[google_search_and_scrape] ⚠️ No content for {url}, skipping summarization.")

        if not summarize_tasks:
            write("[google_search_and_scrape] ⚠️ Warning: No content to summarize (all extraction methods failed)")
            return f"<Summary query=\"{query}\">No usable content extracted from any URL.</Summary>\n"

        summaries = await asyncio.gather(*summarize_tasks, return_exceptions=True)

        output = ""
        for url, summary in zip(valid_urls, summaries):
            if isinstance(summary, Exception):
                write(f"[google_search_and_scrape] ❌ Error summarizing {url}: {summary}")
                output += f"\n<Summary source=\"{url}\">\nError summarizing content: {str(summary)}\n</Summary>\n"
            else:
                output += f"\n<Summary source=\"{url}\">\n{summary}\n</Summary>\n"

        return output
    except Exception as e:
        write(f"[google_search_and_scrape] Error: {str(e)}")
        traceback_str = traceback.format_exc()
        write(f"Traceback: {traceback_str}")
        return f"<Summary query=\"{query}\">Error during search and scrape: {str(e)}</Summary>\n"

# ========================================
# GOOGLE SEARCH AGENTIC (WITH JINA FAILSAFE)
# ========================================

async def google_search_agentic(query, is_news=False):
    """
    Performs Google search and returns raw article content without summarization.
    Used for agentic search where the agent will analyze the raw content.
    NOW WITH JINA FAILSAFE: If FastContentExtractor fails (401s), tries Jina Reader.
    """
    write(f"[google_search_agentic] Called with query='{query}', is_news={is_news}")
    try:
        urls = await google_search(query, is_news)

        if not urls:
            write(f"[google_search_agentic] ❌ No URLs returned for query: '{query}'")
            return f"<RawContent query=\"{query}\">No URLs returned from Google.</RawContent>\n"

        # STEP 1: Try FastContentExtractor first (original behavior)
        async with FastContentExtractor() as extractor:
            write(f"[google_search_agentic] 🔍 Starting content extraction for {len(urls)} URLs")
            results = await extractor.extract_content(urls)
            write(f"[google_search_agentic] ✅ Finished content extraction")

        # STEP 2: NEW FAILSAFE - Check if FastContentExtractor failed, use Jina as fallback
        failed_urls = []
        for url in urls:
            if url not in results or not results[url].get('content', '').strip():
                failed_urls.append(url)
        
        if failed_urls:
            write(f"[google_search_agentic] 🔄 FastContentExtractor failed for {len(failed_urls)} URLs, trying Jina failsafe")
            # Limit to 3 to avoid overwhelming free Jina API
            jina_tasks = [fetch_with_jina(url) for url in failed_urls[:3]]
            jina_results = await asyncio.gather(*jina_tasks)
            
            for url, jina_result in zip(failed_urls[:3], jina_results):
                if jina_result.get('success'):
                    results[url] = jina_result
                    write(f"[google_search_agentic] ✅ Jina failsafe succeeded for {url}")
                else:
                    write(f"[google_search_agentic] ⚠️ Jina failsafe also failed for {url}")

        # STEP 3: Continue with original raw content logic
        output = ""
        no_results = 3
        results_count = 0
        
        for url, data in results.items():
            if results_count >= no_results:
                break
                
            content = (data.get('content') or '').strip()
            if len(content.split()) < 100:
                write(f"[google_search_agentic] ⚠️ Skipping low-content article: {url}")
                continue
                
            if content:
                truncated = content[:8000]
                write(f"[google_search_agentic] ✂️ Including content: {len(truncated)} chars from {url}")
                output += f"\n<RawContent source=\"{url}\">\n{truncated}\n</RawContent>\n"
                results_count += 1
            else:
                write(f"[google_search_agentic] ⚠️ No content for {url}, skipping.")

        if not output:
            write("[google_search_agentic] ⚠️ Warning: No usable content found (all extraction methods failed)")
            return f"<RawContent query=\"{query}\">No usable content extracted from any URL.</RawContent>\n"

        return output
        
    except Exception as e:
        write(f"[google_search_agentic] Error: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        write(f"Traceback: {traceback_str}")
        return f"<RawContent query=\"{query}\">Error during search: {str(e)}</RawContent>\n"

# ========================================
# PROCESS SEARCH QUERIES (FULL MODE)
# ========================================

async def process_search_queries(response: str, forecaster_id: str, question_details: dict):
    """
    Parses out search queries from the forecaster's response, executes them
    (AskNews, Agent/Perplexity or Google/Google News), and returns formatted summaries.
    ALL functionality is preserved with fallbacks.
    """
    try:
        # 1) Extract the "Search queries:" block
        search_queries_block = re.search(r'(?:Search queries:)(.*)', response, re.DOTALL | re.IGNORECASE)
        if not search_queries_block:
            write(f"Forecaster {forecaster_id}: No search queries block found")
            return ""

        queries_text = search_queries_block.group(1).strip()

        # 2) Try to find queries of the form: 1. "text" (Source)
        search_queries = re.findall(
            r'(?:\d+\.\s*)?(["\']?(.*?)["\']?)\s*\((Google|Google News|Assistant|Agent|Perplexity)\)',
            queries_text
        )
        # 3) Fallback to unquoted queries if none found
        if not search_queries:
            search_queries = re.findall(
                r'(?:\d+\.\s*)?([^(\n]+)\s*\((Google|Google News|Assistant|Agent|Perplexity)\)',
                queries_text
            )

        if not search_queries:
            write(f"Forecaster {forecaster_id}: No valid search queries found:\n{queries_text}")
            return ""

        write(f"Forecaster {forecaster_id}: Processing {len(search_queries)} search queries")

        # 4) Kick off one asyncio task per query
        tasks = []
        query_sources = []
        
        for match in search_queries:
            if len(match) == 3:
                _, raw_query, source = match
            else:
                raw_query, source = match

            query = raw_query.strip().strip('"').strip("'")
            if not query:
                continue

            write(f"Forecaster {forecaster_id}: Query='{query}' Source={source}")
            query_sources.append((query, source))

            if source in ("Google", "Google News"):
                tasks.append(
                    google_search_and_scrape(
                        query,
                        is_news=(source == "Google News"),
                        question_details=question_details,
                        date_before=question_details.get("resolution_date")
                    )
                )
            elif source == "Assistant":
                tasks.append(call_asknews(query))
            elif source == "Perplexity":
                tasks.append(call_perplexity(query))
            elif source == "Agent":
                tasks.append(agentic_search(query))

        if not tasks:
            write(f"Forecaster {forecaster_id}: No tasks generated")
            return ""

        # 5) Await all tasks
        formatted_results = ""
        results = await asyncio.gather(*tasks, return_exceptions=True)
            
        # 6) Format the outputs
        for (query, source), result in zip(query_sources, results):
            if isinstance(result, Exception):
                write(f"[process_search_queries] ❌ Forecaster {forecaster_id}: Error for '{query}' → {str(result)}")
                if source == "Assistant":
                    formatted_results += f"\n<Asknews_articles>\nQuery: {query}\nError retrieving results: {str(result)}\n</Asknews_articles>\n"
                elif source in ("Agent", "Perplexity"):
                    formatted_results += f"\n<Agent_report>\nQuery: {query}\nError: {str(result)}\n</Agent_report>\n"
                else:
                    formatted_results += f"\n<Summary query=\"{query}\">\nError retrieving results: {str(result)}\n</Summary>\n"
            else:
                write(f"[process_search_queries] ✅ Forecaster {forecaster_id}: Query '{query}' processed successfully.")
                
                if source == "Assistant":
                    formatted_results += f"\n<Asknews_articles>\nQuery: {query}\n{result}</Asknews_articles>\n"
                elif source in ("Agent", "Perplexity"):
                    formatted_results += f"\n<Agent_report>\nQuery: {query}\n{result}</Agent_report>\n"
                else:
                    formatted_results += result

        return formatted_results

    except Exception as e:
        write(f"Forecaster {forecaster_id}: Error processing search queries: {str(e)}")
        import traceback
        write(f"Traceback: {traceback.format_exc()}")
        return "Error processing some search queries. Partial results may be available."

# ========================================
# PROCESS SEARCH QUERIES LITE (FAST MODE)
# ========================================

async def process_search_queries_lite(response: str, forecaster_id: str, question_details: dict, skip_agent: bool = True):
    """
    LITE/FAST MODE: Processes search queries but skips expensive Agent/Perplexity searches.
    Only executes Google + Google News + Assistant searches.
    """
    try:
        write(f"[process_search_queries_lite] Forecaster {forecaster_id}: Fast mode (skip_agent={skip_agent})")
        
        # 1) Extract the "Search queries:" block
        search_queries_block = re.search(r'(?:Search queries:)(.*)', response, re.DOTALL | re.IGNORECASE)
        if not search_queries_block:
            write(f"Forecaster {forecaster_id}: No search queries block found")
            return ""

        queries_text = search_queries_block.group(1).strip()

        # 2) Try to find queries
        search_queries = re.findall(
            r'(?:\d+\.\s*)?(["\']?(.*?)["\']?)\s*\((Google|Google News|Assistant|Agent|Perplexity)\)',
            queries_text
        )
        if not search_queries:
            search_queries = re.findall(
                r'(?:\d+\.\s*)?([^(\n]+)\s*\((Google|Google News|Assistant|Agent|Perplexity)\)',
                queries_text
            )

        if not search_queries:
            write(f"Forecaster {forecaster_id}: No valid search queries found")
            return ""

        write(f"Forecaster {forecaster_id}: Processing {len(search_queries)} search queries (lite mode)")

        # 4) Filter and execute queries
        tasks = []
        query_sources = []
        
        for match in search_queries:
            if len(match) == 3:
                _, raw_query, source = match
            else:
                raw_query, source = match

            query = raw_query.strip().strip('"').strip("'")
            if not query:
                continue

            # SKIP Agent/Perplexity in fast mode
            if skip_agent and source in ("Agent", "Perplexity"):
                write(f"Forecaster {forecaster_id}: ⏭️ SKIPPING {source} query in fast mode: '{query}'")
                continue
            
            write(f"Forecaster {forecaster_id}: Query='{query}' Source={source}")
            query_sources.append((query, source))

            if source in ("Google", "Google News"):
                tasks.append(
                    google_search_and_scrape(
                        query,
                        is_news=(source == "Google News"),
                        question_details=question_details,
                        date_before=question_details.get("resolution_date")
                    )
                )
            elif source == "Assistant":
                tasks.append(call_asknews(query))

        if not tasks:
            write(f"Forecaster {forecaster_id}: No tasks generated (all skipped in fast mode)")
            return ""

        # 5) Await all tasks
        formatted_results = ""
        results = await asyncio.gather(*tasks, return_exceptions=True)
            
        # 6) Format outputs
        for (query, source), result in zip(query_sources, results):
            if isinstance(result, Exception):
                write(f"[process_search_queries_lite] ❌ Forecaster {forecaster_id}: Error for '{query}' → {str(result)}")
                if source == "Assistant":
                    formatted_results += f"\n<Asknews_articles>\nQuery: {query}\nError: {str(result)}\n</Asknews_articles>\n"
                else:
                    formatted_results += f"\n<Summary query=\"{query}\">\nError: {str(result)}\n</Summary>\n"
            else:
                write(f"[process_search_queries_lite] ✅ Forecaster {forecaster_id}: Query '{query}' processed")
                
                if source == "Assistant":
                    formatted_results += f"\n<Asknews_articles>\nQuery: {query}\n{result}</Asknews_articles>\n"
                else:
                    formatted_results += result

        return formatted_results

    except Exception as e:
        write(f"Forecaster {forecaster_id}: Error in lite mode: {str(e)}")
        import traceback
        write(f"Traceback: {traceback.format_exc()}")
        return "Error processing search queries in lite mode."

# ========================================
# MAIN (for testing)
# ========================================

async def main():
    """
    Demonstrates the usage of process_search_queries with sample search queries.
    """
    print("Starting test for content extraction...")
    
    sample_response = """
    Search queries:
    1. "Nvidia stock price forecast 2024" (Google)
    2. "Ukraine Russia conflict latest developments" (Google News)
    3. "Middle East stability assessment Israel Hamas" (Perplexity)
    4. "Trump tariffs economic impact" (Assistant)
    """
    
    forecaster_id = "demo_forecaster"
    print(f"Processing sample search queries for forecaster: {forecaster_id}")
    
    question_details = {
        "title": "Sample Question",
        "resolution_criteria": "Sample resolution criteria",
        "fine_print": "Sample fine print",
        "description": "Sample background information",
        "resolution_date": "2025-12-31"
    }
    
    results = await process_search_queries(sample_response, forecaster_id, question_details)
    
    print("\n=== SEARCH RESULTS ===\n")
    print(results)
    print("\n=== END OF RESULTS ===\n")

if __name__ == "__main__":
    asyncio.run(main())
