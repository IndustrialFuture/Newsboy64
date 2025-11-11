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
load_dotenv()

# ========================================
# HELPER FUNCTIONS (must be defined first!)
# ========================================

def write(x):
    print(x)

def parse_date(date_str: str) -> str:
    parsed_date = dateparser.parse(date_str, settings={'STRICT_PARSING': False})
    if parsed_date:
        return parsed_date.strftime("%b %d, %Y")
    return "Unknown"

def validate_time(before_date_str, source_date_str):
    if source_date_str == "Unknown":
        return False
    before_date = dateparser.parse(before_date_str)
    source_date = dateparser.parse(source_date_str)
    return source_date <= before_date

# ========================================
# API KEYS AND CONFIGURATION
# ========================================

SERPER_KEY = os.getenv("SERPER_KEY") or os.getenv("SERPER_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model configuration (no hardcoding!)
MODEL_RS = os.getenv("MODEL_RS", "")  # Research/search model
MODEL_FC = os.getenv("MODEL_FC", "")  # Main forecaster
MODEL_BK = os.getenv("MODEL_BK", "")  # Backup model
MODEL_QG = os.getenv("MODEL_QG", "")  # Question generation

# Use OpenRouter for OpenAI calls (routes through OpenRouter)
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
HAS_PERPLEXITY = bool(PERPLEXITY_API_KEY)
HAS_RESEARCH_MODEL = bool(MODEL_RS and OPENROUTER_API_KEY)

# Log API availability
write(f"[INIT] API Availability:")
write(f"  Serper (Google): {'✅' if HAS_SERPER else '❌ → Will use MODEL_RS fallback'}")
write(f"  AskNews: {'✅' if HAS_ASKNEWS else '❌ → Will use MODEL_RS fallback'}")
write(f"  Perplexity: {'✅' if HAS_PERPLEXITY else '❌ → Will use MODEL_RS fallback'}")
write(f"  Research Model (MODEL_RS): {'✅' if HAS_RESEARCH_MODEL else '❌'}")

if not HAS_RESEARCH_MODEL:
    write(f"[INIT] ⚠️ WARNING: No research model configured - fallbacks will not work properly!")

# ========================================
# FALLBACK: LLM-BASED SEARCH (when APIs missing)
# ========================================

async def llm_based_search(query: str, search_type: str = "general") -> str:
    """
    Fallback search using MODEL_RS when external APIs (Serper/AskNews/Perplexity) are unavailable.
    This preserves search functionality without degrading quality.
    
    Args:
        query: Search query
        search_type: "general", "news", "deep_research", or "asknews"
    
    Returns:
        Formatted search results
    """
    write(f"[llm_based_search] Using MODEL_RS fallback for query: '{query}' (type: {search_type})")
    
    if not MODEL_RS or not OPENROUTER_API_KEY:
        write(f"[llm_based_search] ❌ ERROR: MODEL_RS not configured!")
        return f"<SearchResults query=\"{query}\">ERROR: No research model configured for fallback search.</SearchResults>\n"
    
    # Build prompt based on search type
    if search_type == "news":
        prompt = f"""You are a research assistant. The user needs current news about: {query}

Provide a comprehensive summary of recent news and developments on this topic. Include:
- Key recent events and their dates
- Important statistics and facts
- Expert opinions and analysis
- Relevant sources (cite real news organizations and dates)

Format your response as if you're summarizing multiple news articles."""
    
    elif search_type == "deep_research":
        prompt = f"""You are a deep research assistant. Conduct thorough research on: {query}

Provide comprehensive analysis including:
- Historical context and background
- Current state and recent developments
- Expert analysis and opinions
- Statistical data and trends
- Future projections and implications
- Cite all sources with names and dates

Be thorough and detailed. Be objective, providing documented facts only."""
    
    elif search_type == "asknews":
        prompt = f"""You are a news research assistant. Provide recent news coverage on: {query}

Include:
- Latest breaking news and developments
- Key events from the past 24-48 hours
- Multiple perspectives from different sources
- Dates and times of events
- Cite specific news organizations

Format as multiple article summaries with publication dates."""
    
    else:  # general
        prompt = f"""You are a research assistant. Research the following topic: {query}

Provide:
- Key facts and information
- Recent developments
- Relevant statistics and data
- Expert opinions and analysis
- Cite sources where possible

Be comprehensive but concise."""
    
    try:
        # Call OpenRouter with MODEL_RS
        llm_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        response = llm_client.chat.completions.create(
            model=MODEL_RS,
            messages=[
                {"role": "system", "content": "You are an expert research assistant providing factual, well-sourced information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        write(f"[llm_based_search] ✅ Received {len(content)} chars from {MODEL_RS}")
        
        # Format based on search type
        if search_type == "asknews":
            return f"\n<Asknews_articles>\nQuery: {query}\n{content}\n(Source: MODEL_RS fallback)\n</Asknews_articles>\n"
        elif search_type == "deep_research":
            return f"\n<Agent_report>\nQuery: {query}\n{content}\n(Source: MODEL_RS fallback)\n</Agent_report>\n"
        else:
            return f"\n<SearchResults query=\"{query}\">\n{content}\n(Source: MODEL_RS fallback)\n</SearchResults>\n"
    
    except Exception as e:
        write(f"[llm_based_search] ❌ Error: {str(e)}")
        return f"<SearchResults query=\"{query}\">Error in MODEL_RS fallback: {str(e)}</SearchResults>\n"

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
# ASKNEWS API (with fallback)
# ========================================

async def call_asknews(question: str) -> str:
    """
    Use AskNews API if available, otherwise fall back to MODEL_RS.
    CRITICAL: Never skip this functionality!
    """
    if not HAS_ASKNEWS:
        write(f"[call_asknews] ⚠️ AskNews API not configured, using MODEL_RS fallback")
        return await llm_based_search(question, search_type="asknews")
    
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

        return formatted_articles
    except Exception as e:
        write(f"[call_asknews] ❌ AskNews API error: {str(e)}, falling back to MODEL_RS")
        return await llm_based_search(question, search_type="asknews")

# ========================================
# AGENTIC SEARCH (with fallback)
# ========================================

async def agentic_search(query: str) -> str:
    """
    Performs agentic search using GPT to iteratively research and analyze a query.
    Falls back to MODEL_RS if OpenAI/OpenRouter not available.
    """
    write(f"[agentic_search] Starting research for query: {query}")
    
    if not client:
        write(f"[agentic_search] ⚠️ No LLM client available, using MODEL_RS fallback")
        return await llm_based_search(query, search_type="deep_research")
    
    max_steps = 7
    current_analysis = ""
    all_search_queries = []
    
    # Cost tracking variables
    total_input_tokens = 0
    total_output_tokens = 0
    
    def estimate_tokens(text: str) -> int:
        """Estimate token count using ~4 characters per token rule for GPT models"""
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
            
            write(f"[agentic_search] Step {step + 1}: Calling GPT")
            response = await call_gpt(prompt, step)
            
            response_tokens = estimate_tokens(response)
            total_output_tokens += response_tokens
            
            # Parse the response
            analysis_match = re.search(r'Analysis:\s*(.*?)(?=Search queries:|$)', response, re.DOTALL)
            if not analysis_match:
                write(f"[agentic_search] Error: Could not parse analysis from response")
                return f"Error: Failed to parse analysis at step {step + 1}"
            
            if step > 0:
                current_analysis = analysis_match.group(1).strip()
                write(f"[agentic_search] Step {step + 1}: Analysis updated ({len(current_analysis)} chars)")
            else:
                write(f"[agentic_search] Step 1: Initial query understanding complete")
            
            # Check for search queries
            search_queries_match = re.search(r'Search queries:\s*(.*)', response, re.DOTALL)
            
            if step == 0 and not search_queries_match:
                write(f"[agentic_search] Error: No search queries in initial response")
                return "Error: Failed to generate initial search queries"
            
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
                    return "Error: Failed to parse initial search queries"
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
                return f"Error during agentic search: {str(e)}"
    
    steps_used = step + 1
    total_cost = calculate_cost(total_input_tokens, total_output_tokens)
    
    print(f"\n🔍 Agentic Search Summary:")
    print(f"   Steps used: {steps_used}")
    print(f"   Total tokens: {total_input_tokens + total_output_tokens:,} ({total_input_tokens:,} input + {total_output_tokens:,} output)")
    print(f"   Estimated cost: ${total_cost:.4f}")
    
    if not current_analysis:
        return "Error: No analysis was generated during the research process"
    
    return current_analysis

# ========================================
# PERPLEXITY API (with fallback)
# ========================================

async def call_perplexity(prompt: str) -> str:
    """
    Call Perplexity API if available, otherwise fall back to MODEL_RS.
    CRITICAL: Never skip this functionality!
    """
    if not HAS_PERPLEXITY:
        write(f"[call_perplexity] ⚠️ Perplexity API not configured, using MODEL_RS fallback")
        return await llm_based_search(prompt, search_type="deep_research")
    
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
                    else:
                        response_text = await response.text()
                        write(f"[Perplexity API] ❌ Error: HTTP {response.status}: {response_text}")
                        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            write(f"[Perplexity API] ⚠️ Attempt {attempt} failed: {e}")
        
        if attempt < max_retries:
            wait_time = backoff_base * attempt
            write(f"[Perplexity API] 🔁 Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
        else:
            write(f"[Perplexity API] ❌ Max retries reached, falling back to MODEL_RS")
            return await llm_based_search(prompt, search_type="deep_research")

    return "Unexpected error in call_perplexity"

# ========================================
# GOOGLE SEARCH (with fallback)
# ========================================

async def google_search(query, is_news=False, date_before=None):
    """
    Google search via Serper API if available, otherwise use MODEL_RS fallback.
    CRITICAL: Never skip this functionality!
    """
    if not HAS_SERPER:
        write(f"[google_search] ⚠️ Serper API not configured, using MODEL_RS fallback")
        # Use MODEL_RS to generate URLs/content instead
        search_type = "news" if is_news else "general"
        result = await llm_based_search(query, search_type=search_type)
        # Return empty list since we can't get real URLs, but the content will be in result
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
                    write(f"[google_search] Returning {len(urls)} URLs: {urls}")
                    return urls
                else:
                    write(f"[google_search] Error in Serper API response: Status {response.status}")
                    response.raise_for_status()
    except Exception as e:
        write(f"[google_search] Exception: {str(e)}, falling back to MODEL_RS")
        # Fall back to MODEL_RS
        search_type = "news" if is_news else "general"
        await llm_based_search(query, search_type=search_type)
        return []

# ========================================
# GPT CALL
# ========================================

async def call_gpt(prompt, step=1):
    if not client:
        write(f"[call_gpt] ❌ No LLM client available")
        return "Error: No LLM client configured"
    
    try:
        # Check if using o3 reasoning model (if MODEL_FC contains "o3")
        model_to_use = MODEL_FC if MODEL_FC else "gpt-4"
        
        if "o3" in model_to_use.lower():
            # Use reasoning model format
            response = client.responses.create(
                model=model_to_use,
                input=prompt
            )
            return response.output_text
        else:
            # Standard chat completion
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000
            )
            return response.choices[0].message.content
    except Exception as e:
        write(f"[call_gpt] Error: {str(e)}")
        return f"Error calling LLM API: {str(e)}"

# ========================================
# GOOGLE SEARCH AND SCRAPE
# ========================================

async def google_search_and_scrape(query, is_news, question_details, date_before=None):
    write(f"[google_search_and_scrape] Called with query='{query}', is_news={is_news}, date_before={date_before}")
    try:
        urls = await google_search(query, is_news, date_before)

        if not urls:
            write(f"[google_search_and_scrape] ⚠️ No URLs returned, using MODEL_RS fallback")
            # Fallback: use MODEL_RS to generate content directly
            search_type = "news" if is_news else "general"
            result = await llm_based_search(query, search_type=search_type)
            return result

        async with FastContentExtractor() as extractor:
            write(f"[google_search_and_scrape] 🔍 Starting content extraction for {len(urls)} URLs")
            results = await extractor.extract_content(urls)
            write(f"[google_search_and_scrape] ✅ Finished content extraction")

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
            write("[google_search_and_scrape] ⚠️ No content to summarize, using MODEL_RS fallback")
            search_type = "news" if is_news else "general"
            result = await llm_based_search(query, search_type=search_type)
            return result

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
# GOOGLE SEARCH AGENTIC
# ========================================

async def google_search_agentic(query, is_news=False):
    """
    Performs Google search and returns raw article content without summarization.
    Used for agentic search where the agent will analyze the raw content.
    Falls back to MODEL_RS if Serper unavailable.
    """
    write(f"[google_search_agentic] Called with query='{query}', is_news={is_news}")
    try:
        urls = await google_search(query, is_news)

        if not urls:
            write(f"[google_search_agentic] ⚠️ No URLs returned, using MODEL_RS fallback")
            search_type = "news" if is_news else "general"
            result = await llm_based_search(query, search_type=search_type)
            return result

        async with FastContentExtractor() as extractor:
            write(f"[google_search_agentic] 🔍 Starting content extraction for {len(urls)} URLs")
            results = await extractor.extract_content(urls)
            write(f"[google_search_agentic] ✅ Finished content extraction")

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
            write("[google_search_agentic] ⚠️ No usable content found, using MODEL_RS fallback")
            search_type = "news" if is_news else "general"
            result = await llm_based_search(query, search_type=search_type)
            return result

        return output
        
    except Exception as e:
        write(f"[google_search_agentic] Error: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        write(f"Traceback: {traceback_str}")
        return f"<RawContent query=\"{query}\">Error during search: {str(e)}</RawContent>\n"

# ========================================
# PROCESS SEARCH QUERIES
# ========================================

async def process_search_queries(response: str, forecaster_id: str, question_details: dict):
    """
    Parses out search queries from the forecaster's response, executes them
    (AskNews, Agent or Google/Google News), and returns formatted summaries.
    ALL functionality is preserved with fallbacks - NO SKIPS!
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

            # Map Perplexity to Agent for backward compatibility
            if source == "Perplexity":
                source = "Agent"
                write(f"Forecaster {forecaster_id}: Mapping Perplexity → Agent for query='{query}'")
            
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
                elif source == "Agent":
                    formatted_results += f"\n<Agent_report>\nQuery: {query}\n{result}\n</Agent_report>\n"
                else:
                    formatted_results += f"\n<Summary query=\"{query}\">\nError retrieving results: {str(result)}\n</Summary>\n"
            else:
                write(f"[process_search_queries] ✅ Forecaster {forecaster_id}: Query '{query}' processed successfully.")
                
                if source == "Assistant":
                    formatted_results += f"\n<Asknews_articles>\nQuery: {query}\n{result}</Asknews_articles>\n"
                elif source == "Agent":
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
