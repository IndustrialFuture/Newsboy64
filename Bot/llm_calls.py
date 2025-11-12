import asyncio
import numpy as np
import os
from aiohttp import ClientSession, ClientTimeout, ClientError
import json
import sys
from openai import OpenAI
import re
import io
from dotenv import load_dotenv
from prompts import claude_context, gpt_context
"""
LLM calls centralized - routes through OpenRouter with Metaculus proxy fallback.
"""
def write(x):
    print(x)

load_dotenv()

# ========================================
# API KEYS AND CLIENT SETUP
# ========================================

METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Create OpenAI client with OpenRouter
def get_openai_client():
    """
    Get OpenAI client, using OpenRouter if available.
    This allows all OpenAI calls to route through OpenRouter.
    """
    if OPENROUTER_API_KEY:
        write(f"[llm_calls] Using OpenRouter for OpenAI calls")
        return OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    elif OPENAI_API_KEY:
        write(f"[llm_calls] Using OpenAI directly")
        return OpenAI(api_key=OPENAI_API_KEY)
    else:
        raise ValueError("No OpenAI or OpenRouter API key found")

# ========================================
# CLAUDE CALLS (OpenRouter or Metaculus proxy)
# ========================================

async def call_claude_via_openrouter(prompt):
    """
    Call Claude via OpenRouter.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")
    
    try:
        write("[call_claude_via_openrouter] Calling Claude 3.5 Sonnet via OpenRouter")
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        response = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": claude_context},
                {"role": "user", "content": prompt}
            ],
            max_tokens=16000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        write(f"[call_claude_via_openrouter] Error: {str(e)}")
        raise

async def call_anthropic_api(prompt, max_tokens=16000, max_retries=7, cached_content=claude_context):
    """
    Call Claude via Metaculus proxy (requires METACULUS_TOKEN).
    Used as fallback when OpenRouter unavailable.
    """
    if not METACULUS_TOKEN:
        raise ValueError("METACULUS_TOKEN not set - cannot use Metaculus proxy fallback")
    
    url = "https://llm-proxy.metaculus.com/proxy/anthropic/v1/messages/"
    headers = {
        "Authorization": f"Token {METACULUS_TOKEN}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "anthropic-metadata": json.dumps({
            "task_type": "qualitative_forecasting",
            "emphasis": "detailed_reasoning"
        })
    }
    
    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "thinking" : {
            "type": "enabled",
            "budget_tokens": 12000
        },
        "system": [
            {
                "type": "text",
                "text": cached_content,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    for attempt in range(max_retries):
        backoff_delay = min(2 ** attempt, 60)
        
        try:
            write(f"[Metaculus proxy] Starting API call attempt {attempt + 1}")
            timeout = ClientTimeout(total=300)
            
            async with ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        write(f"[Metaculus proxy] API error (status {response.status}): {error_text}")
                        
                        if response.status in [429, 503]:
                            write(f"[Metaculus proxy] Retryable error. Waiting {backoff_delay} seconds...")
                            await asyncio.sleep(backoff_delay)
                            continue
                            
                        response.raise_for_status()
                    
                    result = await response.json()
                    text = ""
                    thinking = ""
                    for block in result.get("content", []):
                        if block.get("type") == "text":
                           text = block.get("text")
                        if block.get("type") == "thinking":
                            thinking = block.get("thinking")
                    
                    if thinking:
                        print(f"[Metaculus proxy] Claude's thinking: {thinking[:200]}...")
                    return text
                        
        except (ClientError, asyncio.TimeoutError) as e:
            write(f"[Metaculus proxy] Retryable error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff_delay)
            
        except Exception as e:
            write(f"[Metaculus proxy] Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff_delay)

    raise Exception(f"[Metaculus proxy] Failed after {max_retries} attempts")


async def call_claude(prompt):
    """
    Call Claude - use OpenRouter if available, otherwise Metaculus proxy fallback.
    """
    if OPENROUTER_API_KEY:
        try:
            write("[call_claude] Using OpenRouter")
            return await call_claude_via_openrouter(prompt)
        except Exception as e:
            write(f"[call_claude] OpenRouter failed: {e}")
            if METACULUS_TOKEN:
                write("[call_claude] Falling back to Metaculus proxy")
                return await call_anthropic_api(prompt)
            else:
                write("[call_claude] No Metaculus token available for fallback")
                raise
    else:
        write("[call_claude] Using Metaculus proxy (no OpenRouter key)")
        if not METACULUS_TOKEN:
            raise ValueError("Neither OPENROUTER_API_KEY nor METACULUS_TOKEN set")
        return await call_anthropic_api(prompt)

# ========================================
# UTILITY FUNCTIONS
# ========================================

def extract_and_run_python_code(llm_output: str) -> str:
    pattern = re.compile(r'<python>(.*?)</python>', re.DOTALL)
    matches = pattern.findall(llm_output)

    if not matches:
        return "No <python> block found."

    python_code = matches[0].strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(python_code, {})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Error executing the extracted Python code:\n{tb}"
    finally:
        sys.stdout = old_stdout

    return new_stdout.getvalue()

# ========================================
# GPT CALLS (via OpenRouter or OpenAI)
# ========================================

async def call_gpt(prompt):
    """Call o4-mini via OpenRouter/OpenAI"""
    client = get_openai_client()
    response = client.responses.create(
        model="openai/o4-mini",
        input= gpt_context + "\n" + prompt
    )
    return response.output_text

async def call_gpt_o3_personal(prompt):
    """Call o3 via OpenRouter/OpenAI"""
    client = get_openai_client()
    response = client.responses.create(
        model="openai/o3-mini",  # ← When ready hange this to model="openai/o3",
        input= gpt_context + "\n" + prompt
    )
    return response.output_text


async def call_gpt_o3(prompt):
    """
    Call o3 - use personal OpenRouter/OpenAI, with Metaculus proxy fallback
    """
    try:
        ans = await call_gpt_o3_personal(prompt)
        return ans
    except Exception as e:
        write(f"[call_gpt_o3] OpenRouter/OpenAI failed: {e}")
        if METACULUS_TOKEN:
            write("[call_gpt_o3] Falling back to Metaculus proxy")
            return await call_gpt_o3_metaculus(prompt)
        else:
            write("[call_gpt_o3] No Metaculus token available for fallback")
            raise

async def call_gpt_o3_metaculus(prompt):
    """Metaculus proxy fallback for o3"""
    if not METACULUS_TOKEN:
        raise ValueError("METACULUS_TOKEN not set - cannot use Metaculus proxy fallback")
    
    prompt = gpt_context + "\n" + prompt
    
    url = "https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {METACULUS_TOKEN}"
    }
    
    data = {
        "model": "o3",
        "messages": [{"role": "user", "content": prompt}],
    }
    
    timeout = ClientTimeout(total=300)
    
    async with ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                write(f"[Metaculus proxy o3] API error (status {response.status}): {error_text}")
                response.raise_for_status()
            
            result = await response.json()
            
            answer = result['choices'][0]['message']['content']
            if answer is None:
                raise ValueError("No answer returned from GPT")
            return answer


async def call_gpt_o4_mini(prompt):
    """
    Call o4-mini - use personal OpenRouter/OpenAI, with Metaculus proxy fallback
    """
    prompt = gpt_context + "\n" + prompt
    
    try:
        client = get_openai_client()
        response = client.responses.create(
            model="openai/o4-mini",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        write(f"[call_gpt_o4_mini] OpenRouter/OpenAI failed: {e}")
        if METACULUS_TOKEN:
            write("[call_gpt_o4_mini] Falling back to Metaculus proxy")
            return await call_gpt_o4_mini_metaculus(prompt)
        else:
            write("[call_gpt_o4_mini] No Metaculus token available for fallback")
            raise

async def call_gpt_o4_mini_metaculus(prompt):
    """Metaculus proxy fallback for o4-mini"""
    if not METACULUS_TOKEN:
        raise ValueError("METACULUS_TOKEN not set - cannot use Metaculus proxy fallback")
    
    url = "https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {METACULUS_TOKEN}"
    }
    
    data = {
        "model": "o4-mini",
        "messages": [{"role": "user", "content": prompt}],
    }
    
    timeout = ClientTimeout(total=300)
    
    async with ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                write(f"[Metaculus proxy o4-mini] API error (status {response.status}): {error_text}")
                response.raise_for_status()
            
            result = await response.json()
            
            answer = result['choices'][0]['message']['content']
            if answer is None:
                raise ValueError("No answer returned from GPT")
            return answer
