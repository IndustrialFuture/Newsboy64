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
This file contains the main forecasting logic, question-type specific functions are abstracted.
"""
def write(x):
    print(x)



load_dotenv()
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def call_anthropic_api(prompt, max_tokens=16000, max_retries=7, cached_content=claude_context):
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
            write(f"Starting API call attempt {attempt + 1}")
            timeout = ClientTimeout(total=300)  # 5 minutes total timeout
            
            async with ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        write(f"API error (status {response.status}): {error_text}")
                        
                        if response.status in [429, 503]:  # Rate limit or service unavailable
                            write(f"Retryable error. Waiting {backoff_delay} seconds...")
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
                    
                    print(f"Claude's thinking: {thinking}")
                    return text
                    
                    write("No 'text' block found in content.")
                    return "No final answer found in Claude response."
                        
        except (ClientError, asyncio.TimeoutError) as e:
            write(f"Retryable error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff_delay)
            
        except Exception as e:
            write(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff_delay)

    raise Exception(f"Failed after {max_retries} attempts")


async def call_claude(prompt):
    try:
        response = await call_anthropic_api(prompt)
        
        if not response:
            write("Warning: Empty response from Anthropic API")
            return "API returned empty response"
            
        return response
        
    except Exception as e:
        write(f"Error in call_claude: {str(e)}")
        return f"Error generating response: {str(e)}"
    

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
        exec(python_code, {})  # use isolated globals
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Error executing the extracted Python code:\n{tb}"
    finally:
        sys.stdout = old_stdout

    return new_stdout.getvalue()

# Calls o4-mini using personal OpenAI credentials
async def call_gpt(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model="o4-mini",
        input= gpt_context + "\n" + prompt
    )
    return response.output_text

async def call_gpt_o3_personal(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model="o3",
        input= gpt_context + "\n" + prompt
    )
    return response.output_text


async def call_gpt_o3(prompt):
    # Temporarily short metaculus proxy using personal credits.
    ans = await call_gpt_o3_personal(prompt)
    return ans
    try:
        url = "https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {METACULUS_TOKEN}"
        }
        
        data = {
            "model": "o3",
            "messages": [{"role": "user", "content": prompt}],
        }
        
        timeout = ClientTimeout(total=300)  # 5 minutes total timeout
        
        async with ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    write(f"API error (status {response.status}): {error_text}")
                    response.raise_for_status()
                
                result = await response.json()
                
                answer = result['choices'][0]['message']['content']
                if answer is None:
                    raise ValueError("No answer returned from GPT")
                return answer
                
    except Exception as e:
        write(f"Error in call_gpt: {str(e)}")
        return f"Error generating response: {str(e)}"


async def call_gpt_o4_mini(prompt):
    prompt = gpt_context + "\n" + prompt
    try:
        url = "https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {METACULUS_TOKEN}"
        }
        
        data = {
            "model": "o4-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        
        timeout = ClientTimeout(total=300)  # 5 minutes total timeout
        
        async with ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    write(f"API error (status {response.status}): {error_text}")
                    response.raise_for_status()
                
                result = await response.json()
                
                answer = result['choices'][0]['message']['content']
                if answer is None:
                    raise ValueError("No answer returned from GPT")
                return answer
                
    except Exception as e:
        write(f"Error in call_gpt: {str(e)}")
        return f"Error generating response: {str(e)}"
    
