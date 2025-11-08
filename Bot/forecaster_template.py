import datetime
import re
import numpy as np
import os
from aiohttp import ClientSession, ClientTimeout
import dotenv
from search import call_asknews, call_perplexity
from llm_calls import call_claude
from openai import OpenAI
import asyncio

def write(x):
    print(x)

dotenv.load_dotenv()

METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

NUM_RUNS = 5

BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Question background:
{background}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}

Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time. Think deeply about the question and approach it from multiple possible viewpoints.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

The options are: {options}

Background:
{background}

{resolution_criteria}

{fine_print}

Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options}. Format your output **EXACTLY** as below, ensuring that the **probabilities are between 0 and 100, sum to 100, and are not followed by a % sign**:

Probabilities: [Probability_1, Probability_2, ..., Probability_N]
"""

async def run_research(question: str, write=print) -> str:
    research = ""
    if ASKNEWS_CLIENT_ID and ASKNEWS_SECRET:
        prompt = f"Please fetch all news articles relevant to this forecasting question: {question}"
        research = await call_asknews(question)

    prompt = f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}
            """
    
    pplx = call_perplexity(prompt)
    research += pplx

    write(f"########################\nResearch Found:\n{research}\n########################")

    return research

# Calls o4-mini using personal OpenAI credentials
async def call_llm(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model="o4-mini",
        input= prompt
    )
    return response.output_text


async def call_gpt(prompt):
    # We are temporarily going to short gpt while my o1 credits are out
    prompt = prompt
    try:
        url = "https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {METACULUS_TOKEN}"
        }
        
        data = {
            "model": "o1",
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


def extract_binary_probability(text: str) -> float:
    matches = re.findall(r"(\d+)%", text)
    if matches:
        prob = int(matches[-1])
        return float(np.clip(prob, 1, 99))  # clip between 1 and 99
    raise ValueError(f"Could not extract binary probability from text: {text}")


def extract_mcq_probabilities(forecast_text: str, num_options: int) -> list[float]:
    import re

    matches = re.findall(r"Probabilities:\s*\[([0-9.,\s%]+)\]", forecast_text)
    if not matches:
        raise ValueError(f"Could not extract 'Probabilities' list from response:\n{forecast_text}")
    last_match = matches[-1]

    # Parse numbers
    raw_numbers = [n.strip().replace("%", "") for n in last_match.split(",") if n.strip()]
    numbers = [float(n) for n in raw_numbers]

    # Fix % scaling if necessary
    total = sum(numbers)
    if total > 1.5 and total <= 110:  # probably in %
        numbers = [x / 100 for x in numbers]
        total = sum(numbers)

    if len(numbers) != num_options:
        raise ValueError(f"Expected {num_options} probabilities, got {len(numbers)}: {numbers}")
    if not 0.98 <= total <= 1.02:
        raise ValueError(f"Probabilities do not sum to 1: {numbers}")
    
    # Normalize if close but off
    numbers = [x / total for x in numbers]
    return numbers

def format_binary_prompt(details: dict, summary: str) -> str:
    return BINARY_PROMPT_TEMPLATE.format(
        title=details["title"],
        today=datetime.datetime.now().strftime("%Y-%m-%d"),
        background=details.get("description", ""),
        resolution_criteria=details.get("resolution_criteria", ""),
        fine_print=details.get("fine_print", ""),
        summary_report=summary
    )

def format_mcq_prompt(details: dict, summary: str) -> str:
    return MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=details["title"],
        today=datetime.datetime.now().strftime("%Y-%m-%d"),
        background=details.get("description", ""),
        resolution_criteria=details.get("resolution_criteria", ""),
        fine_print=details.get("fine_print", ""),
        summary_report=summary,
        options=details.get("options", [])
    )

async def binary_forecast(details: dict, write=print) -> tuple[float, str]:
    summary = await run_research(details["title"], write)
    prompt = format_binary_prompt(details, summary)
    responses = await asyncio.gather(*[call_llm(prompt) for _ in range(5)])
    parsed = [extract_binary_probability(r) for r in responses]
    avg = np.mean(parsed)
    comment = f"Binary forecast (mean): {avg}%\n\n" + "\n\n".join(responses)
    write(comment)
    return avg, comment

async def multiple_choice_forecast(details: dict, write=print) -> tuple[dict[str, float], str]:
    options = details["options"]
    summary = await run_research(details["title"], write)
    prompt = format_mcq_prompt(details, summary)

    responses = await asyncio.gather(*[call_llm(prompt) for _ in range(5)])
    
    extracted = []
    outputs = []
    for i, response in enumerate(responses):
        write(f"\n=== Raw Output #{i+1} ===")
        write(response)
        write("------------------------------------------------------------------------------------------------")
        try:
            probs = extract_mcq_probabilities(response, len(options))
            extracted.append(probs)
            outputs.append(response)
        except Exception as e:
            write(f"⚠️ Error extracting probabilities: {e}")
            write("Skipping this response.\n")
    
    if not extracted:
        raise ValueError("No valid probability sets extracted.")

    avg_probs = np.mean(extracted, axis=0)
    result = {opt: float(p) for opt, p in zip(options, avg_probs)}
    comment = f"MCQ forecast (mean): {result}\n\n" + "\n\n".join(outputs)
    write(comment)
    return result, comment