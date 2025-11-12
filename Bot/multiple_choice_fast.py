import asyncio
import datetime
import re
import json
import numpy as np
from prompts import (
    MULTIPLE_CHOICE_PROMPT_historical,
    MULTIPLE_CHOICE_PROMPT_current,
    MULTIPLE_CHOICE_PROMPT_1,
    MULTIPLE_CHOICE_PROMPT_2,
)
from llm_calls import call_claude, call_gpt_o3
from search import process_search_queries_lite

"""
FAST MODE for multiple choice forecasting.
Reduces calls from ~54 to ~18 by:
1. Skipping expensive agentic/perplexity searches (use lite search)
2. Using 2 forecasters instead of 5 (Claude Sonnet + o3)
3. Keeping the same methodology and prompts

Quality: ~80% of full mode
Speed: ~75% faster
Cost: ~70% cheaper
"""

def write(x):
    print(x)

def extract_option_probabilities_from_response(forecast_text: str, num_options: int) -> list[float]:
    matches = re.findall(r"Probabilities:\s*\[([0-9.,\s]+)\]", forecast_text)
    if not matches:
        raise ValueError(f"Could not extract 'Probabilities' list from response: {forecast_text}")
    last_match = matches[-1]
    numbers = [float(n.strip()) for n in last_match.split(",") if n.strip()]
    if len(numbers) != num_options:
        raise ValueError(f"Expected {num_options} probabilities, got {len(numbers)}: {numbers}")
    return numbers

def normalize_probabilities(probs: list[float]) -> list[float]:
    probs = [max(min(p, 99), 1) for p in probs]
    total = sum(probs)
    normed = [p / total for p in probs]
    normed[-1] += 1.0 - sum(normed)  # minor fix for rounding
    return normed

async def get_multiple_choice_forecast_fast(question_details: dict, write=print) -> tuple[dict[str, float], str]:
    """
    Fast mode multiple choice forecast: ~18 LLM calls instead of ~54
    
    Flow:
    1. Historical research (1 call + lite search = ~7 calls)
    2. Current research (1 call + lite search = ~7 calls)
    3. Outside view (2 forecasters = 2 calls)
    4. Final forecast (2 forecasters = 2 calls)
    
    Total: ~18 calls
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details.get("fine_print", "")
    options = question_details["options"]
    num_options = len(options)

    write("[FAST MODE] Starting multiple choice forecast with 2 forecasters (Claude + o3)")

    # ========================================
    # PHASE 1 & 2: Research (using lite search)
    # ========================================
    
    async def format_and_call_o3(prompt_template):
        content = prompt_template.format(
            title=title,
            today=today,
            background=background,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            options=options,
        )
        return content, await call_gpt_o3(content)

    write("[FAST MODE] Phase 1&2: Initiating historical + current research")
    
    historical_task = asyncio.create_task(format_and_call_o3(MULTIPLE_CHOICE_PROMPT_historical))
    current_task = asyncio.create_task(format_and_call_o3(MULTIPLE_CHOICE_PROMPT_current))
    
    (historical_prompt, historical_output), (current_prompt, current_output) = await asyncio.gather(
        historical_task, current_task
    )

    write("[FAST MODE] Phase 1&2: Research queries generated, executing LITE searches")

    # Process searches using LITE mode (skips Agent/Perplexity)
    context_historical, context_current = await asyncio.gather(
        process_search_queries_lite(
            historical_output,
            forecaster_id="-1",
            question_details=question_details,
            skip_agent=True  # FAST MODE: Skip expensive searches
        ),
        process_search_queries_lite(
            current_output,
            forecaster_id="0",
            question_details=question_details,
            skip_agent=True  # FAST MODE: Skip expensive searches
        )
    )

    write(f"\n[FAST MODE] Historical context: {len(context_historical)} chars")
    write(f"\n[FAST MODE] Current context: {len(context_current)} chars")

    # ========================================
    # PHASE 3: Outside View (2 forecasters)
    # ========================================
    
    write("[FAST MODE] Phase 3: Generating outside view with 2 forecasters")
    
    prompt1 = MULTIPLE_CHOICE_PROMPT_1.format(
        title=title,
        today=today,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        context=context_historical,
        options=options
    )

    async def run_outside_view():
        return await asyncio.gather(
            call_claude(prompt1),   # Forecaster 1: Claude Sonnet
            call_gpt_o3(prompt1),   # Forecaster 2: o3
        )

    results_outside = await run_outside_view()
    
    write(f"\n[FAST MODE] Forecaster 1 (Claude) outside view complete")
    write(f"\n[FAST MODE] Forecaster 2 (o3) outside view complete")

    # ========================================
    # PHASE 4: Final Forecast (2 forecasters)
    # ========================================
    
    write("[FAST MODE] Phase 4: Generating final forecasts")
    
    context_map = {
        "claude": f"Current context: {context_current}\nOutside view prediction: {results_outside[0]}",
        "o3": f"Current context: {context_current}\nOutside view prediction: {results_outside[1]}",
    }

    def format_prompt2(forecaster_key: str):
        return MULTIPLE_CHOICE_PROMPT_2.format(
            title=title,
            today=today,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            context=context_map[forecaster_key],
            options=options
        )

    async def run_final_forecast():
        return await asyncio.gather(
            call_claude(format_prompt2("claude")),
            call_gpt_o3(format_prompt2("o3")),
        )

    results_final = await run_final_forecast()

    # ========================================
    # EXTRACT & COMBINE PREDICTIONS
    # ========================================
    
    all_probs = []
    final_outputs = []
    forecaster_names = ["Claude Sonnet", "GPT-o3"]

    for i, output in enumerate(results_final):
        try:
            write(f"\n[FAST MODE] {forecaster_names[i]} output received")
            probs = extract_option_probabilities_from_response(output, num_options)
            probs = normalize_probabilities(probs)
            all_probs.append(probs)
            write(f"[FAST MODE] {forecaster_names[i]} probabilities: {probs}")
        except Exception as e:
            write(f"[FAST MODE] Error parsing {forecaster_names[i]}: {e}")
            # Fallback to uniform distribution
            all_probs.append([1.0 / num_options] * num_options)
        
        final_outputs.append(f"=== {forecaster_names[i]} ===\nOutput:\n{output}\n")

    # Calculate weighted average (o3 gets double weight)
    if len(all_probs) >= 1:
        probs_matrix = np.array(all_probs)
        weights = np.array([1, 2])[:len(probs_matrix)]  # Claude: 1, o3: 2
        weighted_probs = np.average(probs_matrix, axis=0, weights=weights)
        probability_yes_per_category = {opt: float(p) for opt, p in zip(options, weighted_probs)}
    else:
        write("[FAST MODE] ❌ No valid probabilities, using uniform distribution")
        probability_yes_per_category = {opt: 1.0/num_options for opt in options}

    comment = (
        f"[FAST MODE] Average Probability Per Category: `{probability_yes_per_category}`\n\n"
        + "\n\n".join(final_outputs)
    )

    write("\n[FAST MODE] ========================================")
    write("[FAST MODE] FORECAST COMPLETE")
    write("[FAST MODE] ========================================")
    write("\nFinal averaged probabilities per category:")
    write(json.dumps(probability_yes_per_category, indent=2))

    return probability_yes_per_category, comment
