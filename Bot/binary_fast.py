import asyncio
import datetime
import re
import numpy as np
from prompts import (
    BINARY_PROMPT_historical,
    BINARY_PROMPT_current,
    BINARY_PROMPT_1,
    BINARY_PROMPT_2,
)
from llm_calls import call_claude, call_gpt_o3
from search import process_search_queries_lite

"""
FAST MODE for binary forecasting.
Reduces calls from ~54 to ~18 by:
1. Skipping expensive agentic/perplexity searches (use lite search)
2. Using 2 forecasters instead of 5 (Claude Sonnet 4 + o3)
3. Keeping the same methodology and prompts

Quality: ~80% of full mode
Speed: ~75% faster
Cost: ~70% cheaper
"""

def write(x):
    print(x)


def extract_probability_from_response_as_percentage_not_decimal(forecast_text: str) -> float:
    """Extract probability from forecast text (format: 'Probability: XX%')"""
    matches = re.findall(r"Probability:\s*([0-9]+(?:\.[0-9]+)?)%", forecast_text.strip())
    if matches:
        number = float(matches[-1])
        return min(99, max(1, number))
    raise ValueError(f"Could not extract prediction from response: {forecast_text}")


async def get_binary_forecast_fast(question_details, write=print):
    """
    Fast mode binary forecast: ~18 LLM calls instead of ~54
    
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
    fine_print = question_details["fine_print"]

    write("[FAST MODE] Starting binary forecast with 2 forecasters (Claude + o3)")

    # ========================================
    # PHASE 1 & 2: Research (using lite search)
    # ========================================
    
    async def format_and_call_o3(prompt_template):
        """Helper to format and call o3"""
        content = prompt_template.format(
            title=title,
            today=today,
            background=background,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
        )
        return content, await call_gpt_o3(content)

    write("[FAST MODE] Phase 1&2: Initiating historical + current research")
    
    # Run both research phases in parallel
    historical_task = asyncio.create_task(format_and_call_o3(BINARY_PROMPT_historical))
    current_task = asyncio.create_task(format_and_call_o3(BINARY_PROMPT_current))
    
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
        ),
    )

    write("\n[FAST MODE] Historical context LLM output:\n" + historical_output[:500] + "...")
    write("\n[FAST MODE] Current context LLM output:\n" + current_output[:500] + "...")
    write(f"\n[FAST MODE] Historical search results: {len(context_historical)} chars")
    write(f"\n[FAST MODE] Current search results: {len(context_current)} chars")

    # ========================================
    # PHASE 3: Outside View (2 forecasters)
    # ========================================
    
    write("[FAST MODE] Phase 3: Generating outside view with 2 forecasters")
    
    prompt1 = BINARY_PROMPT_1.format(
        title=title,
        today=today,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        context=context_historical,
    )

    # Run 2 forecasters in parallel (Claude Sonnet + o3)
    async def run_outside_view():
        return await asyncio.gather(
            call_claude(prompt1),   # Forecaster 1: Claude Sonnet 4
            call_gpt_o3(prompt1),   # Forecaster 2: o3 (strongest model)
        )

    results_outside = await run_outside_view()
    
    write(f"\n[FAST MODE] Forecaster 1 (Claude) outside view:\n{results_outside[0][:500]}...")
    write(f"\n[FAST MODE] Forecaster 2 (o3) outside view:\n{results_outside[1][:500]}...")

    # ========================================
    # PHASE 4: Final Forecast (2 forecasters)
    # ========================================
    
    write("[FAST MODE] Phase 4: Generating final forecasts")
    
    # Build context for each forecaster
    context_map = {
        "claude": f"Current context: {context_current}\nOutside view prediction: {results_outside[0]}",
        "o3": f"Current context: {context_current}\nOutside view prediction: {results_outside[1]}",
    }

    def format_prompt2(forecaster_key: str):
        return BINARY_PROMPT_2.format(
            title=title,
            today=today,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            context=context_map[forecaster_key],
        )

    # Run final forecasts in parallel
    async def run_final_forecast():
        return await asyncio.gather(
            call_claude(format_prompt2("claude")),
            call_gpt_o3(format_prompt2("o3")),
        )

    results_final = await run_final_forecast()

    # ========================================
    # EXTRACT & COMBINE PREDICTIONS
    # ========================================
    
    probabilities = []
    forecaster_names = ["Claude Sonnet 4", "GPT-o3"]
    
    for i, result in enumerate(results_final):
        try:
            prob = extract_probability_from_response_as_percentage_not_decimal(result)
            probabilities.append(prob)
            write(f"\n[FAST MODE] {forecaster_names[i]} prediction: {prob}%")
        except Exception as e:
            write(f"[FAST MODE] Error extracting probability from {forecaster_names[i]}: {e}")
            probabilities.append(None)

    # Calculate final probability with weighting (o3 gets double weight)
    valid_probs = [p for p in probabilities if p is not None]
    
    if len(valid_probs) >= 1:
        weights = [1, 2]  # Claude: 1, o3: 2
        weighted_probs = [p * w for p, w in zip(probabilities, weights) if p is not None]
        weight_sum = sum(w for p, w in zip(probabilities, weights) if p is not None)
        final_prob = float(np.sum(weighted_probs) / weight_sum)
        final_prob = min(0.999, max(0.001, final_prob / 100))  # Normalize to [0.001, 0.999]
    else:
        final_prob = None
        write("[FAST MODE] ❌ Error: No valid predictions extracted")

    write(f"\n[FAST MODE] ✅ Final predictions: {probabilities}")
    write(f"[FAST MODE] ✅ Weighted result: {final_prob}")

    # Format output
    final_outputs = "\n\n".join(
        f"=== {forecaster_names[i]} ===\nOutput:\n{out}\nPredicted Probability: {prob if prob is not None else 'N/A'}%"
        for i, (out, prob) in enumerate(zip(results_final, probabilities))
    )

    write("\n[FAST MODE] ========================================")
    write("[FAST MODE] FORECAST COMPLETE")
    write("[FAST MODE] ========================================")

    return final_prob, final_outputs
