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
    MULTIPLE_CHOICE_PROMPT_MONTE_CARLO,
)
from llm_calls import call_claude, call_gpt_o3, call_gpt_o4_mini
from search import process_search_queries

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

async def get_multiple_choice_forecast(question_details: dict, write=print) -> tuple[dict[str, float], str]:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details.get("fine_print", "")
    options = question_details["options"]
    num_options = len(options)

    async def format_and_call_gpt(prompt_template):
        content = prompt_template.format(
            title=title,
            today=today,
            background=background,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            options=options,
        )
        return content, await call_gpt_o3(content)

    historical_task = asyncio.create_task(format_and_call_gpt(MULTIPLE_CHOICE_PROMPT_historical))
    current_task = asyncio.create_task(format_and_call_gpt(MULTIPLE_CHOICE_PROMPT_current))
    (historical_prompt, historical_output), (current_prompt, current_output) = await asyncio.gather(historical_task, current_task)

    context_historical, context_current = await asyncio.gather(
        process_search_queries(historical_output, forecaster_id="-1", question_details=question_details),
        process_search_queries(current_output, forecaster_id="0", question_details=question_details)
    )

    write("\nHistorical context LLM output:\n" + historical_output)
    write("\nCurrent context LLM output:\n" + current_output)
    write("\nHistorical context search results:\n" + context_historical)
    write("\nCurrent context search results:\n" + context_current)

    prompt1 = MULTIPLE_CHOICE_PROMPT_1.format(
        title=title,
        today=today,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        context=context_historical,
        options=options
    )

    async def run_prompt1():
        return await asyncio.gather(
            call_claude(prompt1),      # forecaster 1
            call_claude(prompt1),      # forecaster 2
            call_gpt_o4_mini(prompt1), # forecaster 3
            call_gpt_o3(prompt1), # forecaster 4
            call_gpt_o3(prompt1),      # forecaster 5
        )

    results_prompt1 = await run_prompt1()

    for i, res in enumerate(results_prompt1):
        write(f"\nForecaster_{i+1} step 1 output:\n{res}")

    context_map = {
        "1": f"Current context: {context_current}\nOutside view prediction: {results_prompt1[0]}",
        "2": f"Current context: {context_current}\nOutside view prediction: {results_prompt1[2]}",
        "3": f"Current context: {context_current}\nOutside view prediction: {results_prompt1[1]}",
        "4": f"Current context: {context_current}\nInside view prediction: {results_prompt1[3]}",
        "5": f"Current context: {context_current}\nInside view prediction: {results_prompt1[4]}",
    }

    def format_prompt2(f_id):
        return MULTIPLE_CHOICE_PROMPT_2.format(
            title=title,
            today=today,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            context=context_map[f_id],
            options=options
        )

    async def run_prompt2():
        return await asyncio.gather(
            call_claude(format_prompt2("1")),
            call_claude(format_prompt2("2")),
            call_gpt_o4_mini(format_prompt2("3")),
            call_gpt_o3(format_prompt2("4")),
            call_gpt_o3(format_prompt2("5")),
        )

    results_prompt2 = await run_prompt2()

    all_outputs = results_prompt2
    all_probs = []
    final_outputs = []

    for i, out in enumerate(all_outputs):
        try:
            write(f"Forecaster {i+1} step 2 output: {out}")
            probs = extract_option_probabilities_from_response(out, num_options)
            probs = normalize_probabilities(probs)
            all_probs.append(probs)
        except Exception as e:
            write(f"Error parsing probabilities from Forecaster {i+1}: {e}")
            all_probs.append([1.0 / num_options] * num_options)
        final_outputs.append(f"=== Forecaster {i+1} ===\nOutput:\n{out}\n")

    probs_matrix = np.array(all_probs)
    weights = np.array([1, 1, 1, 2, 2])[:len(probs_matrix)]
    weighted_probs = np.average(probs_matrix, axis=0, weights=weights)
    probability_yes_per_category = {opt: float(p) for opt, p in zip(options, weighted_probs)}

    comment = (
        f"Average Probability Yes Per Category: `{probability_yes_per_category}`\n\n"
        + "\n\n".join(final_outputs)
    )


    write("\nFinal averaged probabilities per category:")
    write(json.dumps(probability_yes_per_category, indent=2))
    write("\nForecast comment:")
    write(comment)

    return probability_yes_per_category, comment
