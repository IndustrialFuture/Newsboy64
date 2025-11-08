import asyncio
import datetime
import json
import os
import re
import dotenv
dotenv.load_dotenv()
from forecaster import binary_forecast, numeric_forecast, multiple_choice_forecast

import numpy as np
import requests
from asknews_sdk import AskNewsSDK
from search import call_gpt


OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Q2_tournament_forecasts"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

######################### CONSTANTS #########################
# Constants
SUBMIT_PREDICTION = True  # set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False # set to True to forecast example questions rather than the tournament questions
NUM_RUNS_PER_QUESTION = 5  # The median forecast is taken between NUM_RUNS_PER_QUESTION runs
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
RUN = True

# Environment variables
# You only need *either* Exa or Perplexity or AskNews keys for online research
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # You'll also need the OpenAI API Key if you want to use the Exa Smart Searcher

# The tournament IDs below can be used for testing your bot.
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
Q4_2024_QUARTERLY_CUP_ID = 3672 
Q2_2025_AI_BENCHMARKING_ID = 32721
Q1_2025_QUARTERLY_CUP_ID = 32630 #open
AXC_2025_TOURNAMENT_ID = 32564 #open
GIVEWELL_ID = 3600 #open
RESPIRATORY_OUTLOOK_ID = 3411 #open

TOURNAMENT_ID = Q2_2025_AI_BENCHMARKING_ID

# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
   # (35828, 36422),
    (7596, 7596), 
    #(35826, 36420),  
    #(22427, 22427),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
]

# Also, we realize the below code could probably be cleaned up a bit in a few places
# Though we are assuming most people will dissect it enough to make this not matter much

######################### HELPER FUNCTIONS #########################

# @title Helper functions
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"


def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise RuntimeError(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,  # type: ignore
    )
    print(f"Prediction Post status code: {response.status_code}")
    if not response.ok:
        raise RuntimeError(response.text)


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int = TOURNAMENT_ID, offset: int = 0, count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data

def get_question_details(question_id: int) -> dict:
    url = f"https://www.metaculus.com/api/questions/{question_id}/"
    response = requests.get(url, **AUTH_HEADERS)
    if not response.ok:
        print(f"Question extraction with url {url} failed")
        raise Exception(response.text)
    data = json.loads(response.content)
    print("Question data retrieved successfully")
    print(data)
    return data

def get_open_question_ids_from_tournament() -> list[tuple[int, int]]:
    posts = list_posts_from_tournament()

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            # single question post
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(
        url,
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        print(f"Post extraction with url {url} failed")
        raise Exception(response.text)
    details = json.loads(response.content)
    return details


################### FORECASTING ###################
def forecast_is_already_made(question_details: dict) -> bool:
    try:
        forecast_values = question_details["my_forecasts"]["latest"]["forecast_values"]
        return forecast_values is not None
    except Exception:
        return False

async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    try:
        post_details = get_post_details(post_id)
        question_details = post_details["question"]
    except KeyError:
        print(f"Fallback to question details API for question_id={question_id}")
        question_details = get_question_details(question_id)

    title = question_details["title"]
    question_type = question_details["type"]

    if forecast_is_already_made(question_details) and skip_previously_forecasted_questions:
        summary_of_forecast = f"-----------------------------------------------\nQuestion: {title}\n"
        summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"
        summary_of_forecast += "Skipped: Forecast already made\n"
        return summary_of_forecast

    summary_of_forecast = f"-----------------------------------------------\nQuestion: {title}\n"
    summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

    filename = f"{''.join(c if c.isalnum() else '_' for c in title)[:100]}.txt"
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Use safe context manager for writing
    with open(output_path, "w", encoding="utf-8") as f:
        def write_to_file(line: str):
            print(f"[WRITE] {line}")
            f.write(line + "\n")

        if question_type == "multiple_choice":
            options = question_details["options"]
            summary_of_forecast += f"Options: {options}\n"

        # Forecasting per type
        if question_type == "binary":
            forecast, comment = await binary_forecast(question_details, write=write_to_file)
            if forecast > 1:
                forecast /= 100
        elif question_type == "numeric":
            forecast, comment = await numeric_forecast(question_details, write=write_to_file)
        elif question_type == "multiple_choice":
            forecast, comment = await multiple_choice_forecast(question_details, write=write_to_file)
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        print(f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n")
        print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
        print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

        summary_of_forecast += f"Forecast: {forecast}\n"

        comment_str = json.dumps(comment, indent=2) if not isinstance(comment, str) else comment
        summary_of_forecast += f"Comment:\n```\n{comment_str}...\n```\n\n"

        summary_prompt = f"""Below is a detailed explanation for a forecast posted on Metaculus, comprising reasoning from a team of five forecasters.
Please summarize it into a concise 5-7 sentence paragraph suitable for a forecast comment. 
Ensure you preserve the key reasoning, especially if relevant sources, probabilities, or 
contextual comparisons are mentioned. Reference key agreements and possible disagreements between forecasters. You may conclude by briefly referencing the five final forecast values. 

Please begin the summary straightaway by briefly describing the question, DO NOT prefix your answer with something like 'Here is the summarized reasoning:' or 'Forecaster summary:'.

Forecast Explanation:
{comment_str}
"""

        try:
            short_comment = await call_gpt(summary_prompt)
        except Exception as e:
            print(f"Summarization failed, using original comment. Error: {e}")
            short_comment = comment_str

        print(f"Forecast was retrieved successfully with value {forecast}")
        print(f"Forecast is of type {type(forecast)}")

        if submit_prediction:
            forecast_payload = create_forecast_payload(forecast, question_type)
            post_question_prediction(question_id, forecast_payload)
            post_question_comment(post_id, short_comment)
            summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

        write_to_file(summary_of_forecast)

    return summary_of_forecast



async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    forecast_tasks = [
        forecast_individual_question(
            question_id,
            post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
        )
        for question_id, post_id in open_question_id_post_id
    ]
    forecast_summaries = await asyncio.gather(*forecast_tasks, return_exceptions=True)
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\nError: {forecast_summary.__class__.__name__} {forecast_summary}\nURL: https://www.metaculus.com/questions/{post_id}/\n"
            )
            errors.append(forecast_summary)
        else:
            print(forecast_summary)

    if errors:
        print("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        print(error_message)
        raise RuntimeError(error_message)




######################## FINAL RUN #########################
if __name__ == "__main__":
    if not RUN:
        exit()
    if USE_EXAMPLE_QUESTIONS:
        open_question_id_post_id = EXAMPLE_QUESTIONS
    else:
        open_question_id_post_id = get_open_question_ids_from_tournament()

    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )