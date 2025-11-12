import asyncio
import time
import numpy as np
from numeric import get_numeric_forecast
from numeric_fast import get_numeric_forecast_fast  # NEW IMPORT
from binary import get_binary_forecast
from binary_fast import get_binary_forecast_fast
import os
import requests
from aiohttp import ClientSession, ClientTimeout, ClientError
import json
import sys
import re
import io
from multiple_choice import get_multiple_choice_forecast
from multiple_choice_fast import get_multiple_choice_forecast_fast  # NEW IMPORT
from asknews_sdk import AskNewsSDK
from prompts import context
from dotenv import load_dotenv
import aiohttp
from prompts import claude_context, gpt_context

"""
This file contains the main forecasting logic, question-type specific functions are abstracted.
Supports both FULL and FAST modes via environment variable.
"""

load_dotenv()
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for mode setting
PANSHUL_MODE = os.getenv("PANSHUL_MODE", "fast").lower()  # Default to "fast"

def write(x):
    print(x)

async def numeric_forecast(question_details, write=print):
    """
    Route to fast or full mode based on PANSHUL_MODE environment variable.
    """
    if PANSHUL_MODE == "fast":
        write("[ROUTING] Using FAST mode for numeric forecast")
        return await get_numeric_forecast_fast(question_details, write)
    else:
        write("[ROUTING] Using FULL mode for numeric forecast")
        return await get_numeric_forecast(question_details, write)

async def binary_forecast(question_details, write=print):
    """
    Route to fast or full mode based on PANSHUL_MODE environment variable.
    """
    if PANSHUL_MODE == "fast":
        write("[ROUTING] Using FAST mode for binary forecast")
        return await get_binary_forecast_fast(question_details, write)
    else:
        write("[ROUTING] Using FULL mode for binary forecast")
        return await get_binary_forecast(question_details, write)

async def multiple_choice_forecast(question_details, write=print):
    """
    Route to fast or full mode based on PANSHUL_MODE environment variable.
    """
    if PANSHUL_MODE == "fast":
        write("[ROUTING] Using FAST mode for multiple choice forecast")
        return await get_multiple_choice_forecast_fast(question_details, write)
    else:
        write("[ROUTING] Using FULL mode for multiple choice forecast")
        return await get_multiple_choice_forecast(question_details, write)
