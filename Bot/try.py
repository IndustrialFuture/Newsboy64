import asyncio
from search import agentic_search
import numpy as np
import os
from aiohttp import ClientSession, ClientTimeout, ClientError
from openai import OpenAI

from dotenv import load_dotenv


load_dotenv()
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def call_gpt(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model="gpt-4.1",
        tools=[{"type": "web_search_preview", "search_context_size": "high",}],
        input= prompt
    )
    return response.output_text

async def main():
    query = """

    Please retrieve weekly Billboard Artist 100 top 10 charts for the past 12 months and identify how many artists each week were new entrants (previous rank ≥11 or unranked), then summarize the weekly counts and overall trend. 

    """
    ans = await agentic_search(query)

    print(ans)


if __name__ == "__main__":
    asyncio.run(main())