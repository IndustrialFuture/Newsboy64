claude_context = """
Position yourself as a professional forecaster placing in the top 1% of forecasters who participated in the Good Judgement Project. Your approach closely mirrors the one outlined in the book Superforecasting: The Art and Science of Prediction.
To summarize the approach, you carefully analyze a question and think about simpler sub-questions (Fermi analysis). Using historical context, you generate an outview view prediction as a baseline. Then, based on the latest news pertaining to the question, you adjust your base rate prediction(s) to make an inside view prediction(s), which you submit.
For each question, you also consider (depending on question type)
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) Combination of sub-factors that result in a No outcome (for binary questions).
(d) Combination of sub-factors that result in in a Yes outcome (for binary questions).
(e) Combination of sub-factors that result in unexpected outcomes (for multiple choice questions).
(f) Combination of sub-factors that results in a low outcome (for numeric questions).
(g) Combination of sub-factors that results in a high outcome (for numeric questions).
(h) The expectation of experts and markets.
In each of your analyses, you write your rationale clearly and spare little detail so your colleagues can understand the nuances that governed your thoughtful forecast. 
"""


gpt_context = """ Position yourself as a professional forecaster placing in the top 1% of forecasters who participated in the Good Judgement Project. Your approach closely mirrors the one outlined in the book Superforecasting: The Art and Science of Prediction.
To summarize the approach, you carefully analyze a question and think about simpler sub-questions (Fermi analysis). Using historical context, you generate an outview view prediction as a baseline. Then, based on the latest news pertaining to the question, you adjust your base rate prediction(s) to make an inside view prediction(s), which you submit.
For each question, you also consider (depending on question type)
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) Combination of sub-factors that result in a No outcome (for binary questions).
(d) Combination of sub-factors that result in in a Yes outcome (for binary questions).
(e) Combination of sub-factors that result in unexpected outcomes (for multiple choice questions).
(f) Combination of sub-factors that results in a low outcome (for numeric questions).
(g) Combination of sub-factors that results in a high outcome (for numeric questions).
(h) The expectation of experts and markets.
In each of your analyses, you write your rationale clearly and spare little detail so your colleagues can understand the nuances that governed your thoughtful forecast.
"""


BINARY_PROMPT_historical =  """
You are currently doing research for historical information on the below forecasting question.

The forecasting question is:
{title}

Question background:
{background}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Today is {today}.

Your task is to analyze the forecasting question and write a series of search queries that will be used by your assistant to find relevant historical context. For each query, indicate whether you wish to utlize google, google news or perplexity to retrieve information.

For google/google news:
Your query for google and google news are processed by classical search engines, so please phrase the queries in a way optimal for keyword optimized search (i.e., the phrase you search is likely to appear on desired web pages). Avoid writing overly specific queries. Limit to six words.

For perplexity:
Your query will be processed by a reasoning model equipped with capable web crawlers and designed to generate lengthy, detailed responses. As such, you may use a longer query with detailed instructions. It is possible to ask multiple questions. 
Nonetheless, you are advised to keep your query to at most four sentences.

You should format your answer exactly as below, always formatting the source in brackets () **on the same line as and after** the query. Do not wrap your query in quotes. Be sure to include two queries for Google/Google News and one for Perplexity.

Analysis:
{{Your initial impression/analysis of the forecasting question followed by reasoning about the most relevant historical context needed to generate an outside view.}}

Search queries:
1. [Query details] (Google)
2. [Query details] (Google News)
3. [Query details] (Perplexity)

"""

BINARY_PROMPT_current =  """
You are currently doing research for current information/news articles on the below forecasting question.

The forecasting question is:
{title}

Question background:
{background}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Today is {today}.

For google/google news:
Your query for google and google news are processed by classical search engines, so please phrase the queries in a way optimal for keyword optimized search (i.e., the phrase you search is likely to appear on desired web pages). Avoid writing overly specific queries. Limit to six words.

For assistant:
Your query for your assistant will be processed more naturally, so feel free to to write a sentence-long query in natural language. If desired, you can indicate multiple kinds of news articles you're looking for in your query. Keep your query to a maximum of two sentences. Important note: Do not prefix your query with 'Assistant:' or something similar. Please add the word 'Assistant' enclosed in brackets at the end of the query, referring closely to the example below.

You should format your answer exactly as below, always formatting the source in brackets () (NOT curly brackets, NOT square brackets [], you need to use normal brackets ()) **on the same line as and after** the query, applicable for Google, Google News and Assistant. Do not wrap your query in quotes. Be sure to include two queries for Google/Google News and one for your assistant.

Use the following verification checklist:
1. One query for google?
2. One query for google news?
3. One query for assistant?
4. All queries followed by either (Google), (Google News) or (Assistant) in brackets () on the same line as the query?

Analysis:
{{Your initial impression/analysis of the forecasting question followed by reasoning about the most relevant current information/news articles needed to generate an inside view.}}

Search queries:
1. [Query details] (Google)
2. [Query details] (Google News)
3. [Query details] (Assistant)
"""


BINARY_PROMPT_1 = """
You are currently analyzing a forecasting question to generate an outside view prediction.

The forecasting question is:
{title}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Today is {today}.

Historical context:
{context}

The information has been sourced from the internet/language models (for perplexity), so it is advisable to exercise healthy skepticism at your discretion.

Your analysis should have the following components, refering the above historical context:
(a) Source analysis: Briefly summarize each information source (either web article or Perplexity report), evaluate source quality and data. 
**Opinions are commonplace in writing. For each source, you must be able to discern factual information from opinions. You are advised to strongly consider only opinions originating from identifiable experts or entities**.
(b) Reference class analysis: Identify a few possible reference classes and evaluate respective suitabilities to the forecasting question. If applicable, choose the most suitable one.
(c) Timeframe analysis: State the prediction timeframe (e.g., how many days/months from now?) and examine historical patterns over similar periods
(d) Justification: Integrate the above factors with other points you found relevant to write a justification for your outside view prediction.

You are free to include other components to deepen the analysis, at your discretion. 

Subsequently, calibrate your outside view prediction, considering:
(a) You aim to predict the true probability of an event occuring, not a hedged or overconfident projection of your beliefs.
(b) Is there a rough figure in the sources you can tether your prediction to?
(c) Small differences in probabilities can be significant: 90% is a 9:1 odds and 99% is a 99:1 odds.

Format you answer as below:

Analysis:
{{Insert your analysis here}}

Outside view calibration:
{{Insert your calibration of your outside view prediction here.}}

Outside View Prediction:
[Provide your outside view prediction here to a 1% significance. It is not necessary for the prediction to be a neat multiple of 5%.]
"""


BINARY_PROMPT_2 = """
You are currently analyzing a forecasting question to generate a final, inside view prediction.

The forecasting question is:
{title}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Today is {today}.

You have completed an outside view prediction and gathered a collection of current information/news articles relevant to making an inside view prediction.

Outside view analysis + current information/news articles:
{context}

The information has been sourced from the internet, so it is advisable to exercise healthy skepticism at your discretion.

Your analysis should have the following components, refering the above historical context and current information:
(a) Source analysis: Briefly summarize each information source (either web article or Asknews articles), evaluate source quality and date.
**Opinions are commonplace in writing. For each source, you must be able to discern factual information from opinions. You are advised to strongly consider only opinions originating from identifiable experts or entities**.
(b) Evidence analysis: Weight evidence/factors relevant to resolution criteria in the sources based on the below weighing system
(c) Timeframe analysis: State the prediction timeframe (e.g., how many days/months from now?) and describe how your prediction might change if this was halfed/doubled? 
(d) Justification: Gather the most compelling factors and justify how they shift your outside view base rate. For large shifts, justify how current factors are sufficiently impactful to render the reference class non-analogous to current circumstances.

You are free to include other components to deepen the analysis, at your discretion. 

Evidence weighing system for (b):
Strong evidence (can warrant relatively large prediction shifts)
- Multiple independent, reliable and identifiable (i.e., you can name the direct source) sources confirming same direction
- Direct causal mechanisms clearly established
- Historical patterns with strong predictive power
- Structural/institutional factors that are difficult to change

Moderate evidence (warrant moderate prediction shifts):
- Single reliable and identifiable source with clear methodology
- Indirect but logical causal links
- Similar historical patterns with some differences
- Current trends with demonstrated momentum

Weak evidence (small adjustments):
- Anecdotal evidence
- Speculative connections
- Limited historical parallels
- Short-term or volatile indicators

Subsequently, calibrate your outside view prediction, considering:
(a) You aim to predict the true probability of an event occuring, not a hedged or overconfident projection of your beliefs.
(b) Small differences in probabilities can be significant: 90% is a 9:1 odds and 99% is a 99:1 odds. How would a +-10% shift in probability resonate with your analysis?

You are suggested to use the below checklist to verify the quality of your forecast **while reasoning**. Adjust your forecast if you have made mistakes. You can very briefly add a shortened version to your answer (see the format below).

------------------------ FORECASTING CHECKLIST ------------------------
1. Paraphrase the title and resolution criteria in <30 words, including relevant time windows. 
  * Check that your analysis exactly aligns with how variables and resolution criterias are defined. Bait-and-switch errors, while commonplace, are costly.

2. State your outside view base rate distribution previously established
  * Check that your final prediction distribution genuinely is rooted to this base rate. If not, verify that current circumstances are vastly different from historic reference classes. Outside first, usually.

3. Consistency check (write a single line)
  * "{{your prediction}} out of 100 times, {{resolution criteria}} happens." Does this make sense and is it aligned with my justification?

4. Write down in <20 words each, the three-five most significant pieces of evidence influencing your final prediction.
  * Verify the factual validity of each of these pieces of evidence. You do not want your forecast to hinge on a few flimsy, possibly inconsistent, pieces of evidence. Use this to consider shifting confidence up or down.

5. Blind-spot statement
  * Name the one scenario most likely to make your forecast look silly in hindsight and decide how it might shift the relative probabilities assigned.

6. Status quo outcome
  * The world changes slowly most of the time. Consider the volatility of the current situation and timeframe to check whether a slight nudge toward the status quo outcome might be advantageous.

------------------------------------------------------------------------

Format your answer as below, it is very important to follow this format exactly, especially for the final probability, as a regex looking for 'Probability:' will be used to extract your answer.

Analysis:
{{Insert your analysis here, following the above components.}}

Probability calibration
{{Insert your calibration of your inside view prediction here.}}

Checklist:
{{Shortened, brief checklist verification here}}

Probability: ZZ%
"""


BINARY_PROMPT_MONTE_CARLO = """
You are analyzing a forecasting question to generate a final, inside-view prediction using a Monte Carlo simulation.

---
Forecasting Question:
{title}

Resolution Criteria:
{resolution_criteria}

Fine Print:
{fine_print}

Today’s Date: {today}

Current Context (outside view + relevant evidence):
{context}
---

Your task has 3 parts:

### Part 1: Written Analysis
Write a concise but structured analysis covering:
(a) **Source Quality**: Evaluate sources in terms of recency, credibility, and scope.  
(b) **Evidence Weighting**: Identify major factors influencing the outcome. Use the weight classification below.  
(c) **Timeframe Reasoning**: Define how long until resolution, and how prediction might shift if the timeline changes.  
(d) **Justification**: Combine the above to justify your belief shift vs. outside view.

#### Evidence Weighing System:
- **Strong**: Multiple independent sources; clear causal mechanisms; strong historical precedent  
- **Moderate**: One good source; indirect links; weak precedent  
- **Weak**: Anecdotes; speculative logic; volatile indicators  

---

### Part 2: Monte Carlo Simulation Strategy
Design a simulation using key variables that drive the resolution.  
Explain each variable:
- What it represents  
- Its distribution and justification  
- Any assumptions, dependencies, or edge cases  
Be careful not to add too many variables and constraints, as this might skew your final probability to 0. A good range is between 2 to 4 variables with well-thought out distributions.The final proability from your simulation should be useful.

---

### Part 3: Python Code
Generate a **syntactically correct, executable Python 3 script** using NumPy.  
You must:
- Simulate at least 100,000 iterations
- Define and clearly document your key assumptions
- Return the **probability** of the outcome as a float between 0 and 1
- Your code should ONLY contain comments using #, NO comments should be triple quotes.
- Your code should consist of a function definition and a single print statement outside, at the end to print the **final probability **between 1 and 100 with a % sign following the number**. Please do not use something like if __name__ == "__main__": before the print statement, it will not run.

Exact output format (you must follow this format exactly):
<python>
# your full code goes here
</python>
#Output must be **exactly** formatted as below:
Probability: XX.X%

The final output should only include:
1. Your written analysis
2. Your simulation strategy
3. Your full Python code block, in the EXACT format shown above.
4. NO FURTHER COMMENTS after the python code block, you are not to make a forecast, just write python code that will output a forecast in the above specified format.

---
Be rigorous. Think like a superforecaster. Aim for clarity, realism, and testability.
"""


MULTIPLE_CHOICE_PROMPT_historical = """

You are currently doing research for historical information on the below forecasting question.

The forecasting question is:
{title}

The options are:
{options}

Question background:
{background}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Today is {today}.

For google/google news:
Your query for google and google news are processed by classical search engines, so please phrase the queries in a way optimal for keyword optimized search (i.e., the phrase you search is likely to appear on desired web pages). Avoid writing overly specific queries. Limit to six words.

For perplexity:
Your query will be processed by a reasoning model equipped with capable web crawlers and designed to generate lengthy, detailed responses. As such, you may use a longer query with detailed instructions. It is possible to ask multiple questions. 
Nonetheless, you are advised to keep your query to at most four sentences.

You should format your answer exactly as below, always formatting the source in brackets () **on the same line as and after** the query. Do not wrap your query in quotes. Be sure to include two queries for Google/Google News and one for Perplexity.

Analysis:
{{Your initial impression/analysis of the forecasting question followed by reasoning about the most relevant historical context needed to generate an outside view.}}

Search queries:
1. [Query details] (Google)
2. [Query details] (Google News)
3. [Query details] (Perplexity)
"""

MULTIPLE_CHOICE_PROMPT_current = """
You are currently doing research for current information/news articles on the below forecasting question.

The forecasting question is:
{title}

Question background:
{background}

The options are:
{options}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Today is {today}.

Your task is to analyze the forecasting question and write a series of search queries that will be used by your assistant to find current information/news articles relevant to the question. For each query, indicate whether you wish to utlize google, google news or your assistant directly to retrieve information.
Your query for google and google news are processed by classical search engines, so please phrase the queries in a way optimal for keyword optimized search. Avoid writing overly specific queries. Limit to six words.

Your query for your assistant will be processed more naturally, so feel free to to write a sentence-long query in natural language. If desired, you can indicate multiple kinds of news articles you're looking for in your query. Keep your query to a maximum of two sentences.

You should format your answer exactly as below, always formatting the source in brackets () (NOT curly brackets, NOT square brackets [], you need to use normal brackets ()) **on the same line as and after** the query, applicable for Google, Google News and Assistant. Do not wrap your query in quotes. Be sure to include two queries for Google/Google News and one for your assistant.

Use the following verification checklist:
1. One query for google?
2. One query for google news?
3. One query for assistant?
4. All queries followed by either (Google), (Google News) or (Assistant) in brackets () on the same line as the query?

Analysis:
{{Your initial impression/analysis of the forecasting question followed by reasoning about the most relevant current information/news articles needed to generate an inside view.}}

Search queries:
1. [Query details] (Google)
2. [Query details] (Google News)
3. [Query details] (Assistant)
"""

MULTIPLE_CHOICE_PROMPT_1 = """

You are currently analyzing a forecasting question to generate an outside view prediction.

The forecasting question is:
{title}

The options are: {options}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Today is {today}.

Historical context:
{context}
The information has been sourced from the internet/language models (for perplexity), so it is advisable to exercise healthy skepticism at your discretion.

Your analysis should have the following components, refering the above historical context:
(a) Source analysis: Briefly summarize each information source (either web article or Perplexity report), evaluate source quality and date.
**Opinions are commonplace in writing. For each source, you must be able to discern factual information from opinions. You are advised to strongly consider only opinions originating from identifiable experts or entities**.
(b) Reference class analysis: Identify a few possible reference classes and evaluate respective suitabilities to the forecasting question. If applicable, choose the most suitable one.
(c) Timeframe analysis: State the prediction timeframe (e.g., how many days/months from now?) and examine historical patterns over similar periods
(d) Justification: Integrate the above factors with other points you found relevant to write a justification for your outside view prediction.

Subsequently, calibrate your outside view prediction, considering:
(a) You aim to predict the true probability of events occuring, not a hedged or overconfident projection of your beliefs.
(b) Are there previously established distributions concerning the options that you can tether your prediction to?
(c) Small differences in probabilities can be significant: 90% is a 9:1 odds and 99% is a 99:1 odds.
(d) Historically, what is the rate of upsets/unexpected outcomes in the domain of this forecasting question? How should this affect your probability distribution?

Format you answer as below:

Analysis:
{{Insert your analysis here, following the above components.}}

Outside view calibration:
{{Insert your calibration of your outside view prediction here.}}

Outside View Prediction:
Write your final probabilites (to a 1% significance, there is no need to have them be neat multiples of 5) for the N options in this order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""

MULTIPLE_CHOICE_PROMPT_2 = """
You are currently analyzing a forecasting question to generate a final, inside view prediction.

The forecasting question is:
{title}

The options are:
{options}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Today is {today}.

You have completed an outside view prediction and gathered a collection of current information/news articles relevant to making an inside view prediction.

Outside view analysis + current information/news articles:
{context}

The information has been sourced from the internet, so it is advisable to exercise healthy skepticism at your discretion.

Your analysis should have the following components, refering the above historical context and current information:
(a) Source analysis: Briefly summarize each information source (either web article or Asknews articles), evaluate source quality and date.
**Opinions are commonplace in writing. For each source, you must be able to discern factual information from opinions. You are advised to strongly consider only opinions originating from identifiable experts or entities**.
(b) Evidence analysis: Weight evidence/factors relevant to resolution criteria in the sources based on the below weighing system
(c) Timeframe analysis: State the prediction timeframe (e.g., how many days/months from now?) and describe how your prediction might change if this was halfed/doubled? 
(d) Justification: Gather the most compelling factors and justify how they shift your outside view base rates. For large shifts, justify how current factors are sufficiently impactful to render the reference class non-analogous to current circumstances.

Evidence weighing system for (b):
Strong evidence (can warrant relatively large prediction shifts)
- Multiple independent, reliable sources confirming same direction
- Direct causal mechanisms clearly established
- Historical patterns with strong predictive power
- Structural/institutional factors that are difficult to change

Moderate evidence (warrant moderate prediction shifts):
- Single reliable source with clear methodology
- Indirect but logical causal links
- Similar historical patterns with some differences
- Current trends with demonstrated momentum

Weak evidence (small adjustments):
- Anecdotal evidence
- Speculative connections
- Limited historical parallels
- Short-term or volatile indicators

Subsequently, calibrate your outside view prediction, considering:
(a) You aim to predict the true probability distribution of events occuring, not a hedged or overconfident projection of your beliefs.
(b) Small differences in probabilities can be significant: 90% is a 9:1 odds and 99% is a 99:1 odds. How would a +-10% shift in probability across options resonate with your analysis?
(c) Are there likely to be any blind spots in your analysis/factors that could sway the outcome (if the resolution is unexpected, what would you think the reason would be)? If yes, should you be less confident on the highest options? If no, should you be more confident on these options?

Return the final probabilities in the list, in the same order they appear in {options}. Format your answer as below, it is very important to follow this format exactly, especially for the final probability list, as a regex looking for 'Probabilities:' will be used to extract your answer. 


You are suggested to use the below checklist to verify the quality of your forecast **while reasoning**. Adjust your forecast if you have made mistakes. You can very briefly add a shortened version to your answer (see the format below).

------------------------ FORECASTING CHECKLIST ------------------------
1. Paraphrase the possible options and resolution criteria in <30 words, including relevant time windows. 
  * Check that your analysis exactly aligns with how options and resolution criterias are defined. Bait-and-switch errors, while commonplace, are costly.

2. State your outside view base rate distribution previously established
  * Check that your final prediction distribution genuinely is rooted to this base rate. If not, verify that current circumstances are vastly different from historic reference classes. Outside first, usually.

3. Consistency check (write a single line)
  * "Most likely category/categories= ____; Least likely category/categories = ____; does this make sense and aligned with my justification?"

4. Write down in <20 words each, the three-five most significant pieces of evidence influencing your final prediction.
  * Verify the factual validity of each of these pieces of evidence. You do not want your forecast to hinge on a few flimsy, possibly inconsistent, pieces of evidence. Use this to consider shifting confidence up or down.

5. Blind-spot statement
  * Name the one scenario most likely to make your forecast look silly in hindsight and decide how it might shift the relative probabilities assigned.

6. Technicalities
  * Please ensure that the probabilities are between **0 and 100, and that they sum to 100, and are not followed by a % sign**. 

------------------------------------------------------------------------

Analysis:
{{Insert your analysis here, following the above components. You can segment your analysis across multiple categories of options if you find it useful.}}

Probability calibration
{{Insert your calibration of your inside view prediction here.}}

Checklist:
{{Shortened, brief checklist verification here}}

Probabilities: [Probability_A, Probability_B, ..., Probability_N]
"""

MULTIPLE_CHOICE_PROMPT_MONTE_CARLO = """
You are analyzing a forecasting question to generate a final, inside-view prediction distribution using a Monte Carlo simulation.

---
Forecasting Question:
{title}

The options are:
{options}

Resolution Criteria:
{resolution_criteria}

Fine Print:
{fine_print}

Today’s Date: {today}

Current Context (outside view + relevant evidence):
{context}
---

Your task has 3 parts:

### Part 1: Written Analysis
Write a concise but structured analysis covering:
(a) **Source Quality**: Evaluate sources in terms of recency, credibility, and scope.  
(b) **Evidence Weighting**: Identify major factors influencing the outcome. Use the weight classification below.  
(c) **Timeframe Reasoning**: Define how long until resolution, and how prediction might shift if the timeline changes.  
(d) **Justification**: Combine the above to justify your belief shift vs. outside view.

#### Evidence Weighing System:
- **Strong**: Multiple independent sources; clear causal mechanisms; strong historical precedent  
- **Moderate**: One good source; indirect links; weak precedent  
- **Weak**: Anecdotes; speculative logic; volatile indicators  

---

### Part 2: Monte Carlo Simulation Strategy
Design a simulation using key variables that drive the resolution.  
Explain each variable:
- What it represents  
- Its distribution and justification  
- Any assumptions, dependencies, or edge cases  
Be careful not to add too many variables and constraints, as this might skew your final probability to 0. A good range is between 2 to 4 variables with well-thought out distributions.The final proability from your simulation should be useful.

---

### Part 3: Python Code
Generate a **syntactically correct, executable Python 3 script** using NumPy.  
You should:
- Simulate at least 100,000 iterations
- Define and clearly document your key assumptions
- Ensure your program prints the **probability** of each outcome in an ordered list, in the same order they appear in {options}. 
- Ensure your code should ONLY contain comments using #, NO comments should be triple quotes.
- Ensure your code should consist of a function definition and a single print statement outside, at the end to print the **final probability list, with each item a float between 1 and 100**. Please do not use something like if __name__ == "__main__": before the print statement, it will not run.

Exact output format (you must follow this format exactly):
<python>
# your full code goes here
</python>
#Output must be **exactly** formatted as below:
Probabilities: [Probability_A, Probability_B, ..., Probability_N]

The final output should only include:
1. Your written analysis
2. Your simulation strategy
3. Your full Python code block, in the EXACT format shown above.
4. NO FURTHER COMMENTS after the python code block, you are not to make a forecast, just write python code that will output a forecast in the above specified format.

---
Be rigorous. Think like a superforecaster. Aim for clarity, realism, and testability.
"""

context = """

You are a professional superforecaster.

"""



NUMERIC_PROMPT_historical = """
You are currently doing research for historical information on the below forecasting question.

---
Forecasting Question:
{title}

Question background:
{background}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Units for answer: {units}

Today is {today}.

{lower_bound_message}
{upper_bound_message}

Note that this is a numeric question, with expected answer format as a discrete CDF (not required for this answer).

For google/google news:
Your query for google and google news are processed by classical search engines, so please phrase the queries in a way optimal for keyword optimized search (i.e., the phrase you search is likely to appear on desired web pages). Avoid writing overly specific queries. Limit to six words.

For perplexity:
Your query will be processed by a reasoning model equipped with capable web crawlers and designed to generate lengthy, detailed responses. As such, you may use a longer query with detailed instructions. It is possible to ask multiple questions. 
Nonetheless, you are advised to keep your query to at most three sentences.

You should format your answer exactly as below, always formatting the source in brackets () **on the same line as and after** the query. Do not wrap your query in quotes or brackets. Be sure to include two queries for Google/Google News and one for Perplexity. 

Analysis:
{{Your initial impression/analysis of the forecasting question followed by reasoning about the most relevant historical context needed to generate an outside view.}}

Search queries:
1. [Query details] (Google)
2. [Query details] (Google News)
3. [Query details] (Perplexity)
"""



NUMERIC_PROMPT_current = """
You are currently doing research for current information/news articles on the below forecasting question.

---
The forecasting question is:
{title}

Question background:
{background}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Today is {today}.

{lower_bound_message}
{upper_bound_message}


For google/google news:
Your query for google and google news are processed by classical search engines, so please phrase the queries in a way optimal for keyword optimized search (i.e., the phrase you search is likely to appear on desired web pages). Avoid writing overly specific queries. Limit to six words.

For assistant:
Your query for your assistant will be processed more naturally, so feel free to to write a sentence-long query in natural language. If desired, you can indicate multiple kinds of news articles you're looking for in your query. Keep your query to a maximum of two sentences.

You should format your answer exactly as below, always formatting the source in brackets () (NOT curly brackets, NOT squared brackets [], you need to use normal brackets ()) **on the same line as and after** the query, applicable for Google, Google News and Assistant. Do not wrap your query in quotes. Be sure to include two queries for Google/Google News and one for your assistant.

Use the following verification checklist:
1. One query for google?
2. One query for google news?
3. One query for assistant?
4. All queries followed by either (Google), (Google News) or (Assistant) in brackets () on the same line as the query?

Analysis:
{{Your initial impression/analysis of the forecasting question followed by reasoning about the most relevant current information/news articles needed to generate an inside view.}}

Search queries:
1. [Query details] (Google)
2. [Query details] (Google News)
3. [Query details] (Assistant)
"""



NUMERIC_PROMPT_1 = """
You are currently analyzing a numeric forecasting question to generate a final, inside view prediction.

The forecasting question is:
{title}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

Additional fine-print:
{fine_print}

Units for answer: {units}

Today is {today}.

{lower_bound_message}
{upper_bound_message}

You should consider the following hint regarding the interval the answer is expected to lie in: 
{hint}

Outside view analysis + current information/news articles:
{context}

The information has been sourced from the internet/language models (for perplexity), so it is advisable to exercise healthy skepticism at your discretion.


Your analysis should have the following components, refering the above historical context:
(a) Source analysis: Briefly summarize each information source (either web article or Perplexity report), evaluate source quality and date.
**Opinions are commonplace in writing. For each source, you must be able to discern factual information from opinions. You are advised to strongly consider only opinions originating from identifiable experts or entities**.
(b) Reference class analysis: Identify a few possible reference classes and evaluate respective suitabilities to the forecasting question. If applicable, choose the most suitable one.
(c) Timeframe analysis: State the prediction timeframe (e.g., how many days/months from now?) and examine historical patterns over similar periods
(d) Justification: Integrate the above factors with other points you found relevant to write a justification for your outside view prediction.

You are free to include other components to deepen the analysis, at your discretion.

Subsequently, calibrate your outside view prediction, considering:
(a) You aim to predict a true probability distribution, not a hedged smooth distribution or an overconfident extremely narrow distribution. In your thinking, always consider ranges over singular values.
(b) Are there previously established distributions that you can tether your prediction to?
(c) Small changes in percentile location values can disproportionately reshape the slope and overall distribution of the extrapolated CDF, esepcially near the tails.
(d) Historically, what is the rate of upsets/unexpected outcomes in the domain of this forecasting question? How should this affect your CDF distribution?

It might be a good idea to set a wide 90/10 confidence intervals to account for unknown unknowns.

For your final outside view prediction, please keep in mind the following:
- Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
- Never use scientific notation.
- Always start with a smaller number (more negative if negative) and then increase from there

Format your answer as below:

Analysis:
{{Insert your analysis here, following the above components.}}

Outside view calibration:
{{Insert your calibration of your outside view prediction here.}}

Outside View Prediction:
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"""



NUMERIC_PROMPT_MONTE_CARLO = """
You are analyzing a numeric forecasting question to generate a final, inside-view prediction distribution using a Monte Carlo simulation.

---
Forecasting Question:
{title}

Resolution Criteria:
{resolution_criteria}

Fine Print:
{fine_print}

Units for answer: {units}

Today is {today}.

{lower_bound_message}
{upper_bound_message}

You should consider the following hint regarding the interval the answer is expected to lie in:
{hint}

Current Context (outside view + relevant evidence):
{context}

The information has been sourced from the internet, so it is advisable to exercise healthy skepticism at your discretion.

Your task has 3 parts:

### Part 1: Written Analysis
Write a concise but structured analysis covering:
(a) **Source Quality**: Evaluate sources in terms of recency, credibility, and scope.  
**Opinions are commonplace in writing. For each source, you must be able to discern factual information from opinions. You are advised to strongly consider only opinions originating from identifiable experts or entities**.
(b) **Evidence Weighting**: Identify major factors influencing the outcome. Use the weight classification below.  
(c) **Timeframe Reasoning**: Define how long until resolution, and how prediction might shift if the timeline changes.  
(d) **Justification**: Combine the above to justify your belief shift vs. outside view.

#### Evidence Weighing System:
- **Strong**: Multiple independent sources; clear causal mechanisms; strong historical precedent  
- **Moderate**: One good source; indirect links; weak precedent  
- **Weak**: Anecdotes; speculative logic; volatile indicators  

---

### Part 2: Monte Carlo Simulation Strategy
Design a simulation using key variables that drive the resolution.  
Explain each variable:
- What it represents  
- Its distribution and justification  
- Any assumptions, dependencies, or edge cases  
Be careful not to add too many variables and constraints, as this might skew your final probability to 0. A good range is between 2 to 4 variables with well-thought out distributions. Your final distribution should be realistic and useful, without being extremely skewed.

---

### Part 3: Python Code
Generate a **syntactically correct, executable Python 3 script** using NumPy.  
You should:
- Sample from at least 10^6 iterations
- Define and clearly document your key assumptions using comments that start with #. Do not use triple-quoted strings for comments.
- Utilize NumPy's vectorized operations instead of Python for-loops to enhance performance and efficiency.
- Structure your code to include:
  - A function definition that performs the Monte Carlo simulation and calculates the desired percentiles.
  - A single print statement outside the function (**WITHOUT** using if __name__ == "__main__":) that calls the function and prints the final percentile list.
- Follow these instructions regarding the numerical percentiles:
  - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
  - Never use scientific notation.

Exact output format (you must follow this format exactly):
<python>
# your full code goes here
</python>
# Output must be **exactly** printed as below:
Percentile 1: XX
Percentile 5: XX
Percentile 10: XX
Percentile 15: XX
Percentile 20: XX
Percentile 25: XX
Percentile 30: XX
Percentile 35: XX
Percentile 40: XX
Percentile 45: XX
Percentile 50: XX
Percentile 55: XX
Percentile 60: XX
Percentile 65: XX
Percentile 70: XX
Percentile 75: XX
Percentile 80: XX
Percentile 85: XX
Percentile 90: XX
Percentile 95: XX
Percentile 99: XX

The final output should only include:
1. Your written analysis
2. Your simulation strategy
3. Your full Python code block, in the EXACT format shown above.
4. NO FURTHER COMMENTS after the python code block, you are not to make your own forecast.
"""


NUMERIC_PROMPT_2 = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

{resolution_criteria}

{fine_print}

Units for answer: {units}

Today is {today}.

{lower_bound_message}
{upper_bound_message}

You should consider the following hint regarding the interval the answer is expected to lie in:
{hint}

Current Context (outside view + relevant evidence):
{context}

The information has been sourced from the internet, so it is advisa

Your analysis should have the following components, refering the above historical context and current information:
(a) Source analysis: Briefly summarize each information source (either web article or Asknews articles), evaluate source quality and date.
**Opinions are commonplace in writing. For each source, you must be able to discern factual information from opinions. You are advised to strongly consider only opinions originating from identifiable experts or entities**.
(b) Evidence analysis: Weight evidence/factors relevant to resolution criteria in the sources based on the below weighing system
(c) Timeframe analysis: State the prediction timeframe (e.g., how many days/months from now?) and describe how your prediction might change if this was halfed/doubled? 
(d) Justification: Gather the most compelling factors and justify how they shift your outside view base rates. For large distribution shifts, justify how current factors are sufficiently impactful to render the reference class non-analogous to current circumstances.

You are free to include other components to deepen the analysis, at your discretion. 

Evidence weighing system for (b):
Strong evidence (can warrant relatively large prediction shifts)
- Multiple independent, reliable sources confirming same direction
- Direct causal mechanisms clearly established
- Historical patterns with strong predictive power
- Structural/institutional factors that are difficult to change

Moderate evidence (warrant moderate prediction shifts):
- Single reliable source with clear methodology
- Indirect but logical causal links
- Similar historical patterns with some differences
- Current trends with demonstrated momentum

Weak evidence (small adjustments):
- Anecdotal evidence
- Speculative connections
- Limited historical parallels
- Short-term or volatile indicators

Subsequently, calibrate your outside view prediction, considering:
(a) The outcome if the current trend continued.
(b) You aim to predict a true probability distribution, not a hedged smooth distribution or an overconfident extremely narrow distribution. In your thinking, always consider ranges over singular values.
(c) Small changes in percentile location values can disproportionately reshape the slope and overall distribution of the extrapolated CDF, esepcially near the tails.

It might be a good idea to set a wide 90/10 confidence intervals to account for unknown unknowns.

Regarding the final prediction, please keep in mind the following:
- Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
- Never use scientific notation.
- Always start with a smaller number (more negative if negative) and then increase from there

You are suggested to use the below checklist to verify the quality of your forecast **while reasoning**. Adjust your forecast if you have made mistakes. You can very briefly add a shortened version to your answer (see the format below).

------------------------ FORECASTING CHECKLIST ------------------------
1. Paraphrase the target variable/resolution criteria in <20 words, including units and the time window
  * Check that your analysis exactly aligns with this target variable/resolution criteria. Bait-and-switch errors, while commonplace, are costly.

2. State your outside view base rate previously established
  * Check that your final prediction genuinely is rooted to this base rate. If not, verify that current circumstances are vastly different from historic reference classes. Outside first, usually.

3. Consistency check (write a single line)
  * "Implied median = ____; mean = ____; does this make sense and aligned with my justification?"

4. Write down in <20 words each, the three-five most significant pieces of evidence influencing your final prediction.
  * Verify the factual validity of each of these pieces of evidence. You do not want your forecast to hinge on a few flimsy, possibly inconsistent, pieces of evidence. Use this to consider shifting confidence up or down.

5. Blind-spot statement
  * Name the one scenario most likely to make your forecast look silly in hindsight and decide whether it would push the outcome up or down.

------------------------------------------------------------------------

**Essential formatting requirements**
(a) For large numbers, please DO NOT output commas between numbers like 1,000,000. Instead, just write 1000000. If not, this will cause a parsing error. 
(b) You MUST prefix the final percentiles with Distribution: as a regex will be programmed to read text below 'Distribution:'. 

Format your answer as below. 

Analysis:
{{Insert your analysis here, following the above components. You can segment your analysis across multiple final answer ranges if you find it useful.}}

Probability calibration
{{Insert your calibration of your inside view prediction here.}}

Checklist:
{{Shortened, brief checklist verification here}}

Distribution:
Percentile 1: XX
Percentile 5: XX
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
Percentile 95: XX
Percentile 99: XX
"""


# Prompts for the agentic search
INITIAL_SEARCH_PROMPT = """You are an expert research assistant tasked with conducting thorough internet research to answer a specific query. Your goal is to understand what information is needed and generate targeted search queries to gather current, accurate information.

Query to research: {query}

Your task:
1. Analyze the query to understand what information is needed
2. Identify the key aspects that need to be researched
3. Generate specific search queries to gather comprehensive information

Important guidelines:
- Do NOT attempt to answer the query yet - focus on understanding what needs to be researched
- Each search query should target a different aspect of the information needed
- Make queries specific and likely to return accurate, authoritative information
- Choose your source as Google or Google News based on the type of information needed.
- Please phrase the queries in a way optimal for keyword optimized search (i.e., the phrase you search is likely to appear on desired web pages). Avoid writing overly specific queries. Limit to six words.
- You can list up to 5 search queries

Important formatting instructions: You should format your answer EXACTLY as below, always formatting the source in brackets () **on the same line as and after** the query. DO NOT use any quotes in your queries. 

Analysis: 
[Brief analysis of what the query is asking for and what types of information would constitute a thorough answer. This is NOT a full response - just an understanding of what needs to be researched.]

Search queries:
1. [Query details] (Google)
2. [Query details] (Google News)
3. [Query details] (Google)
(Additional queries in the same format, if needed, up to five queries)


"""

CONTINUATION_SEARCH_PROMPT = """You are continuing your research to answer a specific query. Based on the search results, provide a comprehensive analysis and identify if additional information is needed.

Original query: {query}

{previous_section}

New search results:
{search_results}

Your task:
1. Write a complete, comprehensive analysis based on the search results
2. Ensure all aspects of the original query are thoroughly addressed
3. Identify any remaining information gaps
4. Generate new search queries if needed (different from previous queries)

Important guidelines:
- This must be a COMPLETE analysis that could serve as the final answer
- Base your analysis primarily on the search results provided
- Cite sources from the search results to establish credibility
- Be thorough and answer the ENTIRE query precisely
- Include all nuanced details available from the search results
- Be objective: present facts without personal opinions
- Only generate new search queries if they would materially improve your answer
- Choose your source as Google or Google News based on the type of information needed.
- Please phrase the queries in a way optimal for keyword optimized search (i.e., the phrase you search is likely to appear on desired web pages). Avoid writing overly specific queries. Limit to six words.
- List a maximum of five search queries, being conservative and using only the maximum number of queries when really necessary. 
- If your analysis is sufficiently complete, omit the "Search queries:" section entirely to signal completion. Absence of regex match for 'Search queries:' will signal that this is your final research step.
- If you encounter difficulties retrieving the exact information being requested (paywall or information unavailable), please consider adding queries to look for alternative sources/proxies for the same data. If utilizing these sources, you must add the alternative source used as a disclaimer in your analysis. Alternatively, consider how you might obtain a small subset of the data requested (perhaps by viewing snapshots at specific times). Some information is better than no information in your conclusion.
- You can consider searching for specific sites using the site prefix in your query. 
- Remember: Write a complete draft, not notes or an outline


Important formatting instructions: You should format your answer EXACTLY as below, always formatting the source in brackets () **on the same line as and after** the query. DO NOT use any quotes in your queries. It is essential to follow these formatting instructions as regex will be used to parse your response.

Analysis: 
[Write your COMPLETE analysis here. This should be a full, comprehensive response that could serve as the final answer. Base your analysis on the search results provided. Cite sources using phrases like "according to [source]" or "as reported by [source]". Be objective and distinguish between facts and opinions. Include all relevant details, statistics, context, and nuances from the search results. Aim for a report length of about 1000 words, keeping the report as structured as possible.]

Search queries:
1. [Query details] (Google)
2. [Query details] (Google News)
3. [Query details] (Google)
(Additional queries in the same format, if needed, up to five queries)

Use the following verification checklist at the end of your response
1. Correct formatting for analysis (Analysis: [your analysis])?
2. Correct formatting for search queries (Search queries: [numbered newline separated search query list following the exact format shown above])?
3. Analysis as exhaustive as possible, minimally with some useful information (i.e., current values) only if web search results don't turn out too useful?
4. All queries followed by either (Google), (Google News) in brackets () on the same line as the query?

"""