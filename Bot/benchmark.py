import asyncio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import dotenv
import requests
import os
from forecaster import binary_forecast, multiple_choice_forecast
from main import get_post_details


dotenv.load_dotenv()

OUTPUT_DIR = "new_benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_CUSTOM_QUESTIONS = True  # Set this to True to use manually-defined non-Metaculus questions

class ForecastableQuestion:
    def __init__(self, details: dict, community_prediction = 50.0, url: str = ""):
        self.title = details["title"]
        self.resolution_criteria = details.get("resolution_criteria", "")
        self.description = details.get("description", "")
        self.fine_print = details.get("fine_print", "")
        self.type = details.get("type", "binary")
        self.options = details.get("options", [])
        self.my_forecasts = details.get("my_forecasts", {})
        self.details_dict = details
        self.community_prediction = community_prediction
        self.url = url

class SimpleQuestion:
    def __init__(self, q_id, title, url, post_id):
        self.id = q_id
        self.title = title
        self.url = url
        self.post_id = post_id

class CustomQuestion(ForecastableQuestion):
    def __init__(self, title, resolution_criteria, description="", fine_print="", community_prediction=50.0, url="", type="binary", options=None):
        details = {
            "title": title,
            "resolution_criteria": resolution_criteria,
            "description": description,
            "fine_print": fine_print,
            "type": type,
            "options": options or [],
        }
        super().__init__(details, community_prediction, url)

async def get_custom_questions():
    print("Using manually specified custom questions...")
    from Dataset import ds
    return [ds[0]]

def parse_year(published_at):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(published_at, fmt).year
        except ValueError:
            continue
    raise ValueError("Invalid date format")

def parse_date(date_str):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError("Invalid date format")

async def get_open_binary_questions(limit=400, crowd_forecasters_gte=0):
    url = "https://www.metaculus.com/api2/questions/"
    params = {
        "limit": limit,
        "offset": 0,
        "order_by": "-nr_forecasters",
        "status": "open",
        "type": "binary"
    }
    response = requests.get(url, params=params)
    if not response.ok:
        raise RuntimeError(response.text)

    questions = response.json()["results"]
    all_questions = []
    deadline = datetime(2026, 1, 2)

    for q in questions:
        if q.get("nr_forecasters", 0) < crowd_forecasters_gte:
            continue
        if parse_year(q.get("published_at", "")) < 2024:
            continue
        close_time = parse_date(q.get("scheduled_resolve_time", "9999-01-01T00:00:00Z"))
        if close_time > deadline:
            continue
        all_questions.append(SimpleQuestion(q_id=q["id"], title=q["title"], url=f"https://www.metaculus.com/questions/{q['id']}/", post_id=q["id"]))

    return all_questions

async def get_metaculus_community_prediction(question_id):
    url = f"https://www.metaculus.com/api/posts/{question_id}/"
    response = requests.get(url)
    if not response.ok:
        raise RuntimeError(response.text)
    data = response.json()
    try:
        cp_history = data["question"]["aggregations"]["recency_weighted"]["history"]
        latest_cp = cp_history[-1]
        cp = latest_cp["means"][0]
        return cp * 100
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Failed to extract community prediction")
    

async def forecast_question(q):
    print(f"[forecast_question] Starting forecast for question: {q.title if hasattr(q, 'title') else q}")
    try:
        if isinstance(q, SimpleQuestion):
            details_dict = get_post_details(q.post_id)["question"]
            community_prob = await get_metaculus_community_prediction(q.id)
            question = ForecastableQuestion(details=details_dict, community_prediction=community_prob, url=q.url)
        else:
            question = q

        print(f"[forecast_question] Forecasting type: {question.type}")

        out_path = os.path.join(OUTPUT_DIR, f"{''.join(c if c.isalnum() else '_' for c in question.title)[:100]}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            def write_to_file(x): f.write(x + "\n")

            if question.type == "binary":
                bot_forecast, comment = await binary_forecast(question.details_dict, write=write_to_file)
                bot_forecast = bot_forecast / 100
                print(f"[forecast_question] Binary forecast result: {bot_forecast}")
            elif question.type == "multiple_choice":
                bot_forecast, comment = await multiple_choice_forecast(question.details_dict, write=write_to_file)
                print(f"[forecast_question] MCQ forecast result: {bot_forecast}")
            else:
                raise ValueError(f"Unsupported question type: {question.type}")

        if isinstance(bot_forecast, dict):
            bot_probs = list(bot_forecast.values())
            comm_probs = question.community_prediction if isinstance(question.community_prediction, list) else [question.community_prediction] * len(bot_probs)
            print(f"[forecast_question] MCQ Formatted Output:\nBot: {bot_probs}\nCommunity: {comm_probs}")
        else:
            print(f"[forecast_question] Binary Formatted Output:\nBot: {bot_forecast:.2f} | Community: {question.community_prediction:.2f}")

        result = {
            "title": question.title,
            "bot": bot_forecast,
            "url": question.url,
            "type": question.type,
            "bot_output": comment
        }

        if question.type == "multiple_choice":
            result["options"] = question.options
            if isinstance(question.community_prediction, dict):
                result["community"] = list(question.community_prediction.values())
            else:
                result["community"] = question.community_prediction
            if isinstance(bot_forecast, dict):
                result["bot"] = list(bot_forecast.values())
            else:
                result["bot"] = bot_forecast
        else:
            result["community"] = question.community_prediction

        print(f"[forecast_question] Returning result for '{question.title}'")
        return result

    except Exception as e:
        print(f"[forecast_question] Error forecasting question '{q.title if hasattr(q, 'title') else q}': {e}")
        return None

    

def normalized_eval(
    bot, community, q_type, *,
    # ── peer‑population parameters ──────────────────────────────
    n_peers = 100,
    tau_good = 80,
    tau_bad  = 20,
    frac_bad = 0.20,
    shift_range_binary = 0.17,
    shift_range_mc     = 0.23,   # base cap; auto‑scaled by k
    # ── Monte‑Carlo control ────────────────────────────────────
    n_runs = 1000,                 # << NEW: how many peer-pop draws to average
    eps = 1e-4,
    seed = None,
):
    """
    Expected Metaculus PEER score, estimated by averaging `n_runs`
    independent simulations of a mixture (good+bad) peer population.
    """

    rng = np.random.default_rng(seed)

    # ---------- helpers to draw peers --------------------------------------
    def _draw_peers_binary(comm_p, size):
        shift = rng.uniform(-shift_range_binary, shift_range_binary, size=size)
        mu = np.clip(comm_p + shift, eps, 1 - eps)              # shape (size,)

        n_bad  = int(round(n_peers * frac_bad))
        n_good = n_peers - n_bad

        peers_good = rng.beta(mu[:,None] * tau_good,
                              (1-mu)[:,None] * tau_good,
                              size=(size, n_good))
        peers_bad  = rng.beta(mu[:,None] * tau_bad,
                              (1-mu)[:,None] * tau_bad,
                              size=(size, n_bad))

        peers = np.concatenate([peers_good, peers_bad], axis=1) # (size, n_peers)
        return np.clip(peers, eps, 1 - eps)

    def _draw_peers_mc(comm_arr, size):            # size == n_runs
        k = len(comm_arr)
        shift_cap = shift_range_mc / max(1, k / 2)

        # ----- 1. draw shifted means for every run -----------------------------
        shift = rng.uniform(-shift_cap, shift_cap, size=size)      # (size,)
        mu_vec = np.clip(comm_arr + shift[:, None], eps, 1 - eps)  # (size, k)
        mu_vec /= mu_vec.sum(axis=1, keepdims=True)

        # ----- 2. allocate arrays ----------------------------------------------
        peers = np.empty((size, n_peers, k), dtype=float)
        n_bad  = int(round(n_peers * frac_bad))
        n_good = n_peers - n_bad

        # ----- 3. draw good & bad peers run‑by‑run ------------------------------
        for i in range(size):
            good = rng.dirichlet(mu_vec[i] * tau_good, size=n_good)
            bad  = rng.dirichlet(mu_vec[i] * tau_bad,  size=n_bad)
            peers[i, :n_good]     = good
            peers[i, n_good:]     = bad

        return np.clip(peers, eps, 1 - eps)         # shape (n_runs, n_peers, k)

    # ---------- expected score, averaged over n_runs -----------------------
    if q_type == "binary":
        bot_p  = np.clip(float(bot), eps, 1 - eps)
        comm_p = np.clip(float(community), eps, 1 - eps)

        peer_p = _draw_peers_binary(comm_p, n_runs)             # (n_runs, n_peers)

        ln_gm_yes = np.log(peer_p).mean(axis=1)                 # (n_runs,)
        ln_gm_no  = np.log(1 - peer_p).mean(axis=1)

        ps_yes = 100 * (np.log(bot_p)     - ln_gm_yes)
        ps_no  = 100 * (np.log(1 - bot_p) - ln_gm_no)
        scores = comm_p * ps_yes + (1 - comm_p) * ps_no         # (n_runs,)

        return float(scores.mean())

    elif q_type == "multiple_choice":
        bot_arr  = np.asarray(bot,       dtype=float)
        comm_arr = np.asarray(community, dtype=float)

        if bot_arr.shape != comm_arr.shape:
            raise ValueError("bot and community distributions must match length")

        bot_arr  = np.clip(bot_arr,  eps, 1 - eps)
        comm_arr = np.clip(comm_arr, eps, 1 - eps)
        comm_arr /= comm_arr.sum()

        peer_mat = _draw_peers_mc(comm_arr, n_runs)             # (n_runs, n_peers, k)
        ln_gm    = np.log(peer_mat).mean(axis=1)                # (n_runs, k)

        peer_scores_each = 100 * (np.log(bot_arr) - ln_gm)      # (n_runs, k)
        scores = peer_scores_each @ comm_arr                    # (n_runs,)

        return float(scores.mean())

    else:
        raise ValueError("q_type must be 'binary' or 'multiple_choice'")

def plot_results(bot_preds, community_preds, question_titles, types, options_list=None):
    binary_indices = [i for i, t in enumerate(types) if t == "binary"]
    mcq_indices = [i for i, t in enumerate(types) if t == "multiple_choice"]

    # Plot binary predictions
    if binary_indices:
        x = np.arange(len(binary_indices))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, [bot_preds[i] for i in binary_indices], width, label='Bot Prediction')
        plt.bar(x + width/2, [community_preds[i] for i in binary_indices], width, label='Community Prediction')
        plt.xticks(x, [f"Q{i+1}" for i in binary_indices], rotation=45)
        plt.ylabel('Probability (%)')
        plt.title('Bot vs Community Predictions (Binary)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plot multiple-choice predictions
    for i in mcq_indices:
        bot = bot_preds[i] if isinstance(bot_preds[i], list) else list(bot_preds[i].values())
        community = community_preds[i]
        options = options_list[i] if options_list else [f"Option {j+1}" for j in range(len(bot))]
        x = np.arange(len(options))
        width = 0.35
        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, bot, width, label="Bot")
        plt.bar(x + width/2, community, width, label="Community")
        plt.xticks(x, options, rotation=45)
        plt.title(f"MCQ Prediction: Q{i+1}")
        plt.ylabel("Probability (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()

def summarize_performance(bot_preds, community_preds, df):
    normalized_errors = []

    for i, row in df.iterrows():
        try:
            err = normalized_eval(row['bot'], row['community'], row['type'])
            normalized_errors.append(err)
        except Exception as e:
            print(f"Error computing score for question '{row['title']}': {e}")
            normalized_errors.append(np.nan)

    df["normalized_error"] = normalized_errors
    df_clean = df.dropna(subset=["normalized_error"])

    print("\n=== Per-question forecast comparison ===")
    for _, row in df_clean.iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Type: {row['type']}")
        print(f"Bot prediction:      {row['bot']}")
        print(f"Community prediction:{row['community']}")
        print(f"Normalized error:    {row['normalized_error']:.4f}")

    print("\n=== Unified Normalized Error Summary ===")
    print(f"Questions evaluated: {len(df_clean)}")
    print(f"Mean normalized error: {df_clean['normalized_error'].mean():.4f}")
    print(f"Median normalized error: {df_clean['normalized_error'].median():.4f}")
    print(f"Min: {df_clean['normalized_error'].min():.4f} | Max: {df_clean['normalized_error'].max():.4f}")

    plt.figure(figsize=(8, 5))
    plt.hist(df_clean['normalized_error'], bins=10, edgecolor='black')
    plt.title("Normalized Error Distribution")
    plt.xlabel("Normalized Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

async def run_benchmark(n=45):
    all_questions = await (get_custom_questions() if USE_CUSTOM_QUESTIONS else get_open_binary_questions(limit=900, crowd_forecasters_gte=30))
    selected_questions = all_questions[:n]
    print(f"Selected {len(selected_questions)} questions")
    results = []
    for i in range(0, len(selected_questions), 2):
        batch = selected_questions[i:i+2]
        batch_results = await asyncio.gather(*(forecast_question(q) for q in batch))
        results.extend([r for r in batch_results if r is not None])


    df = pd.DataFrame(results)
    if df.empty:
        print("No valid forecasts to plot.")
        return

    plot_results(
        df['bot'].tolist(),
        df['community'].tolist(),
        df['title'].tolist(),
        df['type'].tolist(),
        df.get('options', pd.Series([[]]*len(df))).tolist()
    )

    summarize_performance(
        df['bot'].tolist(),
        df['community'].tolist(),
        df
    )

if __name__ == "__main__":
    asyncio.run(run_benchmark(n=45))
