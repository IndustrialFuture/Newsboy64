# Revised version of get_numeric_forecast.py with extensive logging, robust error handling, and fallback logic

import datetime
import numpy as np
import asyncio
import re
import unicodedata
import itertools
from typing import Union
from prompts import (
    NUMERIC_PROMPT_historical,
    NUMERIC_PROMPT_current,
    NUMERIC_PROMPT_1,
    NUMERIC_PROMPT_2,
)
from llm_calls import call_claude, call_gpt_o4_mini, call_gpt_o3
from search import process_search_queries

VALID_KEYS = {1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99}

NUM_PATTERN = re.compile(
    r"^(?:percentile\s*)?(\d{1,3})\s*[:\-]\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$",
    re.IGNORECASE
)

BULLET_CHARS = "•▪●‣–*-"

def _safe_cdf_bounds(cdf, open_lower, open_upper, step):
        """
       • Metaculus requires, for *open* bounds:
            - cdf[0]  ≥ 0.001
            - cdf[-1] ≤ 0.999
        • Also no single step may exceed 0.59.
        """
        # Pin tails to legal open-bound limits
        if open_lower:
            cdf[0] = max(cdf[0], 0.001)
        if open_upper:
            cdf[-1] = min(cdf[-1], 0.999)

        # Enforce the 0.59 maximum step rule
        big_jumps = np.where(np.diff(cdf) > 0.59)[0]
        for idx in big_jumps:
            excess = cdf[idx+1] - cdf[idx] - 0.59
            # Spread the excess evenly over the remaining points
            span = len(cdf) - idx - 1
            cdf[idx+1:] -= excess * np.linspace(1, 0, span)
            # Re-monotonise
            cdf[idx+1:] = np.maximum.accumulate(cdf[idx+1:])

        return cdf

DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")

def clean(s: str) -> str:
    # 1) Normalize compatibility forms
    s = unicodedata.normalize("NFKC", s)
    # 2) Replace every dash‐like char with ASCII hyphen
    s = DASH_RE.sub("-", s)
    # 3) Strip bullets from the start
    s = s.strip().lstrip(BULLET_CHARS)
    # 4) Remove thousands-sep commas & NBSPs
    s = s.replace(",", "").replace("\u00A0", "")
    return s.lower()


def enforce_strict_increasing(pct_dict: dict) -> dict:
    """Ensure strictly increasing values by adding tiny jitter if necessary."""
    sorted_items = sorted(pct_dict.items())
    last_val = -float('inf')
    new_pct_dict = {}
    
    for p, v in sorted_items:
        if v <= last_val:
            v = last_val + 1e-8  # Add a tiny epsilon
        new_pct_dict[p] = v
        last_val = v
        
    return new_pct_dict

def extract_percentiles_from_response(text: Union[str, list], verbose: bool = True) -> dict:
    lines = text if isinstance(text, list) else text.splitlines()
    percentiles = {}
    collecting = False
    for idx, raw in enumerate(lines, 1):
        line = clean(str(raw))
        if not collecting and "distribution:" in line:
            collecting = True
            if verbose:
                print(f"🚩 Found 'Distribution:' anchor at line {idx}")
            continue
        if not collecting:
            continue
        match = NUM_PATTERN.match(line)
        if not match:
            if verbose:
                print(f"⛔ No match on line {idx}: {line}")
            continue
        key, val_text = match.groups()
        try:
            p = int(key)
            val = float(val_text)
            if p in VALID_KEYS:
                percentiles[p] = val
                if verbose:
                    print(f"✅ Matched Percentile {p}: {val}")
        except Exception as e:
            print(f"❌ Failed parsing line {idx}: {line} → {e}")
    if not percentiles:
        raise ValueError("❌ No valid percentiles extracted.")
    return percentiles

def generate_continuous_cdf(percentile_values, open_upper_bound, open_lower_bound, upper_bound, 
                        lower_bound, zero_point=None, *, min_step=5.0e-5, num_points=201):
    """
    Generate a robust continuous CDF with strict enforcement of minimum step size.
    
    Args:
        percentile_values (dict): Dictionary mapping percentiles (1-99) to values
        open_upper_bound (bool): Whether the upper bound is open
        open_lower_bound (bool): Whether the lower bound is open
        upper_bound (float): Maximum possible value
        lower_bound (float): Minimum possible value
        zero_point (float, optional): Reference point for non-linear scaling
        min_step (float): Minimum step size between adjacent CDF points (default: 5.0e-5)
        num_points (int): Number of points in the output CDF (default: 201)
    
    Returns:
        list: A list of CDF values with strictly enforced monotonicity and step size
    """
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    
    # Validate inputs
    if not percentile_values:
        raise ValueError("Empty percentile values dictionary")
    
    if upper_bound <= lower_bound:
        raise ValueError(f"Upper bound ({upper_bound}) must be greater than lower bound ({lower_bound})")
    
    if zero_point is not None:
        if abs(zero_point - lower_bound) < 1e-6 or abs(zero_point - upper_bound) < 1e-6:
            raise ValueError(f"zero_point ({zero_point}) too close to bounds [{lower_bound}, {upper_bound}]")
    
    # Clean and validate percentile values
    pv = {}
    for k, v in percentile_values.items():
        try:
            k_float = float(k)
            v_float = float(v)
            
            if not (0 < k_float < 100):
                continue  # Skip invalid percentiles
                
            if not np.isfinite(v_float):
                continue  # Skip non-finite values
                
            pv[k_float] = v_float
        except (ValueError, TypeError):
            continue  # Skip non-numeric entries
    
    if len(pv) < 2:
        raise ValueError(f"Need at least 2 valid percentile points (got {len(pv)})")
    
    # Handle duplicate values by adding small offsets
    vals_seen = {}
    for k in sorted(pv):
        v = pv[k]
        if v in vals_seen:
            # Add progressively larger offsets for duplicate values
            v += (len(vals_seen[v]) + 1) * 1e-9
        vals_seen.setdefault(v, []).append(k)
        pv[k] = v
    
    # Create arrays of percentiles and values
    percentiles, values = zip(*sorted(pv.items()))
    percentiles = np.array(percentiles) / 100.0  # Convert to [0,1] range
    values = np.array(values)
    
    # Check if values are strictly increasing after de-duplication
    if np.any(np.diff(values) <= 0):
        raise ValueError("Percentile values must be strictly increasing after de-duplication")
    
    # Add boundary points if needed
    if not open_lower_bound and lower_bound < values[0] - 1e-9:
        percentiles = np.insert(percentiles, 0, 0.0)
        values = np.insert(values, 0, lower_bound)
    
    if not open_upper_bound and upper_bound > values[-1] + 1e-9:
        percentiles = np.append(percentiles, 1.0)
        values = np.append(values, upper_bound)
    
    # Determine if log scaling is appropriate (all values positive)
    use_log = np.all(values > 0)
    x_vals = np.log(values) if use_log else values
    
    # Create interpolator with fallback
    try:
        spline = PchipInterpolator(x_vals, percentiles, extrapolate=True)
    except Exception as e:
        # Fallback to linear interpolation
        print(f"PchipInterpolator failed ({str(e)}), falling back to linear interpolation")
        spline = lambda x: np.interp(x, x_vals, percentiles)
    
    # Generate evaluation grid based on zero_point
    def create_grid(num_points):
        t = np.linspace(0, 1, num_points)
        
        if zero_point is None:
            # Linear grid
            return lower_bound + (upper_bound - lower_bound) * t
        else:
            # Non-linear grid based on zero_point
            ratio = (upper_bound - zero_point) / (lower_bound - zero_point)
            # Handle potential numerical issues
            if abs(ratio - 1.0) < 1e-10:
                return lower_bound + (upper_bound - lower_bound) * t
            else:
                return np.array([
                    lower_bound + (upper_bound - lower_bound) * 
                    ((ratio**tt - 1) / (ratio - 1))
                    for tt in t
                ])
    
    # Generate the grid and evaluate
    cdf_x = create_grid(num_points)
    
    # Handle log transformation for evaluation
    eval_x = np.log(cdf_x) if use_log else cdf_x
    
    # Clamp values to avoid extrapolation issues
    eval_x_clamped = np.clip(eval_x, x_vals[0], x_vals[-1])
    
    # Generate initial CDF values and clamp to [0,1]
    cdf_y = spline(eval_x_clamped).clip(0.0, 1.0)
    
    # Ensure monotonicity (non-decreasing)
    cdf_y = np.maximum.accumulate(cdf_y)
    
    # Set boundary values if bounds are closed
    if not open_lower_bound:
        cdf_y[0] = 0.0
    if not open_upper_bound:
        cdf_y[-1] = 1.0
    
    # Strict enforcement of minimum step size with iterative approach
    def enforce_min_steps(y_values, min_step_size):
        """Enforce minimum step size between adjacent points"""
        result = y_values.copy()
        
        # First pass: enforce minimum steps
        for i in range(1, len(result)):
            if result[i] < result[i-1] + min_step_size:
                result[i] = min(result[i-1] + min_step_size, 1.0)
        
        # Second pass: ensure we don't exceed 1.0
        if result[-1] > 1.0:
            # If we've exceeded 1.0 before the end, rescale the steps
            overflow_idx = np.where(result > 1.0)[0][0]
            steps_remaining = len(result) - overflow_idx
            
            for i in range(overflow_idx, len(result)):
                t = (i - overflow_idx) / max(1, steps_remaining - 1)
                result[i] = min(1.0, result[overflow_idx-1] + (1.0 - result[overflow_idx-1]) * t)
            
            # Final check for minimum steps
            for i in range(overflow_idx, len(result)):
                if i > overflow_idx and result[i] < result[i-1] + min_step_size:
                    result[i] = result[i-1] + min_step_size
                    if result[i] > 1.0:
                        # If we exceed 1.0 again, cap at 1.0 and adjust previous values
                        result[i] = 1.0
                        # Backtrack and redistribute
                        for j in range(i-1, overflow_idx-1, -1):
                            max_allowed = result[j+1] - min_step_size
                            if result[j] > max_allowed:
                                result[j] = max_allowed
        
        return result
    
    # Apply strict step enforcement
    cdf_y = enforce_min_steps(cdf_y, min_step)

    cdf_y = _safe_cdf_bounds(cdf_y, open_lower_bound, open_upper_bound, min_step)

    required_range = (len(cdf_y)-1) * min_step
    available_range = cdf_y[-1] - cdf_y[0]
    
    # Double-check minimum step size requirement
    steps = np.diff(cdf_y)
    if np.any(steps < min_step):
        # If still violated, use a more aggressive approach
        print(f"Warning: Minimum step size still violated. Using aggressive step enforcement.")
        
        # Create a strictly monotonic sequence
        if not open_lower_bound:
            start_val = 0.0
        else:
            start_val = cdf_y[0]
            
        if not open_upper_bound:
            end_val = 1.0
        else:
            end_val = min(cdf_y[-1], 1.0)
        
        available_range = end_val - start_val
        # Ensure we have enough room for all steps
        required_range = (len(cdf_y) - 1) * min_step
        
        if required_range > available_range:
            # We don't have enough room for minimum steps
            raise ValueError(
                f"Cannot satisfy minimum step requirement: need {required_range:.6f} "
                f"but only have {available_range:.6f} available in CDF range"
            )
        
        # Create a new CDF with exactly min_step between points where needed
        # and distribute remaining range proportionally
        new_cdf = np.zeros_like(cdf_y)
        new_cdf[0] = start_val
        
        # Get the shape from original CDF but enforce minimum steps
        if len(cdf_y) > 2:
            # Calculate normalized shape from original CDF
            orig_shape = np.diff(cdf_y)
            orig_shape = np.maximum(orig_shape, min_step)  # Enforce minimum
            orig_shape = orig_shape / np.sum(orig_shape)   # Normalize
            
            # Allocate the available range according to shape but ensure minimum steps
            remaining = available_range - (len(cdf_y) - 1) * min_step
            extra_steps = remaining * orig_shape
            
            for i in range(1, len(new_cdf)):
                new_cdf[i] = new_cdf[i-1] + min_step + extra_steps[i-1]
        else:
            # Simple linear spacing if original shape is unavailable
            for i in range(1, len(new_cdf)):
                new_cdf[i] = new_cdf[i-1] + (available_range / (len(new_cdf) - 1))
        
        # Final validation
        if np.any(np.diff(new_cdf) < min_step - 1e-10):
            raise RuntimeError("Internal error: Step size enforcement failed")
        
        cdf_y = new_cdf

    # Final checks
    if np.any(np.diff(cdf_y) < min_step - 1e-10):
        problematic_indices = np.where(np.diff(cdf_y) < min_step - 1e-10)[0]
        error_msg = (f"Failed to enforce minimum step size at indices: {problematic_indices}, "
                    f"values: {np.diff(cdf_y)[problematic_indices]}")
        raise RuntimeError(error_msg)
    
    if not open_lower_bound and abs(cdf_y[0]) > 1e-10:
        raise RuntimeError(f"Failed to enforce lower bound: {cdf_y[0]}")
    
    if not open_upper_bound and abs(cdf_y[-1] - 1.0) > 1e-10:
        raise RuntimeError(f"Failed to enforce upper bound: {cdf_y[-1]}")
    
    
    return cdf_y.tolist()

async def get_numeric_forecast(question_details: dict, write=print):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details.get("fine_print", "")
    open_upper = question_details["open_upper_bound"]
    open_lower = question_details["open_lower_bound"]
    upper = question_details["scaling"]["range_max"]
    lower = question_details["scaling"]["range_min"]
    zero = question_details["scaling"].get("zero_point")
    unit = question_details.get("unit", "(unknown)")

    async def format_call(prompt):
        txt = prompt.format(
            title=title, today=today, background=background,
            resolution_criteria=resolution, fine_print=fine_print,
            lower_bound_message="" if open_lower else f"Cannot go below {lower}.",
            upper_bound_message="" if open_upper else f"Cannot go above {upper}.",
            units=unit,
            hint = f"The answer is expected to be above {lower} and below {upper}. Think carefully, and reconsider your sources, if your projections are outside this range."
        )
        return txt, await call_gpt_o3(txt)

    hist_prompt, hist_out = await format_call(NUMERIC_PROMPT_historical)
    curr_prompt, curr_out = await format_call(NUMERIC_PROMPT_current)

    hist_context = await process_search_queries(hist_out, forecaster_id="-1", question_details=question_details)
    curr_context = await process_search_queries(curr_out, forecaster_id="0", question_details=question_details)

    write(f"Historical output: {hist_out}\nContext: {hist_context}")
    write(f"Current output: {curr_out}\nContext: {curr_context}")

    prompt1 = NUMERIC_PROMPT_1.format(
        title=title, today=today, resolution_criteria=resolution,
        fine_print=fine_print, context=hist_context,
        units=unit, lower_bound_message="", upper_bound_message="",
        hint = f"The answer is expected to be above {lower} and below {upper}. Think carefully, and reconsider your sources, if your projections are outside this range."
    )

    base_forecasts = await asyncio.gather(
        call_claude(prompt1), call_claude(prompt1),
        call_gpt_o4_mini(prompt1), call_gpt_o3(prompt1),
        call_gpt_o3(prompt1)
    )

    for i, out in enumerate(base_forecasts):
        write(f"\nForecaster_{i+1} step 1 output:\n{out}")


    context_map = {
        str(i+1): f"Current context: {curr_context}\nPrior: {base_forecasts[i]}"
        for i in range(5)
    }

    prompts2 = [NUMERIC_PROMPT_2.format(
        title=title, today=today, resolution_criteria=resolution,
        fine_print=fine_print, context=context_map[str(i+1)],
        units=unit, lower_bound_message="", upper_bound_message="",
        hint = f"The answer is expected to be above {lower} and below {upper}. Think carefully, and reconsider your sources, if your projections are outside this range."
    ) for i in range(5)]

    step2_outputs = await asyncio.gather(
        call_claude(prompts2[0]), call_claude(prompts2[1]),
        call_gpt_o4_mini(prompts2[2]), call_gpt_o3(prompts2[3]),
        call_gpt_o3(prompts2[4])
    )

    all_cdfs = []
    final_outputs = []

    for i, output in enumerate(step2_outputs):
        try:
            parsed = extract_percentiles_from_response(output, verbose=True)
            parsed = enforce_strict_increasing(parsed)
            cdf = generate_continuous_cdf(parsed, open_upper, open_lower, upper, lower, zero)
            all_cdfs.append((cdf, 2 if (i == 4 or i == 3) else 1))
        except Exception as e:
            write(f"❌ Forecaster {i+1} failed: {e}")
        final_outputs.append(f"=== Forecaster {i+1} ===\n{output}\n")

    if len(all_cdfs) < 3:
        raise RuntimeError(f"🚨 Only {len(all_cdfs)} valid CDFs — need at least 3 to proceed")

    numer = sum(np.array(cdf) * weight for cdf, weight in all_cdfs)
    denom = sum(weight for _, weight in all_cdfs)
    combined = (numer / denom).tolist()

    if len(combined) != 201:
        raise RuntimeError(f"🚨 Combined CDF malformed: {len(combined)} points")

    comment = "Combined CDF: `" + str(combined[:5]) + "...`\n\n" + "\n\n".join(final_outputs)
    write(comment)
    return combined, comment
