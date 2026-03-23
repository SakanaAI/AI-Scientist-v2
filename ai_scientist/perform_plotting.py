import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from rich import print

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.utils.token_tracker import token_tracker
from ai_scientist.perform_icbinb_writeup import (
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
)
import logging
logger = logging.getLogger(__name__)


MAX_FIGURES = 12

# Global LLM default is 4096 — too small for long auto_plot_aggregator.py (truncates mid-line).
# DeepSeek API allows at most 8192 completion tokens (larger values return HTTP 400); llm.py clamps.
AGGREGATOR_LLM_MAX_TOKENS = 8192


def _syntax_failure_in_output(aggregator_out: str) -> bool:
    return (
        "Static check failed" in aggregator_out
        or "SyntaxError:" in aggregator_out
        or "IndentationError:" in aggregator_out
    )


def _reflection_script_attachment(aggregator_code: str) -> str:
    """When syntax fails, include the broken script so the model can patch it (not hallucinate anew)."""
    if not aggregator_code.strip():
        return ""
    cap = 120_000
    body = (
        aggregator_code
        if len(aggregator_code) <= cap
        else aggregator_code[:cap]
        + "\n\n# ... [truncated for prompt size; edit the lines above, especially near the error] ..."
    )
    return f"""

--- CURRENT FULL SCRIPT (repair this exact code; complete every statement, match try/except pairs) ---
```python
{body}
```
"""


AGGREGATOR_SYSTEM_MSG = f"""You are an ambitious AI researcher who is preparing final plots for a scientific paper submission.
You have multiple experiment summaries (baseline, research, ablation), each possibly containing references to different plots or numerical insights.
There is also a top-level 'research_idea.md' file that outlines the overarching research direction.
Your job is to produce ONE Python script that fully aggregates and visualizes the final results for a comprehensive research paper.

Key points:
1) Combine or replicate relevant existing plotting code, referencing how data was originally generated (from code references) to ensure correctness.
2) Create a complete set of final scientific plots, stored in 'figures/' only (since only those are used in the final paper).
3) Use ONLY empirical inputs: load arrays from .npy paths given in the experiment summaries, and scalar/table values explicitly present in those JSON summaries. Do NOT invent or hallucinate metrics.
4) Only create plots where the data is best presented as a figure and not as a table. E.g. don't use bar plots if the data is hard to visually compare.
5) The script is saved as auto_plot_aggregator.py and executed with `python`; the file MUST be valid Python only. Do NOT put markdown fences (``` or ```python) inside the script body. If you use fences in chat, you MUST include a closing fence; prefer outputting raw Python with no fences.

CRITICAL — no fabricated empirical figures:
- Do NOT create "synthetic", "demonstration", "illustrative", or "toy" plots that use hardcoded numbers, `numpy.random`, or made-up arrays to stand in for experimental results (including fake multi-model "leaderboards" or seed-sweep curves with invented model names).
- Do NOT plot real-world model names (e.g. GPT-4, Claude, Llama, Gemini) with scores unless those exact values appear in the provided summaries or loaded .npy data.
- If the research idea mentions comparisons (e.g. multi-model leaderboard) but the JSON/.npy data do not contain those results, SKIP that figure — do not fill gaps with invented data. Fewer honest plots is better than extra false ones.
- Preprocessing (smoothing, binning, normalization) is allowed ONLY when applied to real loaded data; the plotted values must still trace back to .npy or JSON fields.

Implement best practices:
- Do not produce extraneous or irrelevant plots.
- Maintain clarity, minimal but sufficient code.
- Demonstrate thoroughness for a final research paper submission using only verifiable data paths.
- Do NOT reference non-existent files or images.
- Use the .npy files to get data for the plots and key numbers from the JSON summaries.
- Demarcate each individual plot, and put them in separate try-except blocks so that the failure of one plot does not affect the others. Every `try:` MUST be followed by a non-empty `except` or `finally` block (invalid try/except is a SyntaxError).
- Before finishing, mentally verify: matching parentheses/brackets/braces, closed strings, and complete try/except/finally blocks. The script is compiled before execution; any SyntaxError will block the run.
- Make sure to only create plots that are unique and needed for the final paper and appendix. A good number could be around {MAX_FIGURES} plots in total.
- Aim to aggregate multiple figures into one plot if suitable, i.e. if they are all related to the same topic. You can place up to 3 plots in one row.
- Provide well-labeled plots (axes, legends, titles) that highlight main findings. Use informative names everywhere, including in the legend for referencing them in the final paper. Make sure the legend is always visible.
- Make the plots look professional (if applicable, no top and right spines, dpi of 300, adequate ylim, etc.).
- Do not use labels with underscores, e.g. "loss_vs_epoch" should be "loss vs epoch".
- For image-based plots, select categories/classes only as present in the actual data; do not invent extra categories for visual effect.

Your output should be the entire Python aggregator script. Prefer raw Python with no markdown; if you use triple backticks, the fenced region must be ONLY valid Python (no ``` lines inside) and must include an opening and closing fence.
Never leave a line ending with only a partial expression (e.g. "ax.set", bare "except") — that usually means the response was cut off; keep the script shorter or split logic so every line is complete.
"""


def build_aggregator_prompt(combined_summaries_str, idea_text):
    return f"""
We have three JSON summaries of scientific experiments: baseline, research, ablation.
They may contain lists of figure descriptions, code to generate the figures, and paths to the .npy files containing the numerical results.
Our goal is to produce final, publishable figures.

--- RESEARCH IDEA ---
```
{idea_text}
```

IMPORTANT:
- The aggregator script must load existing .npy experiment data from the "exp_results_npy_files" fields (ONLY using full and exact file paths in the summary JSONs) for thorough plotting.
- It should call os.makedirs("figures", exist_ok=True) before saving any plots.
- Aim for clear empirical visuals in 'figures/' that reflect what was actually computed in these runs. Omit figure types that would require data you do not have.
- If you need .npy paths from the summary, only copy those paths directly (rather than copying and parsing the entire summary).
- Do NOT add figures backed by random numbers, hardcoded score lists, or synthetic arrays to mimic missing experiments (no fake leaderboards or seed curves).

Your generated Python script must:
1) Load or refer to relevant data and .npy files from these summaries. Use the full and exact file paths in the summary JSONs.
2) Build final, scientifically meaningful plots by aggregating and visualizing ONLY that empirical data (and numeric fields explicitly present in the JSON). "Synthesize" here means combine real series — not invent data.
3) Carefully combine or replicate relevant existing plotting code to produce these final aggregated plots in 'figures/' only, since only those are used in the final paper.
4) Do not hallucinate data. Every plotted quantity must come from loaded .npy files or explicit numeric/text fields in the JSON summaries below.
5) The aggregator script must be fully self-contained, and place the final plots in 'figures/'.
6) This aggregator script should visualize the major findings supported by the experiment data actually present in the summaries — not every hypothetical figure suggested by the prose in the research idea.
7) Make sure that every plot is unique and not duplicated from the original plots. Delete any duplicate plots if necessary.
8) Each figure can have up to 3 subplots using fig, ax = plt.subplots(1, 3).
9) Use a font size larger than the default for plot labels and titles to ensure they are readable in the final PDF paper.


Below are the summaries in JSON:

{combined_summaries_str}

Respond with a Python script. The code will be written to a .py file and run as-is: first line must be valid Python (e.g. import or # comment), not ``` or ```python. Prefer raw Python without markdown fences.
It must pass Python's parser with zero syntax errors (including IndentationError) before it is executed.
Do not output plots based on synthetic or fabricated data; ground every figure in the data described above.
"""


def extract_code_snippet(text: str) -> str:
    """
    Extract runnable Python from an LLM response.

    Models often wrap code in ``` / ```python blocks. If they open a fence but
    omit the closing ```, the old logic returned the full string and the saved
    .py file started with ```python -> SyntaxError. We strip incomplete fences
    and prefer the largest well-formed fenced block when multiple exist.
    """
    text = (text or "").strip()
    if not text:
        return text

    def _strip_trailing_fence(s: str) -> str:
        s = s.rstrip()
        if s.endswith("```"):
            return s[:-3].rstrip()
        return s

    # Well-formed fenced blocks (non-greedy inner match)
    for pat in (
        r"```(?:python|py)\s*\r?\n(.*?)```",
        r"```(?:python|py)\s+(.*?)```",
        r"```\s*\r?\n(.*?)```",
    ):
        m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Unclosed fence after optional preamble (e.g. "Here is the script:" then ```python)
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            body = "\n".join(lines[idx + 1 :])
            return _strip_trailing_fence(body).strip()

    # No fences: assume the model returned raw Python
    return text


def validate_python_syntax(code: str, filename: str = "auto_plot_aggregator.py") -> tuple[bool, str]:
    """
    Static check: must compile before we write/run. Catches SyntaxError / IndentationError.
    Runtime bugs (NameError, etc.) are not detected here.
    """
    try:
        compile(code, filename, "exec")
        return True, ""
    except SyntaxError as e:
        parts = [f"SyntaxError: {e.msg} ({filename}, line {e.lineno})"]
        if e.text:
            parts.append(f"  Offending line: {e.text.rstrip()}")
        lines = code.splitlines()
        if e.lineno is not None and lines:
            lo = max(0, e.lineno - 3)
            hi = min(len(lines), e.lineno + 2)
            ctx = "\n".join(f"  {i + 1:4d} | {lines[i]}" for i in range(lo, hi))
            parts.append("Context:\n" + ctx)
        return False, "\n".join(parts)
    except Exception as e:
        return False, f"compile() failed: {e!r}"


def run_aggregator_script(
    aggregator_code, aggregator_script_path, base_folder, script_name
):
    if not aggregator_code.strip():
        logger.info("No aggregator code was provided. Skipping aggregator script run.")
        return ""

    ok, syntax_err = validate_python_syntax(aggregator_code, script_name)
    if not ok:
        logger.info(
            "Aggregator script failed static syntax check; not writing or executing:\n%s",
            syntax_err,
        )
        return (
            syntax_err
            + "\n\n[Static check failed: fix syntax errors above. The script was not run.]"
        )

    with open(aggregator_script_path, "w") as f:
        f.write(aggregator_code)

    logger.info(
        f"Aggregator script written to '{aggregator_script_path}'. Attempting to run it..."
    )

    aggregator_out = ""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=base_folder,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        aggregator_out = result.stdout + "\n" + result.stderr
        logger.info("Aggregator script ran successfully.")
    except subprocess.CalledProcessError as e:
        aggregator_out = (e.stdout or "") + "\n" + (e.stderr or "")
        logger.info("Error: aggregator script returned a non-zero exit code.")
        logger.info(e)
    except Exception as e:
        aggregator_out = str(e)
        logger.info("Error while running aggregator script.")
        logger.info(e)

    return aggregator_out


def aggregate_plots(
    base_folder: str,
    model: str = "deepseek-v3.2",
    n_reflections: int = 5,
    aggregator_max_tokens: int = AGGREGATOR_LLM_MAX_TOKENS,
) -> None:
    filename = "auto_plot_aggregator.py"
    aggregator_script_path = os.path.join(base_folder, filename)
    figures_dir = os.path.join(base_folder, "figures")

    # Clean up previous files
    if os.path.exists(aggregator_script_path):
        os.remove(aggregator_script_path)
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
        logger.info(f"Cleaned up previous figures directory")

    idea_text = load_idea_text(base_folder)
    exp_summaries = load_exp_summaries(base_folder)
    filtered_summaries_for_plot_agg = filter_experiment_summaries(
        exp_summaries, step_name="plot_aggregation"
    )
    # Convert them to one big JSON string for context
    combined_summaries_str = json.dumps(filtered_summaries_for_plot_agg, indent=2)

    # Build aggregator prompt
    aggregator_prompt = build_aggregator_prompt(combined_summaries_str, idea_text)

    # Call LLM
    client, model_name = create_client(model)
    response, msg_history = None, []
    try:
        response, msg_history = get_response_from_llm(
            prompt=aggregator_prompt,
            client=client,
            model=model_name,
            system_message=AGGREGATOR_SYSTEM_MSG,
            print_debug=False,
            msg_history=msg_history,
            max_tokens=aggregator_max_tokens,
        )
    except Exception:
        traceback.print_exc()
        logger.info("Failed to get aggregator script from LLM.")
        return

    aggregator_code = extract_code_snippet(response)
    if not aggregator_code.strip():
        logger.info(
            "No Python code block was found in LLM response. Full response:\n", response
        )
        return

    # First run of aggregator script
    aggregator_out = run_aggregator_script(
        aggregator_code, aggregator_script_path, base_folder, filename
    )

    # Multiple reflection loops
    for i in range(n_reflections):
        # Check number of figures
        figure_count = 0
        if os.path.exists(figures_dir):
            figure_count = len(
                [
                    f
                    for f in os.listdir(figures_dir)
                    if os.path.isfile(os.path.join(figures_dir, f))
                ]
            )
        logger.info(f"[{i + 1} / {n_reflections}]: Number of figures: {figure_count}")
        script_fix = (
            _reflection_script_attachment(aggregator_code)
            if _syntax_failure_in_output(aggregator_out)
            else ""
        )
        # Reflection prompt with reminder for common checks and early exit
        reflection_prompt = f"""We have run your aggregator script and it produced {figure_count} figure(s). The script's output is:
```
{aggregator_out}
```
{script_fix}
If you see a SyntaxError, IndentationError, or static check message above, fix those FIRST — the script did not run until syntax is valid. Then address other issues.

Please criticize the current script for any flaws including but not limited to:
- Remove or rewrite any figure that uses hardcoded scores, `numpy.random` (except for layout seeds), or synthetic arrays to simulate empirical results; every plot must trace to .npy paths or JSON fields from the experiment summaries.
- Are these enough plots for a final paper submission? Don't create more than {MAX_FIGURES} plots.
- Have you made sure to both use key numbers and generate more detailed plots from .npy files?
- Does the figure title and legend have informative and descriptive names? These plots are the final versions, ensure there are no comments or other notes.
- Can you aggregate multiple plots into one figure if suitable?
- Do the labels have underscores? If so, replace them with spaces.
- Make sure that every plot is unique and not duplicated from the original plots.

If you believe you are done, simply say: "I am done". Otherwise, provide an updated script as valid Python only (prefer no markdown fences; if fenced, include closing ```)."""

        logger.info("[green]Reflection prompt:[/green] %s", reflection_prompt)
        try:
            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=model_name,
                system_message=AGGREGATOR_SYSTEM_MSG,
                print_debug=False,
                msg_history=msg_history,
                max_tokens=aggregator_max_tokens,
            )

        except Exception:
            traceback.print_exc()
            logger.info("Failed to get reflection from LLM.")
            return

        # Early-exit check
        if figure_count > 0 and "I am done" in reflection_response:
            logger.info("LLM indicated it is done with reflections. Exiting reflection loop.")
            break

        aggregator_new_code = extract_code_snippet(reflection_response)

        # If new code is provided and differs, run again
        if (
            aggregator_new_code.strip()
            and aggregator_new_code.strip() != aggregator_code.strip()
        ):
            aggregator_code = aggregator_new_code
            aggregator_out = run_aggregator_script(
                aggregator_code, aggregator_script_path, base_folder, filename
            )
        else:
            logger.info(
                f"No new aggregator script was provided or it was identical. Reflection step {i+1} complete."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate and execute a final plot aggregation script with LLM assistance."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the experiment folder with summary JSON files.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v3.2",
        help="LLM model to use (default: deepseek-v3.2).",
    )
    parser.add_argument(
        "--reflections",
        type=int,
        default=5,
        help="Number of reflection steps to attempt (default: 5).",
    )
    parser.add_argument(
        "--aggregator-max-tokens",
        type=int,
        default=AGGREGATOR_LLM_MAX_TOKENS,
        help=(
            "Max completion tokens for aggregator LLM calls (default: %(default)s). "
            "DeepSeek caps at 8192 (values above are clamped). Other models may allow more."
        ),
    )
    args = parser.parse_args()
    aggregate_plots(
        base_folder=args.folder,
        model=args.model,
        n_reflections=args.reflections,
        aggregator_max_tokens=args.aggregator_max_tokens,
    )


if __name__ == "__main__":
    main()
