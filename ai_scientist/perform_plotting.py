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

MAX_FIGURES = 12

AGGREGATOR_SYSTEM_MSG = f"""You are an ambitious AI researcher who is preparing final plots for a scientific paper submission.
You have multiple experiment summaries (baseline, research, ablation), each possibly containing references to different plots or numerical insights.
There is also a top-level 'research_idea.md' file that outlines the overarching research direction.
Your job is to produce ONE compact Python script that aggregates and visualizes the final results for a research paper.

Key points:
1) Combine or replicate relevant existing plotting code, referencing how data was originally generated (from code references) to ensure correctness.
2) Create a small final set of high-value scientific plots, stored in 'figures/' only (since only those are used in the final paper).
3) Make sure to use existing .npy data for analysis; do NOT hallucinate data. If single numeric results are needed, these may be copied from the JSON summaries.
4) Only create plots where the data is best presented as a figure and not as a table. E.g. don't use bar plots if the data is hard to visually compare.
5) The final aggregator script must be in triple backticks and stand alone so it can be dropped into a codebase and run.
6) If there are plots based on synthetic data, include them in the appendix.

Implement best practices:
- Do not produce extraneous or irrelevant plots.
- Maintain clarity, minimal but sufficient code.
- Demonstrate thoroughness with a short script that is likely to execute correctly.
- Do NOT reference non-existent files or images.
- Use the .npy files to get data for the plots and key numbers from the JSON summaries.
- Demarcate each individual plot, and put them in separate try-catch blocks so that the failure of one plot does not affect the others.
- Make sure to only create plots that are unique and needed for the final paper and appendix. Generate 4 to 6 figure files, never more than 8.
- Aim to aggregate multiple related results into one figure. You can place up to 3 subplots in one row.
- Provide well-labeled plots (axes, legends, titles) that highlight main findings. Use informative names everywhere, including in the legend for referencing them in the final paper. Make sure the legend is always visible.
- Make the plots look professional (if applicable, no top and right spines, dpi of 300, adequate ylim, etc.).
- Do not use labels with underscores, e.g. "loss_vs_epoch" should be "loss vs epoch".
- Prefer robust numeric plots over brittle prediction tables if the data shape is unclear.
- Keep the script compact, ideally under 160 lines, and avoid long comments.

Your output should be the entire Python aggregator script in triple backticks.
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
- Focus on the smallest set of strong figures that still covers the main result plus key ablations.
- If you need .npy paths from the summary, only copy those paths directly (rather than copying and parsing the entire summary).

Your generated Python script must:
1) Load or refer to relevant data and .npy files from these summaries. Use the full and exact file paths in the summary JSONs.
2) Synthesize or directly create final, scientifically meaningful plots for a final research paper, referencing the original code if needed to see how the data was generated.
3) Carefully combine or replicate relevant existing plotting code to produce these final aggregated plots in 'figures/' only, since only those are used in the final paper.
4) Do not hallucinate data. Data must either be loaded from .npy files or copied from the JSON summaries.
5) The aggregator script must be fully self-contained, and place the final plots in 'figures/'.
6) This aggregator script should produce a concise final set of scientific plots, reflecting the main findings from the experiment data.
7) Make sure that every plot is unique and not duplicated from the original plots. Delete any duplicate plots if necessary.
8) Each figure can have up to 3 subplots using fig, ax = plt.subplots(1, 3).
9) Use a font size larger than the default for plot labels and titles to ensure they are readable in the final PDF paper.
10) Keep the script compact and executable. Prefer 4 to 6 figure files and avoid long example tables if the data shape is uncertain.


Below are the summaries in JSON:

{combined_summaries_str}

Respond with a Python script in triple backticks.
"""


def extract_code_snippet(text: str) -> str:
    """
    Look for a Python code block in triple backticks in the LLM response.
    Return only that code. If no code block is found, return an empty string so
    we do not accidentally write natural language into a Python file.
    """
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[0].strip()

    start = text.find("```")
    if start != -1:
        snippet = text[start + 3 :]
        if snippet.startswith("python"):
            snippet = snippet[len("python") :]
        return snippet.lstrip()

    return ""


def validate_python_snippet(code: str) -> str | None:
    """Return a syntax error message if the snippet is not valid Python."""
    try:
        compile(code, "<auto_plot_aggregator>", "exec")
    except SyntaxError as error:
        return f"SyntaxError: {error.msg} (line {error.lineno})"
    except Exception as error:
        return f"ValidationError: {error}"
    return None


def run_aggregator_script(
    aggregator_code, aggregator_script_path, base_folder, script_name
):
    if not aggregator_code.strip():
        print("No aggregator code was provided. Skipping aggregator script run.")
        return "No Python code block was provided by the model."

    validation_error = validate_python_snippet(aggregator_code)
    if validation_error:
        print("Aggregator script validation failed before execution.")
        print(validation_error)
        return validation_error

    with open(aggregator_script_path, "w") as f:
        f.write(aggregator_code)

    print(
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
        print("Aggregator script ran successfully.")
    except subprocess.CalledProcessError as e:
        aggregator_out = (e.stdout or "") + "\n" + (e.stderr or "")
        print("Error: aggregator script returned a non-zero exit code.")
        print(e)
    except Exception as e:
        aggregator_out = str(e)
        print("Error while running aggregator script.")
        print(e)

    return aggregator_out


def aggregate_plots(
    base_folder: str, model: str = "o1-2024-12-17", n_reflections: int = 5
) -> None:
    filename = "auto_plot_aggregator.py"
    aggregator_script_path = os.path.join(base_folder, filename)
    figures_dir = os.path.join(base_folder, "figures")

    # Clean up previous files
    if os.path.exists(aggregator_script_path):
        os.remove(aggregator_script_path)
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
        print(f"Cleaned up previous figures directory")

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
        )
    except Exception:
        traceback.print_exc()
        print("Failed to get aggregator script from LLM.")
        return

    aggregator_code = extract_code_snippet(response)
    if not aggregator_code.strip():
        print(
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
        print(f"[{i + 1} / {n_reflections}]: Number of figures: {figure_count}")
        # Reflection prompt with reminder for common checks and early exit
        reflection_prompt = f"""We have run your aggregator script and it produced {figure_count} figure(s). The script's output is:
```
{aggregator_out}
```

Please criticize the current script for any flaws including but not limited to:
- Are these enough plots for a final paper submission? Don't create more than {MAX_FIGURES} plots.
- Have you made sure to both use key numbers and generate more detailed plots from .npy files?
- Does the figure title and legend have informative and descriptive names? These plots are the final versions, ensure there are no comments or other notes.
- Can you aggregate multiple plots into one figure if suitable?
- Do the labels have underscores? If so, replace them with spaces.
- Make sure that every plot is unique and not duplicated from the original plots.

If you believe you are done, simply say: "I am done".
Otherwise, respond with exactly one updated Python script in triple backticks and no other prose.
Keep the revised script compact and executable, ideally under 160 lines, and prefer fewer stronger figures over many brittle ones."""

        print("[green]Reflection prompt:[/green] ", reflection_prompt)
        try:
            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=model_name,
                system_message=AGGREGATOR_SYSTEM_MSG,
                print_debug=False,
                msg_history=msg_history,
            )

        except Exception:
            traceback.print_exc()
            print("Failed to get reflection from LLM.")
            return

        # Early-exit check
        if figure_count > 0 and "I am done" in reflection_response:
            print("LLM indicated it is done with reflections. Exiting reflection loop.")
            break

        aggregator_new_code = extract_code_snippet(reflection_response)

        if not aggregator_new_code.strip() and "I am done" not in reflection_response:
            aggregator_out = (
                "The previous response did not contain a Python code block. "
                "Return only a Python script in triple backticks or exactly 'I am done'."
            )
            print(aggregator_out)
            continue

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
            print(
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
        default="o1-2024-12-17",
        help="LLM model to use (default: o1-2024-12-17).",
    )
    parser.add_argument(
        "--reflections",
        type=int,
        default=5,
        help="Number of reflection steps to attempt (default: 5).",
    )
    args = parser.parse_args()
    aggregate_plots(
        base_folder=args.folder, model=args.model, n_reflections=args.reflections
    )


if __name__ == "__main__":
    main()
