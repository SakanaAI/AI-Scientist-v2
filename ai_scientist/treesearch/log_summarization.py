import json
import os
import sys

import openai

from .journal import Node, Journal

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, parent_dir)
import os
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers

base_url = os.getenv('BASE_URL')
client = openai.OpenAI(base_url=base_url if base_url else None)
model = "gpt-4o-2024-08-06"

report_summarizer_sys_msg = """You are an expert machine learning researcher.
You are given multiple experiment logs, each representing a node in a stage of exploring scientific ideas and implementations.
Your task is to aggregate these logs and provide scientifically insightful information.

Important instructions:
- Do NOT hallucinate or fabricate information that is not present in the logs.
- Do NOT introduce errors when repeating information from the logs.
- Identify notable insights or differences across the nodes without repeating the same information.
"""

output_format_control = """Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <THOUGHT>, thoroughly reason as an expert researcher. First, reason about each node, and then reason carefully by combining all the information. It is okay to be very detailed.

In <JSON>, provide the review in JSON format with the following fields in exactly this order:
- "Experiment_description": a string describing the conducted experiments
- "Significance": a string explaining why these experiments are important and what impact their findings might have
- "Description": a string describing the methods, steps taken, and any pertinent context needed to understand the experiments
- "List_of_included_plots": a list of plots that should be included. Each entry should include:
  • "path" (the plot path)
  • "description" (its original description)
  • "analysis" (your analysis of its scientific insights)
- "Key_numerical_results": a list of all important numerical results. Be selective about results that contribute to scientific insights. Each entry should include:
  • "result" (float number)
  • "description" (your short description of the result)
  • "analysis" (your analysis of its scientific insights)

Ensure the JSON is valid and properly formatted, as it will be automatically parsed."""

report_summarizer_prompt = (
    """You are given multiple experiment logs from different "nodes". Each node represents attempts and experiments exploring various scientific ideas.

One key point is that these nodes collectively illustrate a stage of testing different methods or approaches. The crucial task is to identify the scientific insights gleaned from this stage. For example, if one node tries method A and another node tries method B, you should compare any observed differences in performance or outcomes. Summarize both experiments in "Experiment_description", explain the processes in "Description", and place any key numerical findings (such as accuracy metrics, loss values, or runtime comparisons) in "Key_numerical_results."

Be concise and avoid repeating the same information from different nodes. You are encouraged to be thorough, but you do not need to include information from every node. Reason carefully about which results from which nodes are scientifically insightful.

The name of this stage of the experiment: {stage_name}

Here are the experiment logs of the nodes:

{node_infos}
"""
    + output_format_control
)

stage_aggregate_prompt = """You are given:

1) The summary of all previous experiment stages:
{prev_summary}

2) The name of the current experiment stage:
{stage_name}

3) The summary of the current stage:
{current_summary}


Your task is to produce an **updated comprehensive summary** of all experiment stages, including the newly introduced results from the current stage.

**Key Requirements:**
1. **No Loss of Critical Information**
   - Preserve valuable insights from the summary of all previous experiment stages. Do not remove or alter crucial texts.
   - Absolutely no hallucinations: if something does not appear in the logs or summaries, do not invent it. If something appears in the previous summary, do not make any mistakes when repeating it.
2. **Merge New Stage Data**
   - Integrate relevant results from the current stage into the existing summary.
   - Identify any overlap or repetition between new and old content, and remove only that which is clearly redundant or no longer scientifically insightful.
   - Be very careful if you want to remove or shorten the old content. By default, you can keep most of it and append new text.
   - Highlight how new findings connect to or differ from previous findings.
3. **Numerical Results and Visuals**
   - Carefully maintain the most insightful plots, figures, and numerical results.
   - Do not delete crucial quantitative findings or meaningful visual references.
4. **Length and Format**
   - The final summary will likely be **very long**. That is acceptable.
   - Present the updated summary in a format consistent with the style of the previous summaries (e.g., same section headings or structure).

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```
Ensure the JSON is valid and properly formatted, as it will be automatically parsed.
"""


def get_nodes_infos(nodes):
    node_infos = ""
    for n in nodes:
        node_info = f"Node ID: {n.id}\n"
        node_info += (
            f"Plan: {n.overall_plan}\n"
            if hasattr(n, "overall_plan")
            else "Plan: Not available\n"
        )
        node_info += (
            f"Analysis: {n.analysis}\n"
            if hasattr(n, "analysis")
            else "Analysis: Not available\n"
        )
        node_info += (
            f"Numerical Results: {n.metric}\n"
            if hasattr(n, "metric")
            else "Numerical Results: Not available\n"
        )
        node_info += "Plot Analyses:\n"
        if hasattr(n, "plot_analyses") and n.plot_analyses:
            for plot in n.plot_analyses:
                node_info += f"- Plot Path: {plot.get('plot_path', 'Not available')}, Description: {plot.get('analysis', 'Not available')}\n"
        else:
            node_info += "No plot analyses available\n"
        node_infos += node_info + "\n"
    return node_infos


def get_summarizer_prompt(journal, stage_name):
    good_leaf_nodes = [n for n in journal.good_nodes if n.is_leaf]
    if not good_leaf_nodes:
        print("NO GOOD LEAF NODES!!!")
        good_leaf_nodes = [n for n in journal.good_nodes]
    node_infos = get_nodes_infos(good_leaf_nodes)
    return report_summarizer_sys_msg, report_summarizer_prompt.format(
        node_infos=node_infos, stage_name=stage_name
    )


def get_stage_summary(journal, stage_name, model, client):
    sys_msg, prompt = get_summarizer_prompt(journal, stage_name)
    response = get_response_from_llm(prompt, client, model, sys_msg)
    summary_json = extract_json_between_markers(response[0])
    return summary_json


def get_node_log(node):
    node_dict = node.to_dict()
    # Only include keys that are relevant for logging/analysis
    keys_to_include = [
        "overall_plan",
        "analysis",
        "metric",
        "code",
        "plot_code",
        "plot_plan",
        "plot_analyses",
        "plot_paths",
        "vlm_feedback_summary",
        "exp_results_dir",
        "ablation_name",
    ]
    ret = {
        key: node_dict[key]
        for key in keys_to_include
        if key in node_dict and node_dict[key] is not None
    }
    if "exp_results_dir" in ret:
        original_dir_path = ret["exp_results_dir"]
        # Remove leading path segments before "experiment_results"
        idx = original_dir_path.find("experiment_results")
        short_dir_path = original_dir_path
        if idx != -1:
            short_dir_path = original_dir_path[idx:]

        ret["exp_results_dir"] = short_dir_path

        if os.path.isdir(original_dir_path):
            npy_files = [f for f in os.listdir(original_dir_path) if f.endswith(".npy")]
            # Prepend the shortened path to each .npy filename
            ret["exp_results_npy_files"] = [
                os.path.join(short_dir_path, f) for f in npy_files
            ]
        else:
            ret["exp_results_npy_files"] = []
    return ret


def update_summary(
    prev_summary, cur_stage_name, cur_journal, cur_summary, model, client, max_retry=5
):
    good_leaf_nodes = [n for n in cur_journal.good_nodes if n.is_leaf]
    node_infos = get_nodes_infos(good_leaf_nodes)
    prompt = stage_aggregate_prompt.format(
        prev_summary=prev_summary,
        stage_name=cur_stage_name,
        current_summary=cur_summary,
    )
    try:
        response = get_response_from_llm(
            prompt, client, model, "You are an expert machine learning researcher."
        )
        summary_json = extract_json_between_markers(response[0])
        assert summary_json
    except Exception as e:
        if max_retry > 0:
            print(f"Error occurred: {e}. Retrying... ({max_retry} attempts left)")
            return update_summary(
                prev_summary,
                cur_stage_name,
                cur_journal,
                cur_summary,
                model,
                client,
                max_retry - 1,
            )
        else:
            print(f"Failed to update summary after multiple attempts. Error: {e}")
            raise
    return summary_json


overall_plan_summarizer_prompt = """You have been provided with the plans for both the parent node and the current node. Your task is to synthesize a comprehensive summary of the overall plan by integrating details from both the parent and current node plans.
The summary should be thorough and clearly articulate the underlying motivations.
For example, if in your previous overall plan you were experimenting with a new idea, and now your current plan is to fix certain bugs in the previous implementation, your returned overall plan should focus on your previous overall plan, and briefly mention that the current plan includes bug fixes. If your current plan is more about implementing new ideas, then you should summarize that thoroughly along with the previous overall plan.
The goal is to create a comprehensive summary of all historical plans, focusing on the main scientific planning and objectives.

Previous overall plan:
{prev_overall_plan}

Current plan:
{current_plan}

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <THOUGHT>, thoroughly reason as an expert researcher. First, reason over each node, and then carefully combine all information. It is okay to be very detailed.

In <JSON>, provide the review in JSON format with the following field in exactly this order:
- "overall_plan": a string that describes the overall plan based on the current and previous overall plans

Ensure the JSON is valid and properly formatted, as it will be automatically parsed.
"""


def annotate_history(journal):
    for node in journal.nodes:
        if node.parent:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = get_response_from_llm(
                        overall_plan_summarizer_prompt.format(
                            prev_overall_plan=node.parent.overall_plan,
                            current_plan=node.plan,
                        ),
                        client,
                        model,
                        report_summarizer_sys_msg,
                    )
                    node.overall_plan = extract_json_between_markers(response[0])[
                        "overall_plan"
                    ]
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed after {max_retries} attempts. Error: {e}")
                        raise
                    print(
                        f"Error occurred: {e}. Retrying... ({max_retries - retry_count} attempts left)"
                    )
        else:
            node.overall_plan = node.plan


def overall_summarize(journals):
    from concurrent.futures import ThreadPoolExecutor

    def process_stage(idx, stage_tuple):
        stage_name, journal = stage_tuple
        annotate_history(journal)
        if idx in [1, 2]:
            best_node = journal.get_best_node()
            # get multi-seed results and aggregater node
            child_nodes = best_node.children
            multi_seed_nodes = [
                n for n in child_nodes if n.is_seed_node and not n.is_seed_agg_node
            ]
            agg_node = None
            for n in child_nodes:
                if n.is_seed_node and n.is_seed_agg_node:
                    agg_node = n
                    break
            if agg_node is None:
                # skip agg node
                return {
                    "best node": get_node_log(best_node),
                    "best node with different seeds": [
                        get_node_log(n) for n in multi_seed_nodes
                    ],
                }
            else:
                return {
                    "best node": get_node_log(best_node),
                    "best node with different seeds": [
                        get_node_log(n) for n in multi_seed_nodes
                    ],
                    "aggregated results of nodes with different seeds": get_node_log(
                        agg_node
                    ),
                }
        elif idx == 3:
            good_leaf_nodes = [
                n for n in journal.good_nodes if n.is_leaf and n.ablation_name
            ]
            return [get_node_log(n) for n in good_leaf_nodes]
        elif idx == 0:
            summary_json = get_stage_summary(journal, stage_name, model, client)
            return summary_json

    from tqdm import tqdm

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_stage, range(len(list(journals))), journals),
                desc="Processing stages",
                total=len(list(journals)),
            )
        )
        draft_summary, baseline_summary, research_summary, ablation_summary = results

    return draft_summary, baseline_summary, research_summary, ablation_summary


if __name__ == "__main__":
    # Test
    example_path = "logs/247-run"

    def load_stage_folders(base_path):
        """
        Load the folders that start with 'stage_' followed by a number.

        Args:
            base_path (str): The base directory path where stage folders are located.

        Returns:
            list: A sorted list of stage folder paths.
        """
        stage_folders = []
        for folder_name in os.listdir(base_path):
            if folder_name.startswith("stage_"):
                stage_folders.append(os.path.join(base_path, folder_name))
        return sorted(stage_folders, key=lambda x: int(x.split("_")[1]))

    def reconstruct_journal(journal_data):
        # Create a mapping of node IDs to Node instances
        id_to_node = {}
        for node_data in journal_data["nodes"]:
            # Remove unused or invalid keys if needed
            if "actionable_insights_from_plots" in node_data:
                del node_data["actionable_insights_from_plots"]
            node = Node.from_dict(node_data)
            id_to_node[node.id] = node

        # Set up parent-child relationships using node2parent
        for node_id, parent_id in journal_data["node2parent"].items():
            child_node = id_to_node[node_id]
            parent_node = id_to_node[parent_id]
            child_node.parent = parent_node
            parent_node.children.add(child_node)

        # Create a Journal and add all nodes
        journal = Journal()
        journal.nodes.extend(id_to_node.values())

        return journal

    # Example usage
    stage_folders = load_stage_folders(example_path)
    journals = []
    for index, folder in enumerate(stage_folders, start=1):
        print(f"Stage {index}: {folder}")
        stage_name = os.path.basename(folder)
        journal_path = os.path.join(folder, "journal.json")
        if os.path.exists(journal_path):
            with open(journal_path, "r") as file:
                journal_data = json.load(file)
                print(f"Loaded journal.json for Stage {index}")
        else:
            print(f"No journal.json found for Stage {index}")
        journal = reconstruct_journal(journal_data)
        journals.append((stage_name, journal))

    # Convert manager journals to list of (stage_name, journal) tuples
    (
        draft_summary,
        baseline_summary,
        research_summary,
        ablation_summary,
    ) = overall_summarize(journals)
    log_dir = "logs/247-run"
    draft_summary_path = log_dir + "/draft_summary.json"
    baseline_summary_path = log_dir + "/baseline_summary.json"
    research_summary_path = log_dir + "/research_summary.json"
    ablation_summary_path = log_dir + "/ablation_summary.json"

    with open(draft_summary_path, "w") as draft_file:
        json.dump(draft_summary, draft_file, indent=2)

    with open(baseline_summary_path, "w") as baseline_file:
        json.dump(baseline_summary, baseline_file, indent=2)

    with open(research_summary_path, "w") as research_file:
        json.dump(research_summary, research_file, indent=2)

    with open(ablation_summary_path, "w") as ablation_file:
        json.dump(ablation_summary, ablation_file, indent=2)

    print(f"Summary reports written to files:")
    print(f"- Draft summary: {draft_summary_path}")
    print(f"- Baseline summary: {baseline_summary_path}")
    print(f"- Research summary: {research_summary_path}")
    print(f"- Ablation summary: {ablation_summary_path}")
