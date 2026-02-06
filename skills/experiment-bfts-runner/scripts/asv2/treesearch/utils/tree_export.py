"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph
from ..journal import Journal

from rich import print


def get_edges(journal: Journal):
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)


def generate_layout(n_nodes, edges, layout_type="rt"):
    """Generate visual layout of graph"""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]"""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0))
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def get_completed_stages(log_dir):
    """
    Determine completed stages by checking for the existence of stage directories
    that contain evidence of completion (tree_data.json, tree_plot.html, or journal.json).

    Returns:
        list: A list of stage names (e.g., ["Stage_1", "Stage_2"])
    """
    completed_stages = []

    # Check for each stage (1-4)
    for stage_num in range(1, 5):
        prefix = f"stage_{stage_num}"

        # Find all directories that match this stage number
        matching_dirs = [
            d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
        ]

        # Check if any of these directories have completion evidence
        for stage_dir in matching_dirs:
            has_tree_data = (stage_dir / "tree_data.json").exists()
            has_tree_plot = (stage_dir / "tree_plot.html").exists()
            has_journal = (stage_dir / "journal.json").exists()

            if has_tree_data or has_tree_plot or has_journal:
                # Found evidence this stage was completed
                completed_stages.append(f"Stage_{stage_num}")
                break  # No need to check other directories for this stage

    return completed_stages


def cfg_to_tree_struct(cfg, jou: Journal, out_path: Path = None):
    edges = list(get_edges(jou))
    print(f"[red]Edges: {edges}[/red]")
    try:
        gen_layout = generate_layout(len(jou), edges)
    except Exception as e:
        print(f"Error in generate_layout: {e}")
        raise
    try:
        layout = normalize_layout(gen_layout)
    except Exception as e:
        print(f"Error in normalize_layout: {e}")
        raise

    best_node = jou.get_best_node(cfg=cfg)
    metrics = []
    is_best_node = []

    for n in jou:
        # print(f"Node {n.id} exc_stack: {type(n.exc_stack)} = {n.exc_stack}")
        if n.metric:
            # Pass the entire metric structure for the new format
            if isinstance(n.metric.value, dict) and "metric_names" in n.metric.value:
                metrics.append(n.metric.value)
            else:
                # Handle legacy format by wrapping it in the new structure
                metrics.append(
                    {
                        "metric_names": [
                            {
                                "metric_name": n.metric.name or "value",
                                "lower_is_better": not n.metric.maximize,
                                "description": n.metric.description or "",
                                "data": [
                                    {
                                        "dataset_name": "default",
                                        "final_value": n.metric.value,
                                        "best_value": n.metric.value,
                                    }
                                ],
                            }
                        ]
                    }
                )
        else:
            metrics.append(None)

        # Track whether this is the best node
        is_best_node.append(n is best_node)

    tmp = {}

    # Add each item individually with error handling
    try:
        tmp["edges"] = edges
    except Exception as e:
        print(f"Error setting edges: {e}")
        raise

    try:
        tmp["layout"] = layout.tolist()
    except Exception as e:
        print(f"Error setting layout: {e}")
        raise

    try:
        tmp["plan"] = [
            textwrap.fill(str(n.plan) if n.plan is not None else "", width=80)
            for n in jou.nodes
        ]
    except Exception as e:
        print(f"Error setting plan: {e}")
        raise

    try:
        tmp["code"] = [n.code for n in jou]
    except Exception as e:
        print(f"Error setting code: {e}")
        raise

    try:
        tmp["term_out"] = [
            textwrap.fill(str(n._term_out) if n._term_out is not None else "", width=80)
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting term_out: {e}")
        print(f"n.term_out: {n._term_out}")
        raise

    try:
        tmp["analysis"] = [
            textwrap.fill(str(n.analysis) if n.analysis is not None else "", width=80)
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting analysis: {e}")
        raise

    try:
        tmp["exc_type"] = [n.exc_type for n in jou]
    except Exception as e:
        print(f"Error setting exc_type: {e}")
        raise

    try:
        tmp["exc_info"] = [n.exc_info for n in jou]
    except Exception as e:
        print(f"Error setting exc_info: {e}")
        raise

    try:
        tmp["exc_stack"] = [n.exc_stack for n in jou]
    except Exception as e:
        print(f"Error setting exc_stack: {e}")
        raise

    try:
        tmp["exp_name"] = cfg.exp_name
    except Exception as e:
        print(f"Error setting exp_name: {e}")
        raise

    try:
        tmp["metrics"] = metrics
    except Exception as e:
        print(f"Error setting metrics: {e}")
        raise

    try:
        tmp["is_best_node"] = is_best_node
    except Exception as e:
        print(f"Error setting is_best_node: {e}")
        raise

    try:
        tmp["plots"] = [n.plots for n in jou]
    except Exception as e:
        print(f"Error setting plots: {e}")
        raise

    try:
        tmp["plot_paths"] = [n.plot_paths for n in jou]
    except Exception as e:
        print(f"Error setting plot_paths: {e}")
        raise

    try:
        tmp["plot_analyses"] = [n.plot_analyses for n in jou]
    except Exception as e:
        print(f"Error setting plot_analyses: {e}")
        raise

    try:
        tmp["vlm_feedback_summary"] = [
            textwrap.fill(
                (
                    str(n.vlm_feedback_summary)
                    if n.vlm_feedback_summary is not None
                    else ""
                ),
                width=80,
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting vlm_feedback_summary: {e}")
        raise

    try:
        tmp["exec_time"] = [n.exec_time for n in jou]
    except Exception as e:
        print(f"Error setting exec_time: {e}")
        raise

    try:
        tmp["exec_time_feedback"] = [
            textwrap.fill(
                str(n.exec_time_feedback) if n.exec_time_feedback is not None else "",
                width=80,
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting exec_time_feedback: {e}")
        raise

    try:
        tmp["datasets_successfully_tested"] = [
            n.datasets_successfully_tested for n in jou
        ]
    except Exception as e:
        print(f"Error setting datasets_successfully_tested: {e}")
        raise

    try:
        tmp["plot_code"] = [n.plot_code for n in jou]
    except Exception as e:
        print(f"Error setting plot_code: {e}")
        raise

    try:
        tmp["plot_plan"] = [n.plot_plan for n in jou]
    except Exception as e:
        print(f"Error setting plot_plan: {e}")
        raise

    try:
        tmp["ablation_name"] = [n.ablation_name for n in jou]
    except Exception as e:
        print(f"Error setting ablation_name: {e}")
        raise

    try:
        tmp["hyperparam_name"] = [n.hyperparam_name for n in jou]
    except Exception as e:
        print(f"Error setting hyperparam_name: {e}")
        raise

    try:
        tmp["is_seed_node"] = [n.is_seed_node for n in jou]
    except Exception as e:
        print(f"Error setting is_seed_node: {e}")
        raise

    try:
        tmp["is_seed_agg_node"] = [n.is_seed_agg_node for n in jou]
    except Exception as e:
        print(f"Error setting is_seed_agg_node: {e}")
        raise

    try:
        tmp["parse_metrics_plan"] = [
            textwrap.fill(
                str(n.parse_metrics_plan) if n.parse_metrics_plan is not None else "",
                width=80,
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting parse_metrics_plan: {e}")
        raise

    try:
        tmp["parse_metrics_code"] = [n.parse_metrics_code for n in jou]
    except Exception as e:
        print(f"Error setting parse_metrics_code: {e}")
        raise

    try:
        tmp["parse_term_out"] = [
            textwrap.fill(
                str(n.parse_term_out) if n.parse_term_out is not None else "", width=80
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting parse_term_out: {e}")
        raise

    try:
        tmp["parse_exc_type"] = [n.parse_exc_type for n in jou]
    except Exception as e:
        print(f"Error setting parse_exc_type: {e}")
        raise

    try:
        tmp["parse_exc_info"] = [n.parse_exc_info for n in jou]
    except Exception as e:
        print(f"Error setting parse_exc_info: {e}")
        raise

    try:
        tmp["parse_exc_stack"] = [n.parse_exc_stack for n in jou]
    except Exception as e:
        print(f"Error setting parse_exc_stack: {e}")
        raise

    # Add the list of completed stages by checking directories
    if out_path:
        log_dir = out_path.parent.parent
        tmp["completed_stages"] = get_completed_stages(log_dir)

    return tmp


def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace('"PLACEHOLDER_TREE_DATA"', tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: Journal, out_path: Path):
    print("[red]Checking Journal[/red]")
    try:
        tree_struct = cfg_to_tree_struct(cfg, jou, out_path)
    except Exception as e:
        print(f"Error in cfg_to_tree_struct: {e}")
        raise

    # Save tree data as JSON for loading by the tabbed visualization
    try:
        # Save the tree data as a JSON file in the same directory
        data_path = out_path.parent / "tree_data.json"
        with open(data_path, "w") as f:
            json.dump(tree_struct, f)
    except Exception as e:
        print(f"Error saving tree data JSON: {e}")

    try:
        tree_graph_str = json.dumps(tree_struct)
    except Exception as e:
        print(f"Error in json.dumps: {e}")
        raise
    try:
        html = generate_html(tree_graph_str)
    except Exception as e:
        print(f"Error in generate_html: {e}")
        raise
    with open(out_path, "w") as f:
        f.write(html)

    # Create a unified tree visualization that shows all stages
    try:
        create_unified_viz(cfg, out_path)
    except Exception as e:
        print(f"Error creating unified visualization: {e}")
        # Continue even if unified viz creation fails


def create_unified_viz(cfg, current_stage_viz_path):
    """
    Create a unified visualization that shows all completed stages in a tabbed interface.
    This will be placed in the main log directory.
    """
    # The main log directory is two levels up from the stage-specific visualization
    log_dir = current_stage_viz_path.parent.parent

    # Get the current stage name from the path
    current_stage = current_stage_viz_path.parent.name
    if current_stage.startswith("stage_"):
        # Extract the stage number from the directory name
        parts = current_stage.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            stage_num = parts[1]
            current_stage = f"Stage_{stage_num}"

    # Create a combined visualization at the top level
    unified_viz_path = log_dir / "unified_tree_viz.html"

    # Copy the template files
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.html") as f:
        html = f.read()

    with open(template_dir / "template.js") as f:
        js = f.read()

    # Get completed stages by checking directories
    completed_stages = get_completed_stages(log_dir)

    # Try to load the current stage's tree data to use as a basis
    try:
        current_stage_data_path = current_stage_viz_path.parent / "tree_data.json"
        if current_stage_data_path.exists():
            with open(current_stage_data_path, "r") as f:
                base_data = json.load(f)
                # Add the necessary metadata
                base_data["current_stage"] = current_stage
                base_data["completed_stages"] = completed_stages
        else:
            # If we can't load the tree data, create a minimal structure
            base_data = {
                "current_stage": current_stage,
                "completed_stages": completed_stages,
                # Add empty layout and edges to prevent errors
                "layout": [],
                "edges": [],
            }
    except Exception as e:
        print(f"Error loading stage data: {e}")
        # Create a minimal data structure that won't cause JS errors
        base_data = {
            "current_stage": current_stage,
            "completed_stages": completed_stages,
            "layout": [],
            "edges": [],
        }

    # Replace the placeholder in the JS with our data
    js = js.replace('"PLACEHOLDER_TREE_DATA"', json.dumps(base_data))

    # Replace the placeholder in the HTML with our JS
    html = html.replace("<!-- placeholder -->", js)

    # Write the unified visualization
    with open(unified_viz_path, "w") as f:
        f.write(html)

    print(f"[green]Created unified visualization at {unified_viz_path}[/green]")
