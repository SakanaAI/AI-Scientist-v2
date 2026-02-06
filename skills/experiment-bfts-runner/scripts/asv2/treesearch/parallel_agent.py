from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Set, Any, Callable, cast, Dict, Tuple
import random
import subprocess
import os
from queue import Queue
import logging
import humanize
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code
import copy
import pickle
from dataclasses import asdict
from omegaconf import OmegaConf

from rich import print
from pathlib import Path
import base64
import sys

logger = logging.getLogger("ai-scientist")

AI_SCIENTIST_ROOT = "AI_SCIENTIST_ROOT"
AI_SCIENTIST_SKILLS_ROOT = "AI_SCIENTIST_SKILLS_ROOT"

ExecCallbackType = Callable[[str, bool], ExecutionResult]


def _safe_pickle_test(obj, name="object"):
    """Test if an object can be pickled"""
    try:
        pickle.dumps(obj)
        return True
    except Exception as e:
        logger.error(f"Cannot pickle {name}: {str(e)}")
        return False


def _parse_keyword_prefix_response(
    response: str, keyword_prefix1: str, keyword_prefix2: str
) -> Tuple[Optional[str], Optional[str]]:
    """Parse the response into name and description based on keyword prefix"""
    try:
        # Split response into lines and clean up
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        # Find the idea and description
        name = None
        description = None

        for line in lines:
            if line.startswith(keyword_prefix1):
                name = line.replace(keyword_prefix1, "").strip()
            elif line.startswith(keyword_prefix2):
                description = line.replace(keyword_prefix2, "").strip()
                # Combine any following lines that don't start with a marker
                desc_lines = []
                for next_line in lines[lines.index(line) + 1 :]:
                    if not next_line.startswith((keyword_prefix1, keyword_prefix2)):
                        desc_lines.append(next_line)
                    else:
                        break
                if desc_lines:
                    description = " ".join([description] + desc_lines)

        if name is None or description is None:
            raise ValueError(
                f"Missing required keywords in response: {keyword_prefix1} and/or {keyword_prefix2}"
            )

        return name, description

    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        logger.debug(f"Raw response: {response}")
        return None, None


review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, summarize the bug and propose a fix. Otherwise, leave it empty.",
            },
        },
        "required": [
            "is_bug",
            "summary",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

vlm_feedback_spec = FunctionSpec(
    name="analyze_experiment_plots",
    json_schema={
        "type": "object",
        "properties": {
            "plot_analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "string",
                            "description": "Detailed analysis of the plot's results and implications",
                        },
                    },
                    "required": ["analysis"],
                },
            },
            "valid_plots_received": {
                "type": "boolean",
                "description": "True if valid plots were received, False otherwise. For example, if the plots are empty or not meaningful, this should be False.",
            },
            "vlm_feedback_summary": {
                "type": "string",
                "description": "Summarize the feedback from the VLM. If the task involves generative modeling, make sure to focus on the generated samples.",
            },
        },
        "required": ["plot_analyses", "valid_plots_received", "vlm_feedback_summary"],
    },
    description="Analyze experimental plots and provide detailed feedback on the results.",
)

metric_parse_spec = FunctionSpec(
    name="parse_metrics",
    json_schema={
        "type": "object",
        "strict": True,
        "properties": {
            "valid_metrics_received": {
                "type": "boolean",
                "description": "True if the metrics were successfully received, False otherwise. For example if the execution output does not contain any metrics, set this to False.",
            },
            "metric_names": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "Specify the metric name clearly. Avoid vague terms like 'train,' 'val,' or 'test.' Instead, use precise labels such as 'train accuracy,' 'validation loss,' or 'test F1 score,' etc.",
                        },
                        "lower_is_better": {
                            "type": "boolean",
                            "description": "Whether lower values are better for this metric",
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the metric",
                        },
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "dataset_name": {
                                        "type": "string",
                                        "description": "The name of the dataset. Never include 'train', 'val', or 'test' in the dataset name.",
                                    },
                                    "final_value": {
                                        "type": "number",
                                        "description": "The final value of the metric for this dataset",
                                    },
                                    "best_value": {
                                        "type": "number",
                                        "description": "The best value of the metric for this dataset",
                                    },
                                },
                                "required": [
                                    "dataset_name",
                                    "final_value",
                                    "best_value",
                                ],
                            },
                        },
                    },
                    "required": [
                        "data",
                        "metric_name",
                        "lower_is_better",
                        "description",
                    ],
                },
                "additionalProperties": False,
            },
        },
        "required": ["valid_metrics_received", "metric_names"],
        "additionalProperties": False,
    },
    description="Parse metrics from execution output",
)


plot_selection_spec = FunctionSpec(
    name="select_plots",
    json_schema={
        "type": "object",
        "properties": {
            "selected_plots": {
                "type": "array",
                "description": "List of selected plot file paths",
                "items": {"type": "string", "description": "Full path to a plot file"},
                "maxItems": 10,
            }
        },
        "required": ["selected_plots"],
    },
    description="Select the 10 most relevant plots for analysis",
)


class AblationConfig:
    """Track state of ablation experiments"""

    def __init__(self, name: str, description: str, code: str, base_node: Node):
        self.name = name
        self.description = description
        self.code = code
        self.base_node = base_node
        self.attempts = 0
        self.max_attempts = 3  # Maximum number of retry attempts
        self.last_error = None
        self.completed = False
        self.current_node = None


class AblationIdea:
    """Ablation idea"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class HyperparamTuningIdea:
    """Hyperparameter tuning idea"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class MinimalAgent:
    """A minimal agent class that only contains what's needed for processing nodes"""

    def __init__(
        self,
        task_desc,
        cfg,
        memory_summary=None,
        evaluation_metrics=None,
        stage=None,
        stage_name=None,
    ):
        self.task_desc = task_desc
        self.memory_summary = memory_summary
        self.cfg = cfg
        self.evaluation_metrics = evaluation_metrics
        self.stage_name = stage_name
        self.data_preview = None

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
            "albumentations",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "CRITICAL GPU REQUIREMENTS - Your code MUST include ALL of these:",
            "  - At the start of your code, add these lines to handle GPU/CPU:",
            "    ```python",
            "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "    print(f'Using device: {device}')",
            "    ```",
            "  - ALWAYS move models to device using the `.to(device)` method",
            "  - ALWAYS move input tensors to device using the `.to(device)` method",
            "  - ALWAYS move model related tensors to device using the `.to(device)` method",
            "  - For optimizers, create them AFTER moving model to device",
            "  - When using DataLoader, move batch tensors to device in training loop: `batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}`",
            "CRITICAL MODEL INPUT GUIDELINES:",
            "  - Always pay extra attention to the input to the model being properly normalized",
            "  - This is extremely important because the input to the model's forward pass directly affects the output, and the loss function is computed based on the output",
        ]
        if hasattr(self.cfg.experiment, "num_syn_datasets"):
            num_syn_datasets = self.cfg.experiment.num_syn_datasets
            if num_syn_datasets > 1:
                impl_guideline.extend(
                    [
                        f"You MUST evaluate your solution on at least {num_syn_datasets} different synthetic datasets to ensure robustness:",
                        "  - Use standard benchmark datasets when available",
                        f"  - If using synthetic data, generate at least {num_syn_datasets} variants with different characteristics",
                        "  - Report metrics separately for each dataset",
                        "  - Compute and report the average metric across all datasets",
                    ]
                )
        impl_guideline.extend(
            [
                "For generative modeling tasks, you must:",
                "  - Generate a set of samples from your model",
                "  - Compare these samples with ground truth data using appropriate visualizations",
                "  - When saving plots, always use the 'working_dir' variable that will be defined at the start of the script",
                "  - Make sure to give each figure a unique and appropriate name based on the dataset it represents, rather than reusing the same filename.",
                "Important code structure requirements:",
                "  - Do NOT put any execution code inside 'if __name__ == \"__main__\":' block",
                "  - All code should be at the global scope or in functions that are called from the global scope",
                "  - The script should execute immediately when run, without requiring any special entry point",
                "The code should start with:",
                "  import os",
                "  working_dir = os.path.join(os.getcwd(), 'working')",
                "  os.makedirs(working_dir, exist_ok=True)",
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Your response should only contain a single code block.",
                f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
                'You can also use the "./working" directory to store any temporary files that your code needs to create.',
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'dataset_name_1': {",
                "          'metrics': {'train': [], 'val': []},",
                "          'losses': {'train': [], 'val': []},",
                "          'predictions': [],",
                "          'ground_truth': [],",
                "          # Add other relevant data",
                "      },",
                "      # Add additional datasets as needed:",
                "      'dataset_name_2': {",
                "          'metrics': {'train': [], 'val': []},",
                "          'losses': {'train': [], 'val': []},",
                "          'predictions': [],",
                "          'ground_truth': [],",
                "          # Add other relevant data",
                "      },",
                "  }",
                "  # During training/evaluation:",
                "  experiment_data['dataset_name_1']['metrics']['train'].append(train_metric)",
                "  ```",
                "- Include timestamps or epochs with the saved metrics",
                "- For large datasets, consider saving in chunks or using np.savez_compressed()",
                "CRITICAL EVALUATION REQUIREMENTS - Your code MUST include ALL of these:",
                "  1. Track and print validation loss at each epoch or at suitable intervals:",
                "     ```python",
                "     print(f'Epoch {{epoch}}: validation_loss = {{val_loss:.4f}}')",
                "     ```",
                "  2. Track and update ALL these additional metrics: "
                + str(self.evaluation_metrics),
                "  3. Update metrics at EACH epoch:",
                "  4. Save ALL metrics at the end:",
                "     ```python",
                "     np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)",
                "     ```",
            ]
        )

        if self.cfg.agent.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.cfg.agent.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements this solution and prints out the evaluation metric(s) if applicable. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Make sure to write concise code."
            )
        }

    def _prompt_metricparse_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code for the metric parsing. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Your generated code should be complete and executable. "
            )
        }

    @property
    def _prompt_debug_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the bugfix/solution. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Your generated code should be complete and executable. Do not omit any part of the code, even if it was part of a previous implementation."
                "Make sure to write concise code."
            )
        }

    @property
    def _prompt_hyperparam_tuning_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including hyperparameter tuning. "
                "There should be no additional headings or text in your response. Do not omit any part of the code, "
                "Your generated code should be complete and executable."
                "Make sure to write concise code."
            )
        }

    @property
    def _prompt_ablation_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the ablation study. "
                "There should be no additional headings or text in your response. Do not omit any part of the code, "
                "Your generated code should be complete and executable."
                "Make sure to write concise code."
            )
        }

    def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are an AI researcher who is looking to publish a paper that will contribute significantly to the field."
                "Your first task is to write a python code to implement a solid baseline based on your research idea provided below, "
                "from data preparation to model training, as well as evaluation and visualization. "
                "Focus on getting a simple but working implementation first, before any sophisticated improvements. "
                "We will explore more advanced variations in later stages."
            ),
            "Research idea": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Experiment design sketch guideline": [
                "This first experiment design should be relatively simple, without extensive hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design. ",
                "The solution sketch should be 6-10 sentences. ",
                "Don't suggest to do EDA.",
                "Make sure to create synthetic data if needed.",
                "",
            ],
            "Evaluation Metric(s)": self.evaluation_metrics,
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.cfg.agent.data_preview:
            prompt["Data Overview"] = self.data_preview

        print("[cyan]--------------------------------[/cyan]")
        print("[cyan]self.task_desc[/cyan]")
        print("[cyan]" + self.task_desc + "[/cyan]")
        print("[cyan]--------------------------------[/cyan]")

        print("MinimalAgent: Getting plan and code")
        plan, code = self.plan_and_code_query(prompt)
        print("MinimalAgent: Draft complete")
        return Node(plan=plan, code=code)

    def _debug(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. Your previous code for research experiment had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Research idea": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Feedback based on generated plots": parent_node.vlm_feedback_summary,
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_debug_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.cfg.agent.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def _improve(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. You are provided with a previously developed "
                "implementation. Your task is to improve it based on the current experimental stage."
            ),
            "Research idea": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Feedback based on generated plots": parent_node.vlm_feedback_summary,
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _generate_seed_node(self, parent_node: Node):
        return Node(
            plan="Seed node",
            code=parent_node.code,
            parent=parent_node,
            is_seed_node=True,
        )

    def _generate_hyperparam_tuning_node(
        self, parent_node: Node, hyperparam_idea: HyperparamTuningIdea
    ):
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. You are provided with a previously developed "
                "baseline implementation. Your task is to implement hyperparameter tuning for the following idea: "
                + hyperparam_idea.name
                + ". "
                + hyperparam_idea.description
            ),
            "Base code you are working on": wrap_code(parent_node.code),
            "Instructions": {},
        }
        prompt["Instructions"] |= {
            "Implementation guideline": [
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'hyperparam_tuning_type_1': {",
                "          'dataset_name_1': {",
                "              'metrics': {'train': [], 'val': []},",
                "              'losses': {'train': [], 'val': []},",
                "              'predictions': [],",
                "              'ground_truth': [],",
                "              # Add other relevant data",
                "          },",
                "          # Add additional datasets as needed:",
                "      },",
                "      # Add additional hyperparam tuning types as needed",
                "  }",
                "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
            ]
        }
        prompt["Instructions"] |= self._prompt_hyperparam_tuning_resp_fmt
        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan="Hyperparam tuning name: " + hyperparam_idea.name + ".\n" + plan,
            code=code,
            parent=parent_node,
            hyperparam_name=hyperparam_idea.name,
        )

    def _generate_ablation_node(self, parent_node: Node, ablation_idea: AblationIdea):
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. You are provided with a previously developed "
                "baseline implementation. Your task is to implement the ablation study for the following idea: "
                + ablation_idea.name
                + ". "
                + ablation_idea.description
            ),
            "Base code you are working on": wrap_code(parent_node.code),
            "Instructions": {},
        }
        prompt["Instructions"] |= {
            "Implementation guideline": [
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'ablation_type_1': {",
                "          'dataset_name_1': {",
                "              'metrics': {'train': [], 'val': []},",
                "              'losses': {'train': [], 'val': []},",
                "              'predictions': [],",
                "              'ground_truth': [],",
                "              # Add other relevant data",
                "          },",
                "          # Add additional datasets as needed:",
                "          'dataset_name_2': {",
                "              'metrics': {'train': [], 'val': []},",
                "              'losses': {'train': [], 'val': []},",
                "              'predictions': [],",
                "              'ground_truth': [],",
                "              # Add other relevant data",
                "          },",
                "      },",
                "      # Add additional ablation types as needed",
                "  }",
                "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
            ]
        }
        prompt["Instructions"] |= self._prompt_ablation_resp_fmt
        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan="Ablation name: " + ablation_idea.name + ".\n" + plan,
            code=code,
            parent=parent_node,
            ablation_name=ablation_idea.name,
        )

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                "The code extraction failed. Make sure to use the format ```python ... ``` for the code blocks."
            )
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def parse_exec_result(
        self, node: Node, exec_result: ExecutionResult, workspace: str
    ):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        prompt = {
            "Introduction": (
                "You are an experienced AI researcher. "
                "You have written code for your research experiment and now need to evaluate the output of the code execution. "
                "Analyze the execution output, determine if there were any bugs, and provide a summary of the findings. "
            ),
            "Research idea": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            ),
        )

        node.analysis = response["summary"]
        node.is_buggy = response["is_bug"] or node.exc_type is not None
        print(
            "[red]Checking if response contains metric name and description[/red]",
            flush=True,
        )
        print(response)

    def _generate_plotting_code(
        self, node: Node, working_dir: str, plot_code_from_prev_stage: str = None
    ) -> str:
        """Generate code for plotting experiment results"""
        prompt_guideline = [
            "AVAILABLE DATA: ",
            "Experiment Data: experiment_data.npy",
        ]
        prompt_guideline += [
            "REQUIREMENTS: ",
            "The code should start with:",
            "  import matplotlib.pyplot as plt",
            "  import numpy as np",
            "  import os",
            "  working_dir = os.path.join(os.getcwd(), 'working')",
            "Create standard visualizations of experiment results",
            "Save all plots to working_dir",
            "Include training/validation curves if available",
            "ONLY plot data that exists in experiment_data.npy - DO NOT make up or simulate any values",
            "Use basic matplotlib without custom styles",
            "Each plot should be in a separate try-except block",
            "Always close figures after saving",
            "Always include a title for each plot, and be sure to use clear subtitles—such as 'Left: Ground Truth, Right: Generated Samples'—while also specifying the type of dataset being used.",
            "Make sure to use descriptive names for figures when saving e.g. always include the dataset name and the type of plot in the name",
            "When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.",
            "Use the following experiment code to infer the data to plot: " + node.code,
            "Example to extract data from experiment_data: experiment_data['dataset_name_1']['metrics']['train']",
        ]
        prompt_guideline += [
            "Example data loading and plot saving code: ",
            """
                try:
                    experiment_data = np.load(os.path.join(working_dir, 'experiment_data.npy'), allow_pickle=True).item()
                except Exception as e:
                    print(f'Error loading experiment data: {{e}}')

                try:
                    # First plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_1].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot1: {{e}}")
                    plt.close()  # Always close figure even if error occurs

                try:
                    # Second plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_2].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot2: {{e}}")
                    plt.close()
            """,
        ]
        # add instruction for format
        plotting_prompt = {
            "Instructions": {},
        }
        plotting_prompt["Instructions"] |= self._prompt_resp_fmt
        plotting_prompt["Instructions"] |= {
            "Plotting code guideline": prompt_guideline,
        }

        # For stage 3, initialize with stage 2's plotting code
        if (
            self.stage_name
            and self.stage_name.startswith("3_")
            and plot_code_from_prev_stage
        ):
            prompt_guideline.extend(
                [
                    "IMPORTANT: Use the following base plotting code as a starting point:",
                    "Base plotting code: " + plot_code_from_prev_stage,
                    "Modify the base plotting code to:",
                    "1. Keep the same numpy data structure and plotting style",
                    "2. Add comparison plots between different datasets",
                    "3. Add dataset-specific visualizations if needed",
                    "4. Include clear labels indicating which plots are from which dataset",
                    "5. Use consistent naming conventions for saved files",
                ]
            )
        # For stage 4, initialize with stage 3's plotting code
        elif (
            self.stage_name
            and self.stage_name.startswith("4_")
            and plot_code_from_prev_stage
        ):
            prompt_guideline.extend(
                [
                    "IMPORTANT: This is an ablation study. Use the following base plotting code as a starting point:",
                    "Base plotting code: \n" + plot_code_from_prev_stage,
                    "Modify the base plotting code to:",
                    "1. Keep the same numpy data structure and plotting style",
                    "2. Add comparison plots between ablation and baseline results",
                    "3. Add ablation-specific visualizations if needed",
                    "4. Include clear labels indicating which plots are from ablation vs baseline",
                    "5. Use consistent naming conventions for saved files",
                ]
            )

        # Get plotting code from LLM
        plan, code = self.plan_and_code_query(plotting_prompt)

        # Ensure the code starts with imports
        if not code.strip().startswith("import"):
            code = "import matplotlib.pyplot as plt\nimport numpy as np\n\n" + code

        node.plot_code = code
        node.plot_plan = plan

        return code

    def _determine_datasets_successfully_tested(self, node: Node) -> List[str]:
        """Determine which datasets are successfully tested based on VLM feedback"""
        plot_analyses = ""
        for i, plot_analysis in enumerate(node.plot_analyses):
            plot_analyses += f"plot {i+1}: {plot_analysis['analysis']}\n"

        determine_prompt = {
            "Introduction": "You are an AI researcher analyzing experiment results. Based on the plot analyses and feedback, determine which datasets are successfully tested. Return reasoning and the dataset names that are successfully executed, or an empty string if no datasets are successfully executed.",
            "Plot analyses": plot_analyses,
            "VLM feedback summary": node.vlm_feedback_summary,
            "Original plotting code": node.plot_code,
            "Response format": (
                "Your response should start with 'REASONING: <reasoning>' to think about the plot analysis and feedback in the first line."
                "In the second line, you should have a list of dataset names that are successfully executed, starting with 'SUCCESSFULLY_TESTED_DATASETS: <list_datasets_successfully_tested>', "
            ),
        }

        retry_count = 0
        retry_limit = 5
        while retry_count < retry_limit:
            response = query(
                system_message=determine_prompt,
                user_message=None,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            )

            (
                reasoning,
                datasets_successfully_tested_str,
            ) = _parse_keyword_prefix_response(
                response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
            )
            print(f"[green]Reasoning:[/green] {reasoning}")
            print(
                f"[green]Datasets successfully tested:[/green] {datasets_successfully_tested_str}"
            )
            if reasoning is not None and datasets_successfully_tested_str is not None:
                if datasets_successfully_tested_str == "":
                    return [""]
                # Split by comma and clean each dataset name
                datasets = [
                    ds.strip() for ds in datasets_successfully_tested_str.split(",")
                ]
                # Filter out empty strings and ensure all elements are strings
                datasets = [ds for ds in datasets if isinstance(ds, str) and ds]
                logger.info(f"Successfully parsed datasets: {datasets}")
                return datasets

            retry_count += 1
            logger.warning(
                f"Failed to parse successfully tested datasets response (attempt {retry_count}/{retry_limit})"
            )

        logger.error(
            f"Failed to parse successfully tested datasets response after {retry_limit} retries. Falling back to an empty list."
        )
        return [""]

    def _analyze_plots_with_vlm(self, node: Node) -> None:
        """Analyze experimental plots using VLM"""
        if not node.plot_paths:
            return

        # for debugging
        print(f"[cyan]Plot paths:[/cyan] {node.plot_paths}")

        def encode_image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                try:
                    return base64.b64encode(image_file.read()).decode("utf-8")
                except Exception as e:
                    print(f"[red]Error encoding image {image_path}: {e}[/red]")
                    return None

        if not len(node.plot_paths) > 10:
            selected_plots = node.plot_paths
        else:
            print(
                f"[red]Warning: {len(node.plot_paths)} plots received, this may be too many to analyze effectively. Calling LLM to select the most relevant plots to analyze.[/red]"
            )
            # select 10 plots to analyze
            prompt_select_plots = {
                "Introduction": (
                    "You are an experienced AI researcher analyzing experimental results. "
                    "You have been provided with plots from a machine learning experiment. "
                    "Please select 10 most relevant plots to analyze. "
                    "For similar plots (e.g. generated samples at each epoch), select only at most 5 plots at a suitable interval of epochs."
                    "Format your response as a list of plot paths, where each plot path includes the full path to the plot file."
                ),
                "Plot paths": node.plot_paths,
            }

            try:
                response_select_plots = cast(
                    dict,
                    query(
                        system_message=prompt_select_plots,
                        user_message=None,
                        func_spec=plot_selection_spec,
                        model=self.cfg.agent.feedback.model,
                        temperature=self.cfg.agent.feedback.temp,
                    ),
                )

                print(f"[cyan]Plot selection response:[/cyan] {response_select_plots}")
                # Extract the plot paths list
                selected_plots = response_select_plots.get("selected_plots", [])

                # Validate that all paths exist and are image files
                valid_plots = []
                for plot_path in selected_plots:
                    if (
                        isinstance(plot_path, str)
                        and os.path.exists(plot_path)
                        and plot_path.lower().endswith((".png", ".jpg", ".jpeg"))
                    ):
                        valid_plots.append(plot_path)
                    else:
                        logger.warning(f"Invalid plot path received: {plot_path}")

                # Use the validated list
                if valid_plots:
                    print(f"[cyan]Selected valid plots:[/cyan] {valid_plots}")
                    selected_plots = valid_plots
                else:
                    logger.warning(
                        "No valid plot paths found in response, falling back to first 10 plots"
                    )
                    # fallback to first 10 plots
                    # validate node.plot_paths
                    selected_plots = []
                    for plot_path in node.plot_paths[:10]:
                        if os.path.exists(plot_path) and plot_path.lower().endswith(
                            (".png", ".jpg", ".jpeg")
                        ):
                            selected_plots.append(plot_path)
                        else:
                            logger.warning(f"Invalid plot path received: {plot_path}")

            except Exception as e:
                logger.error(
                    f"Error in plot selection: {str(e)}; falling back to first 10 plots"
                )
                # Fallback to using first 10 plots
                selected_plots = node.plot_paths[:10]

        print("[cyan]Before encoding images[/cyan]")
        user_message = [
            {
                "type": "text",
                "text": (
                    "You are an experienced AI researcher analyzing experimental results. "
                    "You have been provided with plots from a machine learning experiment. "
                    f"This experiment is based on the following research idea: {self.task_desc}"
                    "Please analyze these plots and provide detailed insights about the results. "
                    "If you don't receive any plots, say 'No plots received'. "
                    "Never make up plot analysis. "
                    "Please return the analyzes with strict order of uploaded images, but DO NOT include any word "
                    "like 'the first plot'."
                ),
            }
        ] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(plot_path)}"
                },
            }
            for plot_path in selected_plots
        ]

        response = cast(
            dict,
            query(
                system_message=None,
                user_message=user_message,
                func_spec=vlm_feedback_spec,
                model=self.cfg.agent.vlm_feedback.model,
                temperature=self.cfg.agent.vlm_feedback.temp,
            ),
        )
        print(
            f"[cyan]VLM response from {self.cfg.agent.vlm_feedback.model}:[/cyan] {response}"
        )
        if response["valid_plots_received"]:
            node.is_buggy_plots = False
        else:
            node.is_buggy_plots = True

        for index, analysis in enumerate(response["plot_analyses"]):
            analysis["plot_path"] = node.plot_paths[index]

        node.plot_analyses = response["plot_analyses"]
        node.vlm_feedback_summary = response["vlm_feedback_summary"]

        node.datasets_successfully_tested = (
            self._determine_datasets_successfully_tested(node)
        )

    def _generate_node_summary(self, node: Node) -> dict:
        """Generate a summary of the node's experimental findings"""
        summary_prompt = {
            "Introduction": (
                "You are an AI researcher analyzing experimental results. "
                "Please summarize the findings from this experiment iteration."
            ),
            "Research idea": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Plan": node.plan,
            "Execution output": wrap_code(node.term_out, lang=""),
            "Analysis": node.analysis,
            "Metric": str(node.metric) if node.metric else "Failed",
            "Plot Analyses": (
                node.plot_analyses if hasattr(node, "plot_analyses") else []
            ),
            "VLM Feedback": (
                node.vlm_feedback_summary
                if hasattr(node, "vlm_feedback_summary")
                else ""
            ),
        }

        return cast(
            dict,
            query(
                system_message=summary_prompt,
                user_message=None,
                func_spec={
                    "name": "summarize_experiment",
                    "description": "Summarize experimental findings",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "findings": {
                                "type": "string",
                                "description": "Key findings and results",
                            },
                            "significance": {
                                "type": "string",
                                "description": "Why these results matter",
                            },
                            "next_steps": {
                                "type": "string",
                                "description": "Suggested improvements or next experiments",
                            },
                        },
                        "required": ["findings", "significance"],
                    },
                },
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            ),
        )


class GPUManager:
    """Manages GPU allocation across processes"""

    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.available_gpus: Set[int] = set(range(num_gpus))
        self.gpu_assignments: Dict[str, int] = {}  # process_id -> gpu_id

    def acquire_gpu(self, process_id: str) -> int:
        """Assigns a GPU to a process"""
        if not self.available_gpus:
            raise RuntimeError("No GPUs available")
        print(f"Available GPUs: {self.available_gpus}")
        print(f"Process ID: {process_id}")
        gpu_id = min(self.available_gpus)
        print(f"Acquiring GPU {gpu_id} for process {process_id}")
        self.available_gpus.remove(gpu_id)
        self.gpu_assignments[process_id] = gpu_id
        print(f"GPU assignments: {self.gpu_assignments}")
        return gpu_id

    def release_gpu(self, process_id: str):
        """Releases GPU assigned to a process"""
        if process_id in self.gpu_assignments:
            gpu_id = self.gpu_assignments[process_id]
            self.available_gpus.add(gpu_id)
            del self.gpu_assignments[process_id]


def get_gpu_count() -> int:
    """Get number of available NVIDIA GPUs without using torch"""
    try:
        # First try using nvidia-smi
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = nvidia_smi.stdout.strip().split("\n")
        return len(gpus)
    except (subprocess.SubprocessError, FileNotFoundError):
        # If nvidia-smi fails, try environment variable
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            # Filter out empty strings and -1 values
            devices = [d for d in cuda_visible_devices.split(",") if d and d != "-1"]
            return len(devices)
        return 0


class ParallelAgent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        stage_name=None,
        best_stage3_node=None,
        best_stage2_node=None,
        best_stage1_node=None,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.journal = journal
        self.stage_name = stage_name
        self.best_stage3_node = (
            best_stage3_node  # to initialize ablation stuides (stage 4)
        )
        self.best_stage1_node = (
            best_stage1_node  # to initialize hyperparam tuning (stage 2)
        )
        self.best_stage2_node = (
            best_stage2_node  # to initialize plotting code (stage 3)
        )
        self.data_preview = None
        self.num_workers = cfg.agent.num_workers
        self.num_gpus = get_gpu_count()
        print(f"num_gpus: {self.num_gpus}")
        if self.num_gpus == 0:
            print("No GPUs detected, falling back to CPU-only mode")
        else:
            print(f"Detected {self.num_gpus} GPUs")

        self.gpu_manager = GPUManager(self.num_gpus) if self.num_gpus > 0 else None

        if self.num_gpus > 0:
            self.num_workers = min(self.num_workers, self.num_gpus)
            logger.info(f"Limiting workers to {self.num_workers} to match GPU count")

        self.timeout = self.cfg.exec.timeout
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self._is_shutdown = False
        # Define the metric once at initialization
        self.evaluation_metrics = self._define_global_metrics()
        self._ablation_state = {  # store ablation names
            "completed_ablations": set(),
        }
        self._hyperparam_tuning_state = {  # store hyperparam tuning ideas
            "tried_hyperparams": set(),
        }

    def _define_global_metrics(self) -> str:
        """Define eval metric to be used across all experiments"""
        prompt = {
            "Introduction": (
                "You are an AI researcher setting up experiments. "
                "Please propose meaningful evaluation metrics that will help analyze "
                "the performance and characteristics of solutions for this research task."
            ),
            "Research idea": self.task_desc,
            "Instructions": [
                "Propose a single evaluation metric that would be useful for analyzing the performance of solutions for this research task.",
                "Note: Validation loss will be tracked separately so you don't need to include it in your response.",
                "Format your response as a list containing:",
                "- name: The name of the metric",
                "- maximize: Whether higher values are better (true/false)",
                "- description: A brief explanation of what the metric measures"
                "Your list should contain only one metric.",
            ],
        }

        response = query(
            system_message=prompt,
            user_message=None,
            model=self.cfg.agent.code.model,
            temperature=self.cfg.agent.code.temp,
        )

        print(f"[green]Defined eval metrics:[/green] {response}")
        return response

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code
            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                "The code extraction failed. Make sure to use the format ```python ... ``` for the code blocks."
            )
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text

    def _generate_seed_eval_aggregation_node(
        self, node: Node, agg_plotting_code: str
    ) -> Node:
        """Generate a special aggregation node for seed evaluation results"""
        return Node(
            plan="Aggregate results from multiple seeds",
            code="# plotting aggregation code",
            plot_code=agg_plotting_code,
            parent=node,
            is_seed_node=True,
            is_seed_agg_node=True,
        )

    def _run_multi_seed_evaluation(self, node: Node) -> List[Node]:
        """Run multiple seeds of the same node to get statistical metrics.
        Returns a list of nodes with different random seeds."""

        # Convert node to dict for parallel processing
        node_data = node.to_dict()
        node_code = node.code

        # Submit parallel jobs for different seeds
        seed_nodes = []
        futures = []
        for seed in range(self.cfg.agent.multi_seed_eval.num_seeds):
            gpu_id = None
            if self.gpu_manager is not None:
                try:
                    process_id = f"seed_{seed}_worker"
                    gpu_id = self.gpu_manager.acquire_gpu(process_id)
                    logger.info(f"Assigned GPU {gpu_id} to seed {seed}")
                except RuntimeError as e:
                    logger.warning(
                        f"Could not acquire GPU for seed {seed}: {e}. Running on CPU"
                    )

            # Add seed to node code
            node_data["code"] = (
                f"# Set random seed\nimport random\nimport numpy as np\nimport torch\n\nseed = {seed}\nrandom.seed(seed)\nnp.random.seed(seed)\ntorch.manual_seed(seed)\nif torch.cuda.is_available():\n    torch.cuda.manual_seed(seed)\n\n"
                + node_code
            )

            new_ablation_idea = None
            new_hyperparam_idea = None
            best_stage1_plot_code = None
            best_stage2_plot_code = None
            best_stage3_plot_code = None
            seed_eval = True
            memory_summary = ""
            print("[yellow]Starting multi-seed eval...[/yellow]")
            futures.append(
                self.executor.submit(
                    self._process_node_wrapper,
                    node_data,
                    self.task_desc,
                    self.cfg,
                    gpu_id,
                    memory_summary,
                    self.evaluation_metrics,
                    self.stage_name,
                    new_ablation_idea,
                    new_hyperparam_idea,
                    best_stage1_plot_code,
                    best_stage2_plot_code,
                    best_stage3_plot_code,
                    seed_eval,
                )
            )

        for future in futures:
            try:
                result_data = future.result(timeout=self.timeout)
                result_node = Node.from_dict(result_data, self.journal)
                print(f"Parent node id: {result_node.parent.id}")
                print(f"Sanity check: actual parent node id: {node.id}")
                # Add node to journal's list and assign its step number
                self.journal.append(result_node)
                seed_nodes.append(self.journal.get_node_by_id(result_node.id))
                print("Added result node to journal")
            except Exception as e:
                logger.error(f"Error in multi-seed evaluation: {str(e)}")

        return seed_nodes

    def _run_plot_aggregation(self, node: Node, seed_nodes: List[Node]) -> Node:
        """Generate an aggregation node for seed evaluation results"""
        if seed_nodes:
            try:
                from .interpreter import Interpreter

                # Create aggregation plotting code
                agg_plotting_code = self._aggregate_seed_eval_results(seed_nodes, node)

                # Create a special aggregation node
                agg_node = self._generate_seed_eval_aggregation_node(
                    node, agg_plotting_code
                )
                agg_node.parent = node

                # Execute aggregation plotting code
                print("[blue]Creating Interpreter for seed node aggregation[/blue]")
                process_interpreter = Interpreter(
                    working_dir=self.cfg.workspace_dir,
                    timeout=self.cfg.exec.timeout,
                    format_tb_ipython=self.cfg.exec.format_tb_ipython,
                    agent_file_name=self.cfg.exec.agent_file_name,
                    env_vars={AI_SCIENTIST_ROOT: os.getenv(AI_SCIENTIST_ROOT), AI_SCIENTIST_SKILLS_ROOT: os.getenv(AI_SCIENTIST_SKILLS_ROOT)},
                )

                try:
                    working_dir = process_interpreter.working_dir
                    plot_exec_result = process_interpreter.run(agg_plotting_code, True)
                    print(plot_exec_result)
                    process_interpreter.cleanup_session()
                    # Save aggregated plots
                    plots_dir = Path(working_dir) / "working"
                    print("[red]plots_dir[/red]", plots_dir)
                    if plots_dir.exists():
                        base_dir = Path(self.cfg.workspace_dir).parent  # .parent
                        run_name = Path(self.cfg.workspace_dir).name
                        exp_results_dir = (
                            base_dir
                            / "logs"
                            / run_name
                            / "experiment_results"
                            / f"seed_aggregation_{agg_node.id}"
                        )
                        print("[red]exp_results_dir[/red]", exp_results_dir)
                        exp_results_dir.mkdir(parents=True, exist_ok=True)

                        # Save plotting code
                        with open(
                            exp_results_dir / "aggregation_plotting_code.py", "w"
                        ) as f:
                            f.write(agg_plotting_code)

                        # Move generated plots
                        for plot_file in plots_dir.glob("*.png"):
                            final_path = exp_results_dir / plot_file.name
                            print("mv_from:plot_file.resolve(): ", plot_file.resolve())
                            print("mv_to:final_path: ", final_path)
                            plot_file.resolve().rename(final_path)
                            web_path = f"../../logs/{Path(self.cfg.workspace_dir).name}/experiment_results/seed_aggregation_{agg_node.id}/{plot_file.name}"
                            agg_node.plots.append(web_path)
                            agg_node.plot_paths.append(str(final_path.absolute()))

                    agg_node.is_buggy = False
                    agg_node.exp_results_dir = exp_results_dir
                    agg_node_dict = agg_node.to_dict()
                    agg_node_new = Node.from_dict(
                        agg_node_dict, self.journal
                    )  # to update the parent-child relationship in the journal
                    # Add aggregation node to journal
                    self.journal.append(agg_node_new)
                finally:
                    if process_interpreter:
                        process_interpreter.cleanup_session()

            except Exception as e:
                print(f"Error in seed result aggregation: {str(e)}")

    @staticmethod
    def _process_node_wrapper(
        node_data,
        task_desc,
        cfg,
        gpu_id: int = None,
        memory_summary: str = None,
        evaluation_metrics=None,
        stage_name=None,
        new_ablation_idea=None,
        new_hyperparam_idea=None,
        best_stage3_plot_code=None,
        best_stage2_plot_code=None,
        best_stage1_plot_code=None,
        seed_eval=False,
    ):
        """Wrapper function that creates a fresh environment for each process"""
        from .interpreter import Interpreter
        from .journal import Node, Journal
        from copy import deepcopy
        import os
        import multiprocessing

        print("Starting _process_node_wrapper")

        # Create process-specific workspace
        process_id = multiprocessing.current_process().name
        workspace = os.path.join(cfg.workspace_dir, f"process_{process_id}")
        os.makedirs(workspace, exist_ok=True)
        print(f"Process {process_id} using workspace: {workspace}")
        # Create process-specific working directory
        working_dir = os.path.join(workspace, "working")
        os.makedirs(working_dir, exist_ok=True)

        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Process {process_id} assigned to GPU {gpu_id}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info(f"Process {process_id} running on CPU")

        # Create minimal agent for worker process with the global metric definition
        worker_agent = MinimalAgent(
            task_desc=task_desc,
            cfg=cfg,
            memory_summary=memory_summary,
            evaluation_metrics=evaluation_metrics,
            stage_name=stage_name,
        )

        # Create interpreter instance for worker process
        print("Creating Interpreter")
        process_interpreter = Interpreter(
            working_dir=workspace,
            timeout=cfg.exec.timeout,
            format_tb_ipython=cfg.exec.format_tb_ipython,
            agent_file_name=cfg.exec.agent_file_name,
        )

        try:
            print(f"stage_name: {stage_name}")
            # Recreate node object from node_data, which becomes a parent node.
            if node_data:
                parent_node = Node.from_dict(node_data, journal=None)
                print(f"Recreated parent node: {parent_node.id}")
            else:
                parent_node = None
                print("No parent node to recreate")

            # Process the node using worker agent
            print("Starting node processing")
            if seed_eval:
                # Use the parent node's code to run the same code again
                child_node = worker_agent._generate_seed_node(parent_node)
                child_node.parent = parent_node
                # Plot code should also be the same as the parent node
                child_node.plot_code = parent_node.plot_code
            else:
                if parent_node is None:
                    print("Drafting new node")
                    child_node = worker_agent._draft()
                elif parent_node.is_buggy:
                    print("Debugging node with id: ", parent_node.id)
                    child_node = worker_agent._debug(parent_node)
                    child_node.parent = parent_node
                else:
                    if (
                        new_hyperparam_idea is not None and new_ablation_idea is None
                    ):  # stage 2
                        child_node = worker_agent._generate_hyperparam_tuning_node(
                            parent_node, new_hyperparam_idea
                        )
                        child_node.parent = parent_node
                        logger.info(
                            f"Processing hyperparam tuning: {child_node.hyperparam_name}"
                        )
                        print(
                            f"[cyan]Running hyperparam tuning: {child_node.hyperparam_name}[/cyan]"
                        )
                    elif (
                        new_ablation_idea is not None and new_hyperparam_idea is None
                    ):  # stage 4
                        child_node = worker_agent._generate_ablation_node(
                            parent_node, new_ablation_idea
                        )
                        child_node.parent = parent_node
                        logger.info(f"Processing ablation: {child_node.ablation_name}")
                        print(
                            f"[cyan]Running ablation study: {child_node.ablation_name}[/cyan]"
                        )
                    else:
                        print("Improving node with id: ", parent_node.id)
                        child_node = worker_agent._improve(parent_node)
                        child_node.parent = parent_node

            # Execute and parse results
            print("Running code")
            exec_result = process_interpreter.run(child_node.code, True)
            process_interpreter.cleanup_session()

            print("Parsing execution results")
            worker_agent.parse_exec_result(
                node=child_node, exec_result=exec_result, workspace=working_dir
            )

            # Add check for saved data files
            data_files = [f for f in os.listdir(working_dir) if f.endswith(".npy")]
            if not data_files:
                logger.warning(
                    "No .npy files found in working directory. Data may not have been saved properly."
                )
            else:
                if seed_eval:
                    # Use the parent node's parse code to parse the same data files again
                    parse_metrics_code = parent_node.parse_metrics_code
                    parse_metrics_plan = parent_node.parse_metrics_plan
                    print(
                        f"[blue]SEED EVAL: Parse metrics plan:[/blue] {parse_metrics_plan}"
                    )
                    print(
                        f"[blue]SEED EVAL: Parse metrics code:[/blue] {parse_metrics_code}"
                    )
                    child_node.parse_metrics_code = parse_metrics_code
                    child_node.parse_metrics_plan = parse_metrics_plan
                else:
                    # Call LLM to parse data files and extract metrics
                    parse_metrics_prompt = {
                        "Introduction": (
                            "You are an AI researcher analyzing experimental results stored in numpy files. "
                            "Write code to load and analyze the metrics from experiment_data.npy."
                        ),
                        "Context": [
                            "Original Code: " + child_node.code,
                        ],
                        "Instructions": [
                            "0. Make sure to get the working directory from os.path.join(os.getcwd(), 'working')",
                            "1. Load the experiment_data.npy file, which is located in the working directory",
                            "2. Extract metrics for each dataset. Make sure to refer to the original code to understand the structure of the data.",
                            "3. Always print the name of the dataset before printing the metrics",
                            "4. Always print the name of the metric before printing the value by specifying the metric name clearly. Avoid vague terms like 'train,' 'val,' or 'test.' Instead, use precise labels such as 'train accuracy,' 'validation loss,' or 'test F1 score,' etc.",
                            "5. You only need to print the best or final value for each metric for each dataset",
                            "6. DO NOT CREATE ANY PLOTS",
                            "Important code structure requirements:",
                            "  - Do NOT put any execution code inside 'if __name__ == \"__main__\":' block. Do not use 'if __name__ == \"__main__\":' at all.",
                            "  - All code should be at the global scope or in functions that are called from the global scope",
                            "  - The script should execute immediately when run, without requiring any special entry point",
                        ],
                        "Example data loading code": [
                            """
                            import matplotlib.pyplot as plt
import numpy as np

                            experiment_data = np.load(os.path.join(os.getcwd(), 'experiment_data.npy'), allow_pickle=True).item()
                            """
                        ],
                        "Response format": worker_agent._prompt_metricparse_resp_fmt(),
                    }

                    (
                        parse_metrics_plan,
                        parse_metrics_code,
                    ) = worker_agent.plan_and_code_query(parse_metrics_prompt)
                    print(f"[blue]Parse metrics plan:[/blue] {parse_metrics_plan}")
                    print(f"[blue]Parse metrics code:[/blue] {parse_metrics_code}")
                    child_node.parse_metrics_plan = parse_metrics_plan
                    child_node.parse_metrics_code = parse_metrics_code
                try:
                    # Execute the parsing code
                    metrics_exec_result = process_interpreter.run(
                        parse_metrics_code, True
                    )
                    process_interpreter.cleanup_session()
                    child_node.parse_term_out = metrics_exec_result.term_out
                    child_node.parse_exc_type = metrics_exec_result.exc_type
                    child_node.parse_exc_info = metrics_exec_result.exc_info
                    child_node.parse_exc_stack = metrics_exec_result.exc_stack

                    if metrics_exec_result.exc_type is None:
                        # Extract metrics from the execution output
                        metrics_prompt = {
                            "Introduction": "Parse the metrics from the execution output. You only need the final or best value of a metric for each dataset, not the entire list during training.",
                            "Execution Output": metrics_exec_result.term_out,
                        }
                        print(
                            f"[blue]Metrics_exec_result.term_out: {metrics_exec_result.term_out}[/blue]"
                        )
                        print(
                            f"[blue]Metrics Parsing Execution Result:\n[/blue] {metrics_exec_result}"
                        )

                        metrics_response = cast(
                            dict,
                            query(
                                system_message=metrics_prompt,
                                user_message=None,
                                func_spec=metric_parse_spec,
                                model=cfg.agent.feedback.model,
                                temperature=cfg.agent.feedback.temp,
                            ),
                        )
                        # If there is any None value, child_node.metric should be set to WorstMetricValue.
                        # This is achieved by raising an error in the MetricValue class,
                        # which sets child_node.is_buggy to True, thereby
                        # causing child_node.metric to be assigned WorstMetricValue.
                        print(f"[blue]Metrics:[/blue] {metrics_response}")
                        if metrics_response["valid_metrics_received"]:
                            child_node.metric = MetricValue(
                                value={"metric_names": metrics_response["metric_names"]}
                            )
                            logger.info(
                                f"Successfully extracted metrics for node {child_node.id}"
                            )
                        else:
                            child_node.metric = WorstMetricValue()
                            child_node.is_buggy = True
                            logger.error(
                                f"No valid metrics received for node {child_node.id}"
                            )
                    else:
                        logger.error(
                            f"Error executing metrics parsing code: {metrics_exec_result.exc_info}"
                        )
                        child_node.metric = WorstMetricValue()
                        child_node.is_buggy = True

                except Exception as e:
                    logger.error(
                        f"Error parsing metrics for node {child_node.id}: {str(e)}"
                    )
                    child_node.metric = WorstMetricValue()
                    child_node.is_buggy = True
                    child_node.parse_exc_type = str(e)
                    child_node.parse_exc_info = None
                    child_node.parse_exc_stack = None
                    child_node.parse_term_out = (
                        "Error parsing metrics. There was an error in the parsing code: "
                        + str(e)
                    )

            # if experiment was successful, generate and run plotting code
            if not child_node.is_buggy:
                try:
                    retry_count = 0
                    while True:
                        if seed_eval:
                            # Use the parent node's plotting code instead of generating new one
                            plotting_code = parent_node.plot_code
                        else:
                            if (
                                worker_agent.stage_name
                                and worker_agent.stage_name.startswith("3_")
                                and best_stage2_plot_code
                            ):
                                plot_code_from_prev_stage = best_stage2_plot_code
                            elif (
                                worker_agent.stage_name
                                and worker_agent.stage_name.startswith("4_")
                                and best_stage3_plot_code
                            ):
                                plot_code_from_prev_stage = best_stage3_plot_code
                            else:
                                plot_code_from_prev_stage = None

                            plotting_code = worker_agent._generate_plotting_code(
                                child_node, working_dir, plot_code_from_prev_stage
                            )
                        plot_exec_result = process_interpreter.run(plotting_code, True)
                        process_interpreter.cleanup_session()
                        child_node.plot_exec_result = plot_exec_result
                        if child_node.plot_exc_type and retry_count < 3:
                            print(
                                f"[red]Plotting code failed with exception: {child_node.plot_exc_type}[/red]"
                            )
                            print(
                                f"[red]Plotting code term out:[/red] {child_node.plot_term_out}"
                            )
                            print(
                                f"[red]Plotting code code:[/red] {child_node.plot_code}"
                            )
                            retry_count += 1
                            continue
                        else:
                            break

                    print("[blue]Plotting result:[/blue] ", plot_exec_result)
                    # Track generated plots
                    plots_dir = Path(working_dir)
                    if plots_dir.exists():
                        print("Plots directory exists, saving plots to node")
                        # Save the plotting code first
                        base_dir = Path(cfg.workspace_dir).parent
                        run_name = Path(cfg.workspace_dir).name
                        exp_results_dir = (
                            base_dir
                            / "logs"
                            / run_name
                            / "experiment_results"
                            / f"experiment_{child_node.id}_proc_{os.getpid()}"
                        )
                        child_node.exp_results_dir = exp_results_dir
                        exp_results_dir.mkdir(parents=True, exist_ok=True)
                        plot_code_path = exp_results_dir / "plotting_code.py"
                        with open(plot_code_path, "w") as f:
                            f.write(plotting_code)
                        logger.info(f"Saved plotting code to {plot_code_path}")
                        # Save experiment code to experiment_results directory
                        exp_code_path = exp_results_dir / "experiment_code.py"
                        with open(exp_code_path, "w") as f:
                            f.write(child_node.code)
                        logger.info(f"Saved experiment code to {exp_code_path}")
                        # Move experiment data files to experiment_results directory
                        for exp_data_file in plots_dir.glob("*.npy"):
                            exp_data_path = exp_results_dir / exp_data_file.name
                            exp_data_file.resolve().rename(exp_data_path)
                            logger.info(f"Saved experiment data to {exp_data_path}")

                        for plot_file in plots_dir.glob("*.png"):
                            # Get the base directory (parent of workspaces/logs)
                            base_dir = Path(cfg.workspace_dir).parent.parent
                            run_name = Path(cfg.workspace_dir).name

                            # Create the final path in logs directory
                            final_path = exp_results_dir / plot_file.name
                            plot_file.resolve().rename(final_path)

                            # Create a web-friendly relative path starting from logs directory
                            web_path = f"../../logs/{Path(cfg.workspace_dir).name}/experiment_results/experiment_{child_node.id}_proc_{os.getpid()}/{plot_file.name}"

                            child_node.plots.append(web_path)  # For visualization
                            child_node.plot_paths.append(
                                str(final_path.absolute())
                            )  # For programmatic access

                            logger.info(
                                f"[green]Generated plot: {plot_file.stem}[/green]"
                            )
                            logger.debug(f"Plot absolute path: {final_path.absolute()}")
                            logger.debug(f"Plot web path: {web_path}")
                except Exception as e:
                    logger.error(
                        f"Error generating plots for node {child_node.id}: {str(e)}"
                    )

                if child_node.plots:
                    try:
                        worker_agent._analyze_plots_with_vlm(child_node)
                        logger.info(
                            f"Generated VLM analysis for plots in node {child_node.id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error analyzing plots for node {child_node.id}: {str(e)}"
                        )

            # Convert result node to dict
            print("Converting result to dict")
            result_data = child_node.to_dict()
            print(f"Result data keys: {result_data.keys()}")
            print(f"Result data size: {len(str(result_data))} chars")
            print("Returning result")
            return result_data

        except Exception as e:
            print(f"Worker process error: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def _generate_hyperparam_tuning_idea(self) -> Optional[HyperparamTuningIdea]:
        """Generate the next hyperparam tuning idea based on what's been done.
        This is minaly for Stage 2 (baseline tuning).
        """
        tried = list(self._hyperparam_tuning_state["tried_hyperparams"])

        hyperparam_tuning_prompt = {
            "Introduction": (
                "You are an AI researcher conducting hyperparameter tuning for baseline experiments. "
                "Based on the current implementation and previous hyperparameter tuning attempts (if any), "
                "propose ONE new hyperparameter tuning idea to see if it improves the performance."
                "You should first check if simply training longer (more epochs) improves the performance."
                "Then try tuning common hyperparameters such as learning rate, batch size, etc."
                "Only propose algorithm-specific and/or model-specific hyperparameters after you have tried the above."
            ),
            "Base code you are working on": wrap_code(self.best_stage1_node.code),
            "Previous Hyperparam Tuning Attempts": {
                "Has been tried": tried if tried else "Nothing has been tried yet.",
            },
            "Instructions": {
                "Requirements": [
                    "1. Identify ONE specific hyperparameter to tune",
                    "2. Ensure the hyperparameter is different from previous attempts",
                ]
            },
            "Response format": (
                "Your response should start with 'HYPERPARAM NAME: <hyperparam name>' on the first line to represent the name of the hyperparameter."
                "The second line should start with 'DESCRIPTION: <description>', a brief description of what hyperparameter is being tuned and why (3-5 sentences). "
            ),
        }

        retry_count = 0
        retry_limit = 5
        while retry_count < retry_limit:
            response = query(
                system_message=hyperparam_tuning_prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            # Parse the response
            hyperparam_name, hyperparam_description = _parse_keyword_prefix_response(
                response, "HYPERPARAM NAME:", "DESCRIPTION:"
            )
            if hyperparam_name and hyperparam_description:
                return HyperparamTuningIdea(
                    name=hyperparam_name, description=hyperparam_description
                )

            retry_count += 1
            logger.warning(
                f"Failed to parse hyperparam tuning response (attempt {retry_count}/{retry_limit})"
            )

        logger.error(
            f"Failed to parse hyperparam tuning response after {retry_limit} retries. Falling back to default idea of increasing learning rate."
        )
        return HyperparamTuningIdea(
            name="increase learning rate", description="increase learning rate"
        )

    def _generate_ablation_idea(self) -> Optional[AblationIdea]:
        """Generate the next ablation idea based on what's been done"""

        # Prepare context of what's been tried
        completed = list(self._ablation_state["completed_ablations"])

        ablation_prompt = {
            "Introduction": (
                "You are an AI researcher conducting ablation studies. "
                "Based on the current implementation and previous ablations (if any), "
                "propose ONE new ablation study that tests a different aspect of the model."
            ),
            "Base code you are working on": wrap_code(self.best_stage3_node.code),
            "Previous Ablations": {
                "Has been tried": (
                    completed if completed else "Nothing has been tried yet."
                ),
            },
            "Instructions": {
                "Requirements": [
                    "1. Identify ONE specific component/feature to ablate",
                    "2. Ensure the ablation is different from previous completed or running attempts",
                    "3. The ablation should be a new idea, not a variation of previous ideas",
                    "4. If you have only used a single synthetic dataset throughout the experiment, one of your ablations should be to use multiple synthetic datasets (at least 3 different datasets)",
                ]
            },
            "Response format": (
                "Your response should start with 'ABLATION NAME: <ablation name>' on the first line to represent the name of the ablation."
                "The second line should start with 'ABLATION DESCRIPTION: <description>', a brief description of what component is being ablated and why (3-5 sentences), "
            ),
        }

        retry_count = 0
        retry_limit = 5
        while retry_count < retry_limit:
            response = query(
                system_message=ablation_prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            # Parse the response
            ablation_name, ablation_description = _parse_keyword_prefix_response(
                response, "ABLATION NAME:", "ABLATION DESCRIPTION:"
            )
            if ablation_name and ablation_description:
                return AblationIdea(
                    name=ablation_name, description=ablation_description
                )

            retry_count += 1
            logger.warning(
                f"Failed to parse ablation response (attempt {retry_count}/{retry_limit})"
            )

        logger.error(
            f"Failed to parse ablation response after {retry_limit} retries. Falling back to default idea of removing dropout."
        )
        return AblationIdea(name="add one more layer", description="add one more layer")

    def _get_leaves(self, node: Node) -> List[Node]:
        """Get all leaf nodes in the subtree rooted at node."""
        if not node.children:
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaves(child))
        return leaves

    def _select_parallel_nodes(self) -> List[Optional[Node]]:
        """Select N nodes to process in parallel,
        balancing between tree exploration and exploitation.
        Note:
        - This function runs in the main process.
        Some design considerations:
        - For Stage 2 and 4, we generate nodes in the main process and
        send them to worker processes.
        This is to make sure we don't run duplicate ideas in parallel.
        - For Stage 1 and 3, we generate nodes in worker processes.
        """
        nodes_to_process = []
        processed_trees = set()
        search_cfg = self.cfg.agent.search
        print(f"[cyan]self.num_workers: {self.num_workers}, [/cyan]")

        while len(nodes_to_process) < self.num_workers:
            # Initial drafting phase, creating root nodes
            print(
                f"Checking draft nodes... num of journal.draft_nodes: {len(self.journal.draft_nodes)}, search_cfg.num_drafts: {search_cfg.num_drafts}"
            )
            if len(self.journal.draft_nodes) < search_cfg.num_drafts:
                nodes_to_process.append(None)
                continue

            # Get viable trees
            viable_trees = [
                root
                for root in self.journal.draft_nodes
                if not all(leaf.is_buggy for leaf in self._get_leaves(root))
            ]

            # Debugging phase (with some probability)
            if random.random() < search_cfg.debug_prob:
                print("Checking debuggable nodes")
                # print(f"Buggy nodes: {self.journal.buggy_nodes}")
                try:
                    debuggable_nodes = None
                    print("Checking buggy nodes...")
                    buggy_nodes = self.journal.buggy_nodes
                    print(f"Type of buggy_nodes: {type(buggy_nodes)}")
                    print(f"Length of buggy_nodes: {len(buggy_nodes)}")

                    for i, n in enumerate(buggy_nodes):
                        if not isinstance(n, Node):
                            print(f"Found non-Node object in journal.buggy_nodes: {n}")
                            raise ValueError(
                                "Found non-Node object in journal.buggy_nodes"
                            )
                    debuggable_nodes = [
                        n
                        for n in self.journal.buggy_nodes
                        if (
                            isinstance(n, Node)
                            and n.is_leaf
                            and n.debug_depth <= search_cfg.max_debug_depth
                        )
                    ]
                except Exception as e:
                    print(f"Error getting debuggable nodes: {e}")
                if debuggable_nodes:
                    print("Found debuggable nodes")
                    node = random.choice(debuggable_nodes)
                    tree_root = node
                    while tree_root.parent:
                        tree_root = tree_root.parent

                    tree_id = id(tree_root)
                    if tree_id not in processed_trees or len(processed_trees) >= len(
                        viable_trees
                    ):
                        nodes_to_process.append(node)
                        processed_trees.add(tree_id)
                        continue

            # Special handling for Stage 4 (Ablation Studies)
            print(f"[red]self.stage_name: {self.stage_name}[/red]")
            # print(f"[red]self.best_stage3_node: {self.best_stage3_node}[/red]")
            if self.stage_name and self.stage_name.startswith("4_"):
                nodes_to_process.append(self.best_stage3_node)
                continue
            # Special handling for Stage 2 (Hyperparam tuning for baseline)
            elif self.stage_name and self.stage_name.startswith("2_"):
                nodes_to_process.append(self.best_stage1_node)
                continue
            else:  # Stage 1, 3 (normal best-first search)
                # Improvement phase
                print("Checking good nodes..")
                good_nodes = self.journal.good_nodes
                if not good_nodes:
                    nodes_to_process.append(None)  # Back to drafting
                    continue

                # Get best node from unprocessed tree if possible
                best_node = self.journal.get_best_node(cfg=self.cfg)
                tree_root = best_node
                while tree_root.parent:
                    tree_root = tree_root.parent

                tree_id = id(tree_root)
                if tree_id not in processed_trees or len(processed_trees) >= len(
                    viable_trees
                ):
                    nodes_to_process.append(best_node)
                    processed_trees.add(tree_id)
                    continue

                # If we can't use best node (tree already processed), try next best nodes
                for node in sorted(good_nodes, key=lambda n: n.metric, reverse=True):
                    tree_root = node
                    while tree_root.parent:
                        tree_root = tree_root.parent
                    tree_id = id(tree_root)
                    if tree_id not in processed_trees or len(processed_trees) >= len(
                        viable_trees
                    ):
                        nodes_to_process.append(node)
                        processed_trees.add(tree_id)
                        break

        return nodes_to_process

    def step(self, exec_callback: ExecCallbackType):
        print("Selecting nodes to process")
        nodes_to_process = self._select_parallel_nodes()
        print(f"Selected nodes: {[n.id if n else None for n in nodes_to_process]}")

        # Convert nodes to dicts
        node_data_list = []
        for node in nodes_to_process:
            if node:
                try:
                    node_data = node.to_dict()
                    _safe_pickle_test(node_data, f"node {node.id} data")
                    node_data_list.append(node_data)
                except Exception as e:
                    logger.error(f"Error preparing node {node.id}: {str(e)}")
                    raise
            else:
                node_data_list.append(None)  # None means new draft

        if self.cfg.agent.get("summary", None) is not None:
            memory_summary = self.journal.generate_summary(
                include_code=False, 
                **{
                    "model": self.cfg.agent.summary.model, 
                    "temp": self.cfg.agent.summary.temp
                }
            )
        else:
            memory_summary = self.journal.generate_summary(include_code=False)

        print("Submitting tasks to process pool")
        futures = []
        for node_data in node_data_list:
            gpu_id = None
            if self.gpu_manager is not None:
                try:
                    # Get current process ID for GPU assignment
                    process_id = f"worker_{len(futures)}"
                    gpu_id = self.gpu_manager.acquire_gpu(process_id)
                    logger.info(f"Assigned GPU {gpu_id} to process {process_id}")
                except RuntimeError as e:
                    logger.warning(f"Could not acquire GPU: {e}. Running on CPU")

            if (
                self.stage_name
                and self.stage_name.startswith("2_")
                and node_data["is_buggy"] is False
            ):
                new_hyperparam_idea = self._generate_hyperparam_tuning_idea()
                self._hyperparam_tuning_state["tried_hyperparams"].add(
                    new_hyperparam_idea.name
                )
                new_ablation_idea = None
            elif (
                self.stage_name
                and self.stage_name.startswith("4_")
                and node_data["is_buggy"] is False
            ):
                new_ablation_idea = self._generate_ablation_idea()
                self._ablation_state["completed_ablations"].add(new_ablation_idea.name)
                new_hyperparam_idea = None
            else:
                new_ablation_idea = None
                new_hyperparam_idea = None

            best_stage1_plot_code = (
                self.best_stage1_node.plot_code if self.best_stage1_node else None
            )
            best_stage2_plot_code = (
                self.best_stage2_node.plot_code if self.best_stage2_node else None
            )
            best_stage3_plot_code = (
                self.best_stage3_node.plot_code if self.best_stage3_node else None
            )
            seed_eval = False
            futures.append(
                self.executor.submit(
                    self._process_node_wrapper,
                    node_data,
                    self.task_desc,
                    self.cfg,
                    gpu_id,
                    memory_summary,
                    self.evaluation_metrics,
                    self.stage_name,
                    new_ablation_idea,
                    new_hyperparam_idea,
                    best_stage1_plot_code,
                    best_stage2_plot_code,
                    best_stage3_plot_code,
                    seed_eval,
                )
            )

        # Add results to journal
        print("Waiting for results")
        for i, future in enumerate(futures):
            try:
                print("About to get result from future")
                result_data = future.result(timeout=self.timeout)
                if "metric" in result_data:
                    print(f"metric type: {type(result_data['metric'])}")
                    print(f"metric contents: {result_data['metric']}")

                # Create node and restore relationships using journal.
                # Journal acts as a database to look up a parent node,
                # and add the result node as a child.
                result_node = Node.from_dict(result_data, self.journal)
                print("[red]Investigating if result node has metric[/red]", flush=True)
                print(result_node.metric)
                # Update hyperparam tuning state if in Stage 2
                self._update_hyperparam_tuning_state(result_node)
                # Update ablation state if in Stage 4
                self._update_ablation_state(result_node)

                # Add node to journal's list and assign its step number
                self.journal.append(result_node)
                print("Added result node to journal")

            except TimeoutError:
                print("Worker process timed out, couldn't get the result")
                logger.error(f"Worker process timed out, couldn't get the result")
            except Exception as e:
                print(f"Error processing node: {str(e)}")
                logger.error(f"Error processing node: {str(e)}")
                import traceback

                traceback.print_exc()
                raise
            finally:
                # Release GPU for this process if it was using one
                process_id = f"worker_{i}"
                if (
                    self.gpu_manager is not None
                    and process_id in self.gpu_manager.gpu_assignments
                ):
                    self.gpu_manager.release_gpu(process_id)
                    logger.info(f"Released GPU for process {process_id}")

    def _update_hyperparam_tuning_state(self, result_node: Node):
        """Update hyperparam tuning tracking state based on execution results."""
        if not self.stage_name or not self.stage_name.startswith("2_"):
            return

        hyperparam_name = result_node.hyperparam_name
        if hyperparam_name is None:
            print(
                f"[red]hyperparam_name is None for result_node: {result_node.id}[/red]"
            )
            return

        if not result_node.is_buggy:
            self._hyperparam_tuning_state["tried_hyperparams"].add(hyperparam_name)
            logger.info(f"Hyperparam tuning {hyperparam_name} ran successfully")
        else:
            logger.warning(f"Hyperparam tuning {hyperparam_name} failed")

    def _update_ablation_state(self, result_node: Node):
        """Update ablation tracking state based on execution results.

        Args:
            result_node: Node containing ablation execution results
        """
        if not self.stage_name or not self.stage_name.startswith("4_"):
            return

        ablation_name = result_node.ablation_name
        if ablation_name is None:
            print(f"[red]ablation_name is None for result_node: {result_node.id}[/red]")
            return

        if not result_node.is_buggy:
            self._ablation_state["completed_ablations"].add(ablation_name)
            logger.info(f"Ablation {ablation_name} completed successfully")

    def _aggregate_seed_eval_results(
        self, seed_nodes: List[Node], parent_node: Node
    ) -> str:
        """Generate aggregated plots from multi-seed evaluation results.

        Args:
            seed_nodes: List of nodes from seed evaluation
            parent_node: The original node that was evaluated

        Returns:
            str: The plotting code for aggregated results
        """
        prompt_guideline = []
        prompt_guideline += [
            "REQUIREMENTS: ",
            "The code should start with:",
            "  import matplotlib.pyplot as plt",
            "  import numpy as np",
            "  import os",
            "  working_dir = os.path.join(os.getcwd(), 'working')",
            "Create standard visualizations of experiment results",
            "Save all plots to working_dir",
            "Include training/validation curves if available",
            "ONLY plot data that exists in experiment_data.npy - DO NOT make up or simulate any values",
            "Use basic matplotlib without custom styles",
            "Each plot should be in a separate try-except block",
            "Always close figures after saving",
            "Always include a title for each plot, and be sure to use clear subtitles—such as 'Left: Ground Truth, Right: Generated Samples'—while also specifying the type of dataset being used.",
            "Make sure to use descriptive names for figures when saving e.g. always include the dataset name and the type of plot in the name",
            "When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.",
            "Example to extract data from experiment_data: experiment_data['dataset_name_1']['metrics']['train']",
            "Make sure to add legend for standard error bars and means if applicable",
        ]
        prompt_guideline += [
            "Example data loading and plot saving code: ",
            """
                try:
                    experiment_data_path_list = # Make sure to use the correct experiment data path that's provided in the Experiment Data Path section
                    all_experiment_data = []
                    for experiment_data_path in experiment_data_path_list:
                        root = os.getenv("AI_SCIENTIST_SKILLS_ROOT") or os.getenv("AI_SCIENTIST_ROOT") or ""
                        experiment_data = np.load(os.path.join(root, experiment_data_path), allow_pickle=True).item()
                        all_experiment_data.append(experiment_data)
                except Exception as e:
                    print(f'Error loading experiment data: {{e}}')

                try:
                    # First plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_1].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot1: {{e}}")
                    plt.close()  # Always close figure even if error occurs

                try:
                    # Second plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_2].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot2: {{e}}")
                    plt.close()
            """,
        ]
        # add instruction for format
        plotting_prompt = {
            "Introduction": (
                "You are an expert in data visualization and plotting. "
                "You are given a set of evaluation results and the code that was used to plot them. "
                "Your task is to write a new plotting code that aggregate the results "
                "e.g. for example, by adding mean values and standard error bars to the plots."
            ),
            "Instructions": {},
        }
        plotting_prompt["Instructions"] |= {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric(s) if applicable. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }
        plotting_prompt["Instructions"] |= {
            "Plotting code guideline": prompt_guideline,
        }
        plotting_prompt["Instructions"] |= {
            "Plotting code reference": (
                "plotting code 1:\n" + seed_nodes[0].plot_code + "\n\n"
                "plotting code 2:\n" + seed_nodes[1].plot_code + "\n\n"
                "plotting code 3:\n" + seed_nodes[2].plot_code + "\n\n"
            ),
            "Experiment Data Path": (
                f"{seed_nodes[0].exp_results_dir}/experiment_data.npy\n"
                f"{seed_nodes[1].exp_results_dir}/experiment_data.npy\n"
                f"{seed_nodes[2].exp_results_dir}/experiment_data.npy\n"
            ),
        }
        plan, code = self.plan_and_code_query(plotting_prompt)

        print("[green]Plan:[/green]\n", plan)
        print(f"[green]Generated aggregated plotting code:[/green]\n{code}")

        return code

    def __enter__(self):
        return self

    def cleanup(self):
        """Cleanup parallel workers and resources"""
        if not self._is_shutdown:
            print("Shutting down parallel executor...")
            try:
                # Release all GPUs
                if self.gpu_manager is not None:
                    for process_id in list(self.gpu_manager.gpu_assignments.keys()):
                        self.gpu_manager.release_gpu(process_id)

                # Shutdown executor first
                self.executor.shutdown(wait=False, cancel_futures=True)

                # Force terminate all worker processes
                if self.executor._processes:
                    ## Get copy of processes
                    processes = list(self.executor._processes.values())

                    # Then terminate processes if they're still alive
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=1)

                print("Executor shutdown complete")

            except Exception as e:
                print(f"Error during executor shutdown: {e}")
            finally:
                self._is_shutdown = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
