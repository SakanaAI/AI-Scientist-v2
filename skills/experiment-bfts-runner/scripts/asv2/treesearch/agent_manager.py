from typing import List, Optional, Dict, Callable, Any, Tuple
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import logging
from .parallel_agent import ParallelAgent
from .journal import Journal, Node
import copy
import re
from .backend import query, FunctionSpec
import json
from rich import print
from .utils.serialize import parse_markdown_to_dict
from .utils.metric import WorstMetricValue


logger = logging.getLogger(__name__)


stage_config_spec = FunctionSpec(
    name="generate_stage_config",
    description="Generate configuration for the next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Brief, descriptive name for the stage",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the stage's purpose",
            },
            "goals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific, measurable goals for this stage",
            },
            "max_iterations": {
                "type": "integer",
                "description": "Maximum number of iterations to run in this stage",
            },
        },
        "required": ["name", "description", "goals", "max_iterations"],
    },
)

stage_progress_eval_spec = FunctionSpec(
    name="evaluate_stage_progression",
    description="Evaluate readiness to progress to next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "ready_for_next_stage": {
                "type": "boolean",
                "description": "Whether the experiment is ready to progress to next stage",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the progression decision",
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific recommendations for current or next stage",
            },
            "suggested_focus": {
                "type": "string",
                "description": "Key areas to focus on in the next iterations",
            },
        },
        "required": ["ready_for_next_stage", "reasoning", "recommendations"],
    },
)


stage_completion_eval_spec = FunctionSpec(
    name="evaluate_stage_completion",
    description="Evaluate if the current stage is complete",
    json_schema={
        "type": "object",
        "properties": {
            "is_complete": {
                "type": "boolean",
                "description": "Whether the current stage is complete",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the decision",
            },
            "missing_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of criteria still needed",
            },
        },
        "required": ["is_complete", "reasoning", "missing_criteria"],
    },
)


@dataclass
class Stage:
    name: str
    description: str
    goals: List[str]
    max_iterations: int
    num_drafts: int
    stage_number: int


@dataclass
class StageTransition:
    """Records transition between stages and the reasoning"""

    from_stage: str
    to_stage: str
    reason: str
    config_adjustments: Dict[str, Any]


class AgentManager:
    def __init__(self, task_desc: str, cfg: Any, workspace_dir: Path):
        self.task_desc = json.loads(task_desc)
        for k in [
            "Title",
            "Abstract",
            "Short Hypothesis",
            "Experiments",
            "Risk Factors and Limitations",
        ]:
            if k not in self.task_desc.keys():
                raise ValueError(f"Key {k} not found in task_desc")
        self.cfg = cfg
        self.workspace_dir = workspace_dir
        self.current_stage_number = 0
        self.stages: List[Stage] = []
        self.current_stage: Optional[Stage] = None
        self.journals: Dict[str, Journal] = {}
        self.stage_history: List[StageTransition] = []
        self.completed_stages: List[str] = []
        self.main_stage_dict: Dict[int, str] = {
            1: "initial_implementation",
            2: "baseline_tuning",
            3: "creative_research",
            4: "ablation_studies",
        }
        self.main_stage_goals: Dict[int, str] = {
            1: """
                - Focus on getting basic working implementation
                - Use a simple dataset
                - Aim for basic functional correctness
                - If you are given \"Code To Use\", you can directly use it as a starting point.""",
            2: """
                - Change hyperparameters such as learning rate, number of epochs, batch size, etc. to improve the performance
                - DO NOT change the model architecture from the previous stage
                - Introduce TWO more new datasets from HuggingFace test the model. Try very hard to think what Huggingface datasets can be used here for testing.""",
            3: """
                - Explore novel improvements
                - Come up with experiments to reveal new insights
                - Be creative and think outside the box
                - MAKE SURE you use THREE HuggingFace dataset in total to test your models""",
            4: """
                - Conduct systematic component analysis that reveals the contribution of each part
                - Use the same datasets you used from the previous stage""",
        }
        # Create initial stage
        self._create_initial_stage()

    def _get_max_iterations(self, stage_number: int) -> int:
        """Get max iterations for a stage from config or default"""
        return getattr(
            self.cfg.agent.stages,
            f"stage{stage_number}_max_iters",
            self.cfg.agent.steps,
        )

    def _get_task_desc_str(self):
        task_desc = """You are an ambitious AI researcher who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to conduct creative experiments to gain scientific insights.
Your aim is to run experiments to gather sufficient results for a top conference paper.
Your research idea:\n\n
"""
        task_desc += (
            "Title:\n"
            + self.task_desc["Title"]
            + "\n"
            + "Abstract:\n"
            + self.task_desc["Abstract"]
            + "\n"
            + "Short Hypothesis:\n"
            + self.task_desc["Short Hypothesis"]
            + "\n"
        )
        if "Code" in self.task_desc:
            task_desc += "Code To Use:\n" + self.task_desc["Code"] + "\n"
        return task_desc

    def _create_initial_stage(self):
        """Create the initial stage configuration"""
        self.current_stage_number += 1
        initial_stage = Stage(
            name="1_initial_implementation_1_preliminary",
            description="preliminary",
            goals=self.main_stage_goals[1],
            max_iterations=self._get_max_iterations(self.current_stage_number),
            num_drafts=self.cfg.agent.search.num_drafts,
            stage_number=self.current_stage_number,
        )

        self.stages.append(initial_stage)
        self.current_stage = initial_stage
        self.journals[initial_stage.name] = Journal()

    def _curate_task_desc(self, stage: Stage) -> str:
        task_desc = self._get_task_desc_str()

        if stage.name.startswith("3_"):
            if isinstance(self.task_desc["Experiments"], list):
                if isinstance(self.task_desc["Experiments"][0], str):
                    experiment_str = "\n".join(self.task_desc["Experiments"])
                elif isinstance(self.task_desc["Experiments"][0], dict):
                    experiment_str = "\n".join(
                        [
                            f"{k}: {v}"
                            for d in self.task_desc["Experiments"]
                            for k, v in d.items()
                        ]
                    )
            elif isinstance(self.task_desc["Experiments"], str):
                experiment_str = self.task_desc["Experiments"]
            else:
                raise ValueError(
                    f"Experiments is not a list or string: {self.task_desc['Experiments']}"
                )
            task_desc += "Experiment Plan: " + experiment_str + "\n"
        elif stage.name.startswith("4_"):
            if isinstance(self.task_desc["Risk Factors and Limitations"], list):
                risk_factors_str = "\n".join(
                    self.task_desc["Risk Factors and Limitations"]
                )
            else:
                risk_factors_str = self.task_desc["Risk Factors and Limitations"]
            task_desc += "Risk Factors and Limitations: " + risk_factors_str + "\n"

        return task_desc

    def _save_checkpoint(self):
        """Save the current state of the experiment"""
        if self.current_stage is None:
            logger.warning("Cannot save checkpoint: current_stage is None")
            return
        stage_name = "stage_" + self.current_stage.name
        save_path = (
            Path(self.workspace_dir).parent
            / "logs"
            / Path(self.workspace_dir).name
            / stage_name
            / "checkpoint.pkl"
        )
        checkpoint = {
            "journals": self.journals,
            "stage_history": self.stage_history,
            "task_desc": self.task_desc,
            "cfg": self.cfg,
            "workspace_dir": self.workspace_dir,
            "current_stage": self.current_stage,
        }
        print("Saving checkpoint to ", save_path)
        with open(save_path, "wb") as f:
            pickle.dump(checkpoint, f)

    def _create_agent_for_stage(self, stage: Stage) -> ParallelAgent:
        """Create a ParallelAgent configured for the given stage"""
        stage_cfg = self.cfg.copy()
        stage_cfg.agent.search.num_drafts = stage.num_drafts
        task_desc = self._curate_task_desc(stage)

        (
            main_stage,
            main_stage_name,
            sub_stage_num,
            sub_stage_name,
        ) = self.parse_stage_names(stage.name)
        task_desc = f"{task_desc}\n\nCurrent Main Stage: {main_stage_name}\n"
        task_desc += f"Sub-stage: {sub_stage_num} - {sub_stage_name}\n"
        task_desc += f"Sub-stage goals: {stage.goals}"
        print("Checking task_desc inside _create_agent_for_stage")
        print(task_desc)

        if main_stage == 2:
            stage1_substages = [s for s in self.stages if s.name.startswith("1_")]
            if not stage1_substages:
                raise ValueError(f"No stage 1 substages found in {self.stages}")
            best_stage1_node = self._get_best_implementation(stage1_substages[-1].name)
            best_stage2_node = None
            best_stage3_node = None
        elif main_stage == 3:
            stage2_substages = [s for s in self.stages if s.name.startswith("2_")]
            if not stage2_substages:
                raise ValueError(f"No stage 2 substages found in {self.stages}")
            best_stage2_node = self._get_best_implementation(stage2_substages[-1].name)
            best_stage1_node = None
            best_stage3_node = None
        elif main_stage == 4:
            # Use the last (sub-)stage's best node
            stage3_substages = [s for s in self.stages if s.name.startswith("3_")]
            if stage3_substages:
                last_substage = stage3_substages[-1]
                best_stage3_node = self._get_best_implementation(last_substage.name)
                best_stage2_node = None
                best_stage1_node = None
            else:
                raise ValueError(f"No stage 3 substages found in {self.stages}")
        else:
            best_stage3_node = None
            best_stage2_node = None
            best_stage1_node = None

        return ParallelAgent(
            task_desc=task_desc,
            cfg=stage_cfg,
            journal=self.journals[stage.name],
            stage_name=stage.name,
            best_stage3_node=best_stage3_node,
            best_stage2_node=best_stage2_node,
            best_stage1_node=best_stage1_node,
        )

    def _parse_vlm_feedback(self, node: Node) -> str:
        """Parse the feedback from the VLM"""
        if len(node.plot_analyses) > 0:
            feedback = f"Plot analyses: {node.plot_analyses[0]['analysis']}\n"
        else:
            feedback = "No plot analyses found\n"
            logger.warning(
                f"No plot analyses found for node {node.id} during stage {self.current_stage.name}"
            )
        feedback += f"VLM Feedback Summary: {node.vlm_feedback_summary}\n"
        return feedback

    def _check_substage_completion(
        self, current_substage: Stage, journal: Journal
    ) -> bool:
        """Check if the current sub-stage is complete"""
        best_node = journal.get_best_node(cfg=self.cfg)
        if not best_node:
            return False, "No best node found"

        vlm_feedback = self._parse_vlm_feedback(best_node)
        eval_prompt = f"""
        Evaluate if the current sub-stage is complete based on the following evidence:
        1. Figure Analysis:
        {vlm_feedback}

        Requirements for completion:
        - {current_substage.goals}

        Provide a detailed evaluation of completion status.
        """

        try:
            evaluation = query(
                system_message=eval_prompt,
                user_message=None,
                func_spec=stage_completion_eval_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            )
            if evaluation["is_complete"]:
                logger.info(
                    f"Stage {current_substage.name} completed: {evaluation['reasoning']}"
                )
                print(
                    f"[green]Stage {current_substage.name} completed: {evaluation['reasoning']}[/green]"
                )
                return True, "Found working implementation"
            else:
                missing = ", ".join(evaluation["missing_criteria"])
                logger.info(
                    f"Stage {current_substage.name} not complete. Missing: {missing}"
                )
                print(
                    f"[yellow]Stage {current_substage.name} not complete. Missing: {missing}[/yellow]"
                )
                return False, "Missing criteria: " + missing
        except Exception as e:
            logger.error(
                f"Error in sub-stage {current_substage.name} completion evaluation: {e}"
            )
            return (
                False,
                f"Error in sub-stage {current_substage.name} completion evaluation",
            )

        # Terminate if max iterations reached
        if len(journal.nodes) >= current_substage.max_iterations:
            logger.info(
                f"Stage {current_substage.name} completed: reached max iterations"
            )
            print(
                f"[green]Stage {current_substage.name} completed: reached max iterations[/green]"
            )
            return True, "Reached max iterations"

        print(f"[green]Stage {current_substage.name} not completed[/green]")
        return False

    def _check_stage_completion(self, stage: Stage) -> bool:
        """Check if current stage is complete based on criteria"""
        journal = self.journals[stage.name]
        # Terminate if max iterations reached
        if len(journal.nodes) >= stage.max_iterations:
            logger.info(f"Stage {stage.name} completed: reached max iterations")
            print(
                f"[green]Stage {stage.name} completed: reached max iterations[/green]"
            )
            if stage.stage_number == 1:
                # For initial stage, if it didn't even find a working implementation until max iterations,
                # end gracefully and stop the experiment.
                logger.error(
                    f"Initial stage {stage.name} did not find a working implementation after {stage.max_iterations} iterations. Consider increasing the max iterations or reducing the complexity of the research idea."
                )
                print(
                    f"[red]Experiment ended: Could not find working implementation in initial stage after {stage.max_iterations} iterations[/red]"
                )
                self.current_stage = None  # This will cause the run loop to exit
                return True, "Failed to find working implementation"
            else:
                return True, "Reached max iterations"

        # For initial stage, complete when we have at least one working implementation
        if stage.stage_number == 1:
            if len(journal.good_nodes) > 0:
                logger.info(
                    f"Stage {stage.name} completed: found working implementation"
                )
                print(
                    f"[green]Stage {stage.name} completed: found working implementation[/green]"
                )
                return True, "Found working implementation"

        if stage.stage_number == 2:
            best_node = journal.get_best_node(cfg=self.cfg)
            if not best_node:
                return False, "No best node found"
            if best_node == journal.nodes[0]:
                return (
                    False,
                    "No improvement found from the base node (which is the best node from the previous stage)",
                )

            # Normal stage 2 completion check
            vlm_feedback = self._parse_vlm_feedback(best_node)
            eval_prompt = f"""
            Evaluate if stage 2 (baseline tuning) is complete based on the following evidence:

            1. Figure Analysis:
            {vlm_feedback}

            2. Datasets Tested: {best_node.datasets_successfully_tested}

            Requirements for completion:
            1. Training curves should show stable convergence
            2. Results should be tested on at least two datasets
            3. No major instabilities or issues in the plots

            Provide a detailed evaluation of completion status.
            """

            try:
                evaluation = query(
                    system_message=eval_prompt,
                    user_message=None,
                    func_spec=stage_completion_eval_spec,
                    model=self.cfg.agent.feedback.model,
                    temperature=self.cfg.agent.feedback.temp,
                )

                if evaluation["is_complete"]:
                    logger.info(
                        f"Stage {stage.name} completed: {evaluation['reasoning']}"
                    )
                    print(
                        f"[green]Stage {stage.name} completed: {evaluation['reasoning']}[/green]"
                    )
                    return True, "Found working implementation"
                else:
                    missing = ", ".join(evaluation["missing_criteria"])
                    logger.info(f"Stage {stage.name} not complete. Missing: {missing}")
                    print(
                        f"[yellow]Stage {stage.name} not complete. Missing: {missing}[/yellow]"
                    )
                    return False, "Missing criteria: " + missing
            except Exception as e:
                logger.error(f"Error in stage 2 completion evaluation: {e}")
                return False, "Error in stage 2 completion evaluation"

        if stage.stage_number == 3:
            best_node = journal.get_best_node(cfg=self.cfg)
            if not best_node:
                return False, "No best node found"
            if best_node == journal.nodes[0]:
                return (
                    False,
                    "No improvement found from the base node (which is the best node from the previous stage)",
                )
            # Check if there are enough research results
            # Or, we could just let the agent run until max iterations is reached
            # Check if the experiment execution time is too short
            exec_time_minutes = best_node.exec_time / 60
            print(f"[cyan]exec_time_minutes: {exec_time_minutes}[/cyan]")
            if len(self.journals[stage.name].nodes) > (
                self.cfg.agent.stages.stage3_max_iters / 2
            ):
                if exec_time_minutes < self.cfg.exec.timeout / 60 / 2:
                    exec_time_feedback = (
                        f"Implementation works but runs too quickly ({exec_time_minutes:.2f} minutes)."
                        "We have up to 60 minutes available for each experiment."
                        "Make sure to scale up the experiment "
                        "by increasing the number of epochs, using a larger model, or working with bigger datasets."
                        "Given that the current execution time is {exec_time_minutes:.2f} minutes, think about how changing the number of epochs to run, or using a larger model, or working with bigger datasets to run"
                        "will affect the execution time, and make sure to scale up the experiment accordingly."
                    )
                    print(f"[cyan]exec_time_feedback: {exec_time_feedback}[/cyan]")
                    self.journals[stage.name].nodes[
                        -1
                    ].exec_time_feedback = exec_time_feedback
                    return False, exec_time_feedback
        if stage.stage_number == 4:
            # Just let the agent run until max iterations is reached
            pass

        print(f"[green]Stage {stage.name} not completed[/green]")
        return False, "stage not completed"

    def _get_best_implementation(self, stage_name: str) -> Optional[Node]:
        """Get the best implementation from a completed stage"""
        if stage_name not in self.journals:
            return None
        best_node = self.journals[stage_name].get_best_node(cfg=self.cfg)
        if best_node:
            # Create a clean copy of the node for the next stage
            copied_node = copy.deepcopy(best_node)
            # Reset parent relationship and children
            copied_node.parent = None
            copied_node.children = set()
            return copied_node
        return None

    def _generate_substage_goal(self, main_stage_goal: str, journal: Journal) -> str:
        """Generate the next sub-stage goal based on what has been done so far.

        Args:
            main_stage_goal: The overall goal for the current main stage
            journal: Journal containing the results and progress so far

        Returns:
            str: Specific goals for the next sub-stage
        """
        # Gather current progress metrics
        metrics = self._gather_stage_metrics(journal)
        issues = self._identify_issues(journal)
        progress = self._analyze_progress(journal)

        # Create prompt for the LLM
        prompt = f"""
        Based on the current experimental progress, generate focused goals for the next sub-stage.

        Main Stage Goals:
        {main_stage_goal}

        Current Progress:
        - Total attempts: {metrics['total_nodes']}
        - Successful implementations: {metrics['good_nodes']}
        - Best performance: {metrics['best_metric']['value'] if metrics['best_metric'] else 'N/A'}
        - Convergence status: {progress['convergence_status']}

        Current Issues:
        {json.dumps(issues, indent=2)}

        Recent Changes:
        {json.dumps(progress['recent_changes'], indent=2)}

        Generate specific, actionable sub-stage goals that:
        1. Address current issues and limitations
        2. Build on recent progress
        3. Move towards main stage goals
        4. Are concrete and measurable
        """

        # Define the function specification for the LLM
        substage_goal_spec = FunctionSpec(
            name="generate_substage_goals",
            description="Generate specific goals for the next experimental sub-stage",
            json_schema={
                "type": "object",
                "properties": {
                    "goals": {
                        "type": "string",
                        "description": "Detailed, specific goals for the next sub-stage",
                    },
                    "sub_stage_name": {
                        "type": "string",
                        "description": "The name of the next sub-stage",
                    },
                },
                "required": ["goals", "sub_stage_name"],
            },
        )

        try:
            # Get response from LLM
            response = query(
                system_message=prompt,
                user_message=None,
                func_spec=substage_goal_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            )

            # Format the response into a structured goal string
            goal_str = f"""
            {response['goals']}
            """

            return goal_str.strip(), response["sub_stage_name"]

        except Exception as e:
            logger.error(f"Error generating sub-stage goals: {e}")
            # Provide fallback goals if LLM fails
            return f"""
            Sub-stage Goals:
            Continue progress on main stage objectives while addressing current issues.
            """

    def _create_next_substage(
        self, current_substage: Stage, journal: Journal, substage_feedback: str
    ) -> Optional[Stage]:
        """Create the next sub-stage. Ask LLM to come up with the next sub-stage name and goals
        based on what has been done so far.
        """
        main_stage_num, main_stage_name, sub_stage_num, _ = self.parse_stage_names(
            current_substage.name
        )
        main_stage_goal = self.main_stage_goals[main_stage_num]
        sub_stage_goal, sub_stage_name = self._generate_substage_goal(
            main_stage_goal, journal
        )

        return Stage(
            name=f"{main_stage_num}_{main_stage_name}_{sub_stage_num + 1}_{sub_stage_name}",
            description=sub_stage_name,
            goals="Main stage goals:\n"
            + main_stage_goal
            + "\n\nSub-stage goals:\n"
            + sub_stage_goal,
            max_iterations=self._get_max_iterations(main_stage_num),
            num_drafts=0,
            stage_number=current_substage.stage_number + 1,
        )

    def _create_next_main_stage(
        self, current_substage: Stage, journal: Journal
    ) -> Optional[Stage]:
        (
            main_stage_num,
            main_stage_name,
            sub_stage_num,
            sub_stage_name,
        ) = self.parse_stage_names(current_substage.name)
        if main_stage_num == 4:
            return None
        next_main_stage_name = self.main_stage_dict[main_stage_num + 1]
        sub_stage_num = 1
        sub_stage_name = "first_attempt"
        num_drafts = 0
        stage_number = current_substage.stage_number + 1
        description = f"first_attempt"
        main_stage_goal = self.main_stage_goals[main_stage_num + 1]

        return Stage(
            name=f"{main_stage_num + 1}_{next_main_stage_name}_{sub_stage_num}_{sub_stage_name}",
            description=description,
            goals=main_stage_goal,
            max_iterations=self._get_max_iterations(main_stage_num + 1),
            num_drafts=num_drafts,
            stage_number=stage_number,
        )

    def run(self, exec_callback, step_callback=None):
        """Run the experiment through generated stages"""
        while self.current_stage:  # Main stage loop
            main_stage = self.parse_stage_names(self.current_stage.name)[0]
            print(f"[green]Starting main stage: {main_stage}[/green]")
            print(f"[cyan]Goals: {self.current_stage.goals}[/cyan]")

            current_substage = self.current_stage
            while current_substage:  # Sub-stage loop
                print(f"[green]Starting sub-stage: {current_substage.name}[/green]")

                with self._create_agent_for_stage(current_substage) as agent:
                    # Initialize with best result from previous sub-stage if available
                    if self.stage_history:
                        prev_stage = self.stage_history[-1].from_stage
                        print(f"[cyan]prev_stage: {prev_stage}[/cyan]")
                        print(f"[cyan]self.stage_history: {self.stage_history}[/cyan]")
                        prev_best = self._get_best_implementation(prev_stage)
                        if prev_best:
                            self.journals[self.current_stage.name].append(prev_best)
                        else:
                            print(
                                f"[red]No previous best implementation found for {self.current_stage.name}. Something went wrong so finishing the experiment...[/red]"
                            )
                            self.current_stage = None
                            current_substage = None
                            break

                    # Run until sub-stage completion
                    while True:
                        agent.step(exec_callback)
                        if step_callback:
                            step_callback(
                                current_substage, self.journals[current_substage.name]
                            )

                        # First check if main stage is complete
                        (
                            main_stage_complete,
                            main_stage_feedback,
                        ) = self._check_stage_completion(current_substage)
                        print(
                            f"[cyan]Feedback from _check_stage_completion: {main_stage_feedback}[/cyan]"
                        )
                        if main_stage_complete:
                            # After main stage completion, run multi-seed eval on the best node
                            if current_substage.stage_number in [1, 2, 3, 4]:
                                best_node = self._get_best_implementation(
                                    current_substage.name
                                )
                                if best_node:
                                    seed_nodes = agent._run_multi_seed_evaluation(
                                        best_node
                                    )
                                    if step_callback:
                                        step_callback(
                                            current_substage,
                                            self.journals[current_substage.name],
                                        )
                                    agent._run_plot_aggregation(best_node, seed_nodes)
                                    if step_callback:
                                        step_callback(
                                            current_substage,
                                            self.journals[current_substage.name],
                                        )
                                    print(
                                        f"Stage {current_substage.name} multi-seed eval done."
                                    )
                                else:
                                    logger.error(
                                        f"No best node found for {current_substage.name} during multi-seed eval, something went wrong so finishing the experiment..."
                                    )
                                    self.current_stage = None
                                    current_substage = None
                                    break

                            # Exit the loop to move to next main stage
                            current_substage = None
                            break

                        (
                            substage_complete,
                            substage_feedback,
                        ) = self._check_substage_completion(
                            current_substage, self.journals[current_substage.name]
                        )

                        if substage_complete:
                            # Create next sub-stage
                            next_substage = self._create_next_substage(
                                current_substage,
                                self.journals[current_substage.name],
                                substage_feedback,
                            )
                            if next_substage:
                                # Record sub-stage transition
                                self.stage_history.append(
                                    StageTransition(
                                        from_stage=current_substage.name,
                                        to_stage=next_substage.name,
                                        reason=substage_feedback,
                                        config_adjustments={},
                                    )
                                )

                                # Setup new sub-stage
                                self.stages.append(next_substage)
                                self.journals[next_substage.name] = Journal()
                                current_substage = next_substage
                            else:
                                # If no next sub-stage could be created, end this main stage
                                current_substage = None
                            break
            self._save_checkpoint()
            # Main stage complete - create next main stage
            if self.current_stage:
                next_main_stage = self._create_next_main_stage(
                    self.stages[-1], self.journals[self.stages[-1].name]
                )
                if next_main_stage:
                    # Record main stage transition
                    self.stage_history.append(
                        StageTransition(
                            from_stage=self.stages[-1].name,
                            to_stage=next_main_stage.name,
                            reason=f"Moving to {next_main_stage.description}",
                            config_adjustments={},
                        )
                    )

                    self.stages.append(next_main_stage)
                    self.journals[next_main_stage.name] = Journal()
                    self.current_stage = next_main_stage
                else:
                    # Exit the outer loop if no more main stages
                    logger.info(f"Completed stage: {self.current_stage.name}")
                    logger.info("No more stages to run -- exiting the loop...")
                    self.current_stage = None

    def _create_stage_analysis_prompt(
        self,
        previous_stages: List[Stage],
        previous_results: Optional[Dict[str, Any]],
        is_initial_stage: bool,
    ) -> str:
        """Create detailed prompt to determine next stage configuration"""
        prompt_parts = [
            f"Task Description: {self._curate_task_desc(previous_stages[-1])}",
            f"Current Stage Number: {previous_stages[-1].stage_number}",
        ]

        if previous_stages:
            stage_history = "\n".join(
                f"Stage {i+1}: {stage.name} - {stage.description}"
                for i, stage in enumerate(previous_stages)
            )
            prompt_parts.append(f"Previous Stages:\n{stage_history}")

        if previous_results:
            # Format node summaries
            if "node_summaries" in previous_results["metrics"]:
                summaries = "\n".join(
                    f"Node {i}: {summary}"
                    for i, summary in enumerate(
                        previous_results["metrics"]["node_summaries"]
                    )
                )
                prompt_parts.append(f"Node Analysis:\n{summaries}")

            # Format VLM feedback and plot analysis
            if "plot_insights" in previous_results:
                plot_insights = previous_results["plot_insights"]
                prompt_parts.append("Visual Analysis Findings:")
                for analysis in plot_insights["analyses"]:
                    prompt_parts.append(f"- {analysis['analysis']}")

            # Format other metrics and findings
            metrics_summary = (
                f"Progress Summary:\n"
                f"- Total attempts: {previous_results['metrics']['total_nodes']}\n"
                f"- Successful implementations: {previous_results['metrics']['good_nodes']}\n"
                f"- Failed attempts: {previous_results['metrics']['buggy_nodes']}\n"
                f"- Best performance: {previous_results['metrics']['best_metric']['value'] if previous_results['metrics']['best_metric'] else 'N/A'}\n"
                f"- Issues identified: {', '.join(previous_results['issues'])}\n"
                f"- Progress status: {previous_results['progress']['convergence_status']}"
            )
            prompt_parts.append(metrics_summary)

            # Save stage transition analysis to notes directory
            base_dir = Path(self.workspace_dir).parent.parent
            run_name = Path(self.workspace_dir).name
            notes_dir = (
                base_dir
                / "logs"
                / run_name
                / "notes"
                / f"stage_{stage_number-1}_to_{stage_number}"
            )
            notes_dir.mkdir(parents=True, exist_ok=True)

            analysis_data = {
                "stage_transition": {
                    "from_stage": stage_number - 1,
                    "to_stage": stage_number,
                    "is_initial_stage": is_initial_stage,  # Add flag for initial stage
                    "metrics_summary": metrics_summary,
                    "node_summaries": previous_results["metrics"].get(
                        "node_summaries", []
                    ),
                    "plot_insights": previous_results.get("plot_insights", {}),
                    "issues": previous_results["issues"],
                    "progress": previous_results["progress"],
                }
            }

            with open(notes_dir / "stage_transition_analysis.json", "w") as f:
                json.dump(analysis_data, f, indent=2)

        prompt_parts.append(
            "Based on the above comprehensive analysis, determine the appropriate "
            "configuration for the next experimental stage. Consider:\n"
            "1. Visual analysis insights from plots\n"
            "2. Individual node performance and patterns\n"
            "3. Overall progress and convergence status\n"
            "4. Identified issues and challenges\n\n"
            "Include:\n"
            "1. Stage name (brief, descriptive)\n"
            "2. Detailed description of the stage's purpose\n"
            "3. Specific, measurable goals\n"
            "4. Maximum iterations needed\n"
            "5. Success metric threshold (if applicable)"
        )

        return "\n\n".join(prompt_parts)

    def parse_stage_names(self, stage_name: str) -> Tuple[int, str, int, str]:
        """Parse stage name into main stage number, main stage name,
        sub-stage number, and sub-stage name"""
        # Find the two numbers in the current stage name
        numbers = [int(n) for n in re.findall(r"\d+", stage_name)]

        main_stage = numbers[0]
        sub_stage_num = numbers[1]
        # Extract main_stage_name (everything between the two numbers)
        parts = re.split(r"\d+", stage_name)[1:-1]
        main_stage_name = "_".join(p.strip("_") for p in parts if p.strip("_"))
        # Extract sub_stage_name (everything after the second number)
        sub_stage_name = re.split(r"\d+", stage_name)[-1].strip("_")

        return main_stage, main_stage_name, sub_stage_num, sub_stage_name

    def _save_stage_summary(
        self, current_results: Dict[str, Any], evaluation: Dict[str, Any]
    ):
        """Save comprehensive stage completion summary"""
        base_dir = Path(self.workspace_dir).parent.parent
        run_name = Path(self.workspace_dir).name
        notes_dir = (
            base_dir
            / "logs"
            / run_name
            / "notes"
            / f"stage_{self.current_stage.stage_number}_complete"
        )
        notes_dir.mkdir(parents=True, exist_ok=True)

        completion_data = {
            "stage_completion": {
                "stage_number": self.current_stage.stage_number,
                "stage_name": self.current_stage.name,
                "final_metrics": current_results["metrics"],
                "identified_issues": current_results["issues"],
                "progress_analysis": current_results["progress"],
                "plot_insights": current_results.get("plot_insights", {}),
                "progression_evaluation": {
                    "ready_for_next_stage": evaluation["ready_for_next_stage"],
                    "reasoning": evaluation["reasoning"],
                    "recommendations": evaluation["recommendations"],
                    "suggested_focus": evaluation["suggested_focus"],
                },
            }
        }

        with open(notes_dir / "stage_completion_summary.json", "w") as f:
            json.dump(completion_data, f, indent=2)

    def _get_response(self, prompt: str) -> Dict[str, Any]:
        """Get structured response from LLM for stage configuration.

        Args:
            prompt: The analysis prompt to send to the LLM

        Returns:
            Dictionary containing stage configuration with keys:
            - name: str
            - description: str
            - goals: List[str]
            - max_iterations: int
            - success_metric_threshold: Optional[float]
        """
        stage_config_spec = {
            "name": "generate_stage_config",
            "json_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Brief, descriptive name for the stage",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the stage's purpose",
                    },
                    "goals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific, measurable goals for this stage",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum number of iterations to run in this stage",
                    },
                },
                "required": ["name", "description", "goals", "max_iterations"],
            },
            "description": "Generate configuration for the next experimental stage",
        }

        try:
            response = query(
                system_message=prompt,
                user_message=None,
                func_spec=stage_config_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            )
            return response

        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            # Provide a fallback configuration in case of errors
            return {
                "name": "fallback_stage",
                "description": "Fallback stage due to LLM error",
                "goals": ["Recover from error and continue execution"],
                "max_iterations": 3,
                "success_metric_threshold": None,
            }

    def _gather_stage_metrics(self, journal: Journal) -> Dict[str, Any]:
        """Gather detailed metrics and analysis from the stage's nodes"""
        metrics = {
            "total_nodes": len(journal.nodes),
            "good_nodes": len(journal.good_nodes),
            "buggy_nodes": len(journal.buggy_nodes),
            "best_metric": None,
            "node_summaries": [],
            "vlm_feedback": [],
        }

        # Gather individual node summaries
        for node in journal.nodes:
            if hasattr(node, "_agent"):
                node_summary = node._agent._generate_node_summary(node)
                metrics["node_summaries"].append(node_summary)

        # Get VLM feedback from plot analysis
        for node in journal.good_nodes:
            if hasattr(node, "_vlm_feedback"):
                metrics["vlm_feedback"].append(node._vlm_feedback)

        best_node = journal.get_best_node(cfg=self.cfg)
        if best_node:
            metrics["best_metric"] = {
                "value": best_node.metric.value,
                "name": (
                    best_node.metric.name
                    if hasattr(best_node.metric, "name")
                    else "validation_metric"
                ),
                "maximize": (
                    best_node.metric.maximize
                    if hasattr(best_node.metric, "maximize")
                    else False
                ),
                "analysis": (
                    best_node.analysis if hasattr(best_node, "analysis") else None
                ),
            }

        return metrics

    def _identify_issues(self, journal: Journal) -> List[str]:
        """Identify systemic issues and challenges from the current stage's results"""
        issues = []

        # Look for patterns in leaf nodes (endpoints of improvement attempts)
        leaf_nodes = [n for n in journal.nodes if n.is_leaf]
        buggy_leaves = [n for n in leaf_nodes if n.is_buggy]

        # If we have buggy leaf nodes, it means we couldn't fix some issues
        if buggy_leaves:
            # Group similar issues
            error_patterns = {}
            for node in buggy_leaves:
                if hasattr(node, "analysis"):
                    # Use the error message as key to group similar issues
                    error_patterns.setdefault(node.analysis, []).append(node.id)

            # Report persistent issues
            for error_msg, node_ids in error_patterns.items():
                if len(node_ids) >= 2:  # If same error occurs multiple times
                    issues.append(f"Persistent issue in nodes {node_ids}: {error_msg}")

        # Include VLM-identified systemic issues
        vlm_issues = set()  # Use set to avoid duplicate issues
        for node in journal.good_nodes:
            if hasattr(node, "_vlm_feedback"):
                vlm_feedback = node._vlm_feedback
                if isinstance(vlm_feedback, dict):
                    # Look for systemic issues identified by VLM
                    if "systemic_issues" in vlm_feedback:
                        vlm_issues.update(vlm_feedback["systemic_issues"])
                    # Look for recurring patterns in plot analysis
                    if "plot_analyses" in vlm_feedback:
                        for analysis in vlm_feedback["plot_analyses"]:
                            if "limitation" in analysis.get("type", "").lower():
                                vlm_issues.add(
                                    f"VLM (Node {node.id}): {analysis['analysis']}"
                                )

        issues.extend(list(vlm_issues))

        return issues

    def _analyze_progress(self, journal: Journal) -> Dict[str, Any]:
        """Analyze progress and convergence in the current stage"""
        progress = {
            "iterations_completed": len(journal.nodes),
            "improvements_found": 0,
            "convergence_status": "not_converged",
            "improvement_trend": [],
            "recent_changes": [],
        }

        # Analyze recent changes
        recent_nodes = journal.nodes[-3:] if len(journal.nodes) >= 3 else journal.nodes
        for node in recent_nodes:
            if not node.is_buggy:
                change = {
                    "node_id": node.id,
                    "metric": node.metric.value,
                    "parent_id": node.parent.id if node.parent else None,
                    "analysis": node.analysis if hasattr(node, "analysis") else None,
                }
                progress["recent_changes"].append(change)

        return progress

    def _evaluate_stage_progression(
        self, current_stage: Stage, previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate whether experiment is ready for next stage"""

        eval_prompt = f"""
        Evaluate whether the current experimental stage should progress to the next stage.
        Consider all available evidence holistically:

        Current Stage Information:
        - Name: {current_stage.name}
        - Description: {current_stage.description}
        - Goals: {', '.join(current_stage.goals) if isinstance(current_stage.goals, list) else current_stage.goals}

        Performance Metrics:
        {json.dumps(previous_results.get('metrics', {}), indent=2)}

        Identified Issues:
        {json.dumps(previous_results.get('issues', []), indent=2)}

        Progress Analysis:
        {json.dumps(previous_results.get('progress', {}), indent=2)}

        Expected Stage Progression:
        1. Initial Implementation: Focus on basic working implementation
        2. Baseline Tuning: Systematic optimization of core parameters
        3. Creative Research: Novel improvements and approaches
        4. Ablation Studies: Systematic component analysis

        Consider factors like:
        - Progress toward stage goals
        - Performance trends and stability
        - Quality and reliability of results
        - Understanding of the problem
        - Presence of systematic issues
        - Convergence indicators
        - Readiness for next stage challenges

        Provide a holistic evaluation of whether the experiment should:
        1. Progress to next stage
        2. Continue current stage with specific focus
        3. Extend current stage with modifications
        """

        try:
            evaluation = query(
                system_message=eval_prompt,
                user_message=None,
                func_spec=stage_progress_eval_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            )

            # Log the evaluation for transparency
            logger.info(
                f"Stage progression evaluation:\n{json.dumps(evaluation, indent=2)}"
            )

            return evaluation

        except Exception as e:
            logger.error(f"Error in stage progression evaluation: {e}")
            return {
                "ready_for_next_stage": False,
                "reasoning": "Error in evaluation process - continuing current stage",
                "recommendations": [
                    "Address evaluation error",
                    "Continue current approach",
                ],
                "suggested_focus": "Maintain current direction while resolving evaluation issues",
            }
