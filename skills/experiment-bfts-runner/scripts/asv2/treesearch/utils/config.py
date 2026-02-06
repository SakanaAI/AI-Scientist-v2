"""configuration and setup utils"""

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, cast, Literal, Optional

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from . import tree_export
from . import copytree, preproc_data, serialize

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("ai-scientist")
logger.setLevel(logging.WARNING)


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class ThinkingConfig:
    type: str
    budget_tokens: Optional[int] = None


@dataclass
class StageConfig:
    model: str
    temp: float
    thinking: ThinkingConfig
    betas: str
    max_tokens: Optional[int] = None


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class DebugConfig:
    stage4: bool


@dataclass
class AgentConfig:
    steps: int
    stages: dict[str, int]
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool

    code: StageConfig
    feedback: StageConfig
    vlm_feedback: StageConfig

    search: SearchConfig
    num_workers: int
    type: str
    multi_seed_eval: dict[str, int]

    summary: Optional[StageConfig] = None
    select_node: Optional[StageConfig] = None

@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class ExperimentConfig:
    num_syn_datasets: int


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig
    experiment: ExperimentConfig
    debug: DebugConfig


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if (current_index := int(p.name.split("-")[0])) > max_index:
                max_index = current_index
        except ValueError:
            pass
    print("max_index: ", max_index)
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=False
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    if cfg.agent.type not in ["parallel", "sequential"]:
        raise ValueError("agent.type must be either 'parallel' or 'sequential'")

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval
    print(task_desc)
    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal, stage_name: str = None):
    if stage_name is None:
        stage_name = "NoStageRun"
    save_dir = cfg.log_dir / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    try:
        serialize.dump_json(journal, save_dir / "journal.json")
    except Exception as e:
        print(f"Error saving journal: {e}")
        raise
    # save config
    try:
        OmegaConf.save(config=cfg, f=save_dir / "config.yaml")
    except Exception as e:
        print(f"Error saving config: {e}")
        raise
    # create the tree + code visualization
    try:
        tree_export.generate(cfg, journal, save_dir / "tree_plot.html")
    except Exception as e:
        print(f"Error generating tree: {e}")
        raise
    # save the best found solution
    try:
        best_node = journal.get_best_node(only_good=False, cfg=cfg)
        if best_node is not None:
            for existing_file in save_dir.glob("best_solution_*.py"):
                existing_file.unlink()
            # Create new best solution file
            filename = f"best_solution_{best_node.id}.py"
            with open(save_dir / filename, "w") as f:
                f.write(best_node.code)
            # save best_node.id to a text file
            with open(save_dir / "best_node_id.txt", "w") as f:
                f.write(str(best_node.id))
        else:
            print("No best node found yet")
    except Exception as e:
        print(f"Error saving best solution: {e}")
