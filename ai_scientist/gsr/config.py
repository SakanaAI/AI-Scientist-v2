"""GSR-specific configuration defaults and schema."""

from dataclasses import dataclass


@dataclass
class GSRConfig:
    """Configuration for the GSR orchestrator.

    Attributes:
        budget: Total number of evaluation rounds (T in the paper).
        initial_mutation_ratio: Starting mutation ratio rho_0.
        generation_batch_size: Number of child ideas per generation (J).
        confidence_gate: Threshold multiplier c_g for triggering refinement.
        lipschitz_bound: Initial Lipschitz constant estimate L_bar.
        committee_size: Number of LLM committee voters (K).
        max_refinement_levels: Upper bound on coarse-to-fine levels m_bar_T.
        evaluation_depth: How many BFTS stages to run per evaluation step.
        model_mutator: LLM model for idea mutation.
        model_evaluator: LLM model for committee-based evaluation.
        temperature_mutator: Temperature for mutation LLM calls.
        temperature_evaluator: Temperature for evaluator LLM calls.
        epsilon_u_0: Initial utility resolution epsilon^U_0.
        noise_std: Assumed sub-Gaussian noise std for utility observations.
        min_evaluations_before_refine: Minimum evaluations on a task before
            it can serve as an anchor.
    """

    budget: int = 20
    initial_mutation_ratio: float = 0.8
    generation_batch_size: int = 3
    confidence_gate: float = 0.5
    lipschitz_bound: float = 1.0
    committee_size: int = 5
    max_refinement_levels: int = 5
    evaluation_depth: int = 1
    model_mutator: str = "deepseek-v3.2"
    model_evaluator: str = "deepseek-v3.2"
    temperature_mutator: float = 0.9
    temperature_evaluator: float = 0.3
    epsilon_u_0: float = 1.0
    noise_std: float = 0.1
    min_evaluations_before_refine: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "GSRConfig":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known_fields})
