"""Launch AI Scientist v2 with GSR (Generate-Select-Refine) orchestration.

This entry point implements the open-ended BO framework from
"Open-Ended Task Discovery via Bayesian Optimization" (Adachi et al., 2026)
on top of the AI Scientist v2 pipeline.

Instead of running a single idea through the full BFTS pipeline,
GSR manages a pool of ideas, selecting which to evaluate next via
UCB, and generating refined ideas via coarse-to-fine mutation.

Usage:
    python launch_scientist_gsr.py \\
        --load_ideas ai_scientist/ideas/my_topic.json \\
        --budget 20 \\
        --generation_batch_size 3 \\
        --model_writeup deepseek-v3.2 \\
        --model_review deepseek-v3.2
"""

import argparse
import json
import os
import os.path as osp
import shutil
import sys

from ai_scientist.gsr.config import GSRConfig
from ai_scientist.gsr.gsr_orchestrator import GSROrchestrator
from ai_scientist.llm import create_client
from ai_scientist.perform_icbinb_writeup import (
    gather_citations,
    perform_writeup as perform_icbinb_writeup,
)
from ai_scientist.perform_llm_review import load_paper, perform_review
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.utils.token_tracker import token_tracker


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run AI Scientist v2 with GSR orchestration"
    )

    # --- Idea input ---
    parser.add_argument(
        "--load_ideas",
        type=str,
        required=True,
        help="Path to JSON file containing seed research ideas",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="Load a .py file with the same stem as the ideas JSON",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="Add HuggingFace dataset reference code",
    )

    # --- GSR parameters ---
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Total GSR evaluation rounds (T)",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=3,
        help="Number of child ideas per generation step (J)",
    )
    parser.add_argument(
        "--initial_mutation_ratio",
        type=float,
        default=0.8,
        help="Starting mutation ratio rho_0",
    )
    parser.add_argument(
        "--confidence_gate",
        type=float,
        default=0.5,
        help="Threshold multiplier c_g for triggering refinement",
    )
    parser.add_argument(
        "--committee_size",
        type=int,
        default=5,
        help="Number of LLM committee voters (K)",
    )
    parser.add_argument(
        "--max_refinement_levels",
        type=int,
        default=5,
        help="Max coarse-to-fine levels",
    )
    parser.add_argument(
        "--lipschitz_bound",
        type=float,
        default=1.0,
        help="Initial Lipschitz constant estimate",
    )
    parser.add_argument(
        "--model_mutator",
        type=str,
        default="deepseek-v3.2",
        help="LLM model for idea mutation",
    )
    parser.add_argument(
        "--model_evaluator",
        type=str,
        default="deepseek-v3.2",
        help="LLM model for committee evaluation",
    )

    # --- Writeup / review (post-GSR) ---
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Type of writeup to generate",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="deepseek-v3.2",
        help="Model for paper writeup",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="deepseek-v3.2",
        help="Smaller model for writeup",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="deepseek-v3.2",
        help="Model for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="deepseek-v3.2",
        help="Model for paper review",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="deepseek-v3.2",
        help="Model for plot aggregation",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="Skip the writeup phase",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="Skip the review phase",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")

    # Load seed ideas
    with open(args.load_ideas, "r") as f:
        seed_ideas = json.load(f)
    print(f"Loaded {len(seed_ideas)} seed ideas from {args.load_ideas}")

    # Load optional code context
    code_context = None
    if args.load_code:
        code_path = args.load_ideas.rsplit(".", 1)[0] + ".py"
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code_context = f.read()
        else:
            print(f"Warning: Code file {code_path} not found")

    if args.add_dataset_ref:
        dataset_ref_path = "hf_dataset_reference.py"
        if os.path.exists(dataset_ref_path):
            with open(dataset_ref_path, "r") as f:
                ref_code = f.read()
            if code_context:
                code_context = ref_code + "\n" + code_context
            else:
                code_context = ref_code

    # Build GSR config
    gsr_config = GSRConfig(
        budget=args.budget,
        initial_mutation_ratio=args.initial_mutation_ratio,
        generation_batch_size=args.generation_batch_size,
        confidence_gate=args.confidence_gate,
        lipschitz_bound=args.lipschitz_bound,
        committee_size=args.committee_size,
        max_refinement_levels=args.max_refinement_levels,
        model_mutator=args.model_mutator,
        model_evaluator=args.model_evaluator,
    )

    # Run GSR
    orchestrator = GSROrchestrator(
        gsr_config=gsr_config,
        bfts_config_path="bfts_config.yaml",
        code_context=code_context,
    )

    summary = orchestrator.run(seed_ideas)

    # Post-GSR: writeup + review on best idea
    best_info = summary.get("best_task")
    if best_info is None:
        print("GSR did not find a viable idea. Exiting.")
        sys.exit(1)

    idea_dir = best_info["experiment_dir"]
    if idea_dir is None:
        print("Best idea has no experiment directory. Exiting.")
        sys.exit(1)

    print(f"\nBest idea: {best_info['idea'].get('Title', '?')}")
    print(f"Utility: {best_info['incumbent_utility']:.4f}")
    print(f"Experiment dir: {idea_dir}")

    # Copy experiment_results to top-level for aggregate_plots / writeup,
    # mirroring the pattern in launch_scientist_bfts.py:257-267.
    exp_results_src = osp.join(idea_dir, "logs/0-run/experiment_results")
    exp_results_dst = osp.join(idea_dir, "experiment_results")
    if os.path.exists(exp_results_src):
        shutil.copytree(exp_results_src, exp_results_dst, dirs_exist_ok=True)

    # Aggregate plots
    try:
        aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)
    except Exception as e:
        print(f"Plot aggregation failed: {e}")

    # Clean up the copy after plotting
    if os.path.exists(exp_results_dst):
        shutil.rmtree(exp_results_dst)

    # Save token tracker
    _save_token_tracker(idea_dir)

    # Writeup
    if not args.skip_writeup:
        writeup_success = False
        citations_text = gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt + 1} of {args.writeup_retries}")
            if args.writeup_type == "normal":
                # perform_writeup handles its own citation gathering
                # via num_cite_rounds, so we don't pass citations_text.
                writeup_success = perform_writeup(
                    base_folder=idea_dir,
                    num_cite_rounds=args.num_cite_rounds,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=8,
                )
            else:
                writeup_success = perform_icbinb_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=4,
                    citations_text=citations_text,
                )
            if writeup_success:
                break

        if not writeup_success:
            print("Writeup did not complete successfully after all retries.")

    _save_token_tracker(idea_dir)

    # Review
    if not args.skip_review and not args.skip_writeup:
        pdf_path = _find_pdf(idea_dir)
        if pdf_path and os.path.exists(pdf_path):
            print(f"Reviewing paper at: {pdf_path}")
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)
            review_img = perform_imgs_cap_ref_review(client, client_model, pdf_path)
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img, f, indent=4)
            print("Paper review completed.")

    # Save final GSR summary alongside experiment
    final_summary_path = osp.join(idea_dir, "gsr_final_summary.json")
    with open(final_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nGSR run complete.")
    print(f"Results in: {summary['gsr_dir']}")
    print(f"Best idea dir: {idea_dir}")

    _save_token_tracker(idea_dir)
    sys.exit(0)


def _save_token_tracker(idea_dir: str) -> None:
    try:
        with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
            json.dump(token_tracker.get_summary(), f)
        with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
            json.dump(token_tracker.get_interactions(), f)
    except Exception as e:
        print(f"Failed to save token tracker: {e}")


def _find_pdf(idea_dir: str) -> str:
    """Find the best PDF in the idea directory."""
    import re

    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if reflection_pdfs:
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            return osp.join(idea_dir, final_pdfs[0])
        nums = []
        for f in reflection_pdfs:
            m = re.search(r"reflection[_.]?(\d+)", f)
            if m:
                nums.append((int(m.group(1)), f))
        if nums:
            return osp.join(idea_dir, max(nums, key=lambda x: x[0])[1])
        return osp.join(idea_dir, reflection_pdfs[0])
    if pdf_files:
        return osp.join(idea_dir, pdf_files[0])
    return osp.join(idea_dir, "paper.pdf")


if __name__ == "__main__":
    main()
