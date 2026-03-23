#!/usr/bin/env python3
"""
Resume the AI-Scientist BFTS pipeline after experiments have finished:
  1) (optional) copy logs/0-run/experiment_results -> <folder>/experiment_results
  2) aggregate_plots (final figures under figures/)
  3) remove the temporary experiment_results copy (same as launch_scientist_bfts.py)
  4) gather_citations + perform_writeup / perform_icbinb_writeup
  5) optional: PDF review (same logic as launch_scientist_bfts.py)
  6) optional: child-process cleanup via psutil

Usage (from repo root):
  python after_experiments.py --folder experiments/2026-03-22_13-03-16_reviewforge_openreview_attempt_0

Requires:
  - Existing run with logs/0-run/*.json summaries (baseline/research/ablation) from BFTS.
  - Run from repository root so relative paths like ai_scientist/blank_* resolve.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import shutil
import sys
import traceback

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continue after BFTS experiments: aggregate plots → writeup (optional review)."
    )
    p.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Existing experiment directory, e.g. experiments/2026-03-22_..._attempt_0",
    )
    p.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="normal=ICML-style 8 pages, icbinb=4 pages (default: icbinb, matches launch_scientist_bfts)",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip aggregate_plots; use existing figures/ if any.",
    )
    p.add_argument(
        "--skip-writeup",
        action="store_true",
        help="Only run plot aggregation (and copies), skip citations + LaTeX writeup.",
    )
    p.add_argument(
        "--run-review",
        action="store_true",
        help="If set, run LLM+VLM paper review when a reflection PDF exists (default: skip).",
    )
    p.add_argument(
        "--model-agg-plots",
        type=str,
        default="deepseek-v3.2",
        help="Model for plot aggregation script generation.",
    )
    p.add_argument(
        "--model-writeup",
        type=str,
        default="deepseek-v3.2",
        help="Big model for writeup.",
    )
    p.add_argument(
        "--model-writeup-small",
        type=str,
        default="deepseek-v3.2",
        help="Small model for writeup sub-steps.",
    )
    p.add_argument(
        "--model-citation",
        type=str,
        default="deepseek-v3.2",
        help="Model for gather_citations (icbinb path).",
    )
    p.add_argument(
        "--model-review",
        type=str,
        default="deepseek-v3.2",
        help="Model for optional text review.",
    )
    p.add_argument(
        "--num-cite-rounds",
        type=int,
        default=20,
        help="Citation gathering rounds (passed to gather_citations).",
    )
    p.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Max writeup attempts.",
    )
    p.add_argument(
        "--vlm-model",
        type=str,
        default="qwen/qwen3-vl-plus",
        help="VLM for writeup / review (use 'auto' or ollama/<tag> per project conventions).",
    )
    p.add_argument(
        "--plot-reflections",
        type=int,
        default=5,
        help="Reflection loops inside aggregate_plots.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return p.parse_args()


def _repo_root() -> str:
    return osp.dirname(osp.abspath(__file__))


def _ensure_repo_layout(idea_dir: str) -> None:
    summaries = [
        osp.join(idea_dir, "logs/0-run/baseline_summary.json"),
        osp.join(idea_dir, "logs/0-run/research_summary.json"),
        osp.join(idea_dir, "logs/0-run/ablation_summary.json"),
    ]
    missing = [s for s in summaries if not osp.exists(s)]
    if missing:
        logger.warning(
            "Some stage summary JSON files are missing (writeup/plots may be low-quality):\n  %s",
            "\n  ".join(missing),
        )
    if not osp.exists(osp.join(idea_dir, "idea.md")) and not osp.exists(
        osp.join(idea_dir, "research_idea.md")
    ):
        logger.warning(
            "No idea.md or research_idea.md under %s — writeup may still use other inputs.",
            idea_dir,
        )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    repo_root = _repo_root()
    os.chdir(repo_root)
    os.environ.setdefault("AI_SCIENTIST_ROOT", repo_root)
    logger.info("AI_SCIENTIST_ROOT=%s", os.environ["AI_SCIENTIST_ROOT"])

    idea_dir = osp.abspath(args.folder)
    if not osp.isdir(idea_dir):
        logger.error("Not a directory: %s", idea_dir)
        return 1

    _ensure_repo_layout(idea_dir)

    # Deferred imports after env / chdir (templates use relative paths)
    from ai_scientist.perform_plotting import aggregate_plots
    from ai_scientist.perform_writeup import perform_writeup
    from ai_scientist.perform_icbinb_writeup import (
        perform_writeup as perform_icbinb_writeup,
        gather_citations,
    )
    from ai_scientist.utils.token_tracker import token_tracker
    from ai_scientist.vlm import resolve_vlm_model

    vlm_resolved = resolve_vlm_model(args.vlm_model)

    experiment_results_src = osp.join(idea_dir, "logs/0-run/experiment_results")
    experiment_results_tmp = osp.join(idea_dir, "experiment_results")

    if not args.skip_plots:
        if osp.exists(experiment_results_src):
            if osp.exists(experiment_results_tmp):
                shutil.rmtree(experiment_results_tmp)
            shutil.copytree(
                experiment_results_src,
                experiment_results_tmp,
                dirs_exist_ok=True,
            )
            logger.info("Copied %s -> %s", experiment_results_src, experiment_results_tmp)
        else:
            logger.warning(
                "No %s — aggregate_plots will rely on JSON paths only.", experiment_results_src
            )

        aggregate_plots(
            base_folder=idea_dir,
            model=args.model_agg_plots,
            n_reflections=args.plot_reflections,
        )

        if osp.exists(experiment_results_tmp):
            shutil.rmtree(experiment_results_tmp)
            logger.info("Removed temporary %s", experiment_results_tmp)

    def save_token_tracker() -> None:
        with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
            json.dump(token_tracker.get_summary(), f)
        with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
            json.dump(token_tracker.get_interactions(), f)

    save_token_tracker()

    if args.skip_writeup:
        logger.info("Skip writeup requested — done.")
        return 0

    # ICBINB writeup expects pre-gathered .bib text; normal perform_writeup gathers in-loop.
    citations_text = None
    if args.writeup_type == "icbinb":
        citations_text = gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )

    writeup_success = False
    for attempt in range(args.writeup_retries):
        logger.info("Writeup attempt %s / %s", attempt + 1, args.writeup_retries)
        try:
            if args.writeup_type == "normal":
                # perform_writeup (ICML) does not take precomputed citations_text in this codebase
                writeup_success = perform_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    vlm_model=vlm_resolved,
                    page_limit=8,
                )
            else:
                writeup_success = perform_icbinb_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    vlm_model=vlm_resolved,
                    page_limit=4,
                    citations_text=citations_text,
                )
        except Exception:
            logger.error("Writeup raised:\n%s", traceback.format_exc())
            writeup_success = False

        if writeup_success:
            break

    if not writeup_success:
        logger.warning("Writeup did not complete successfully after all retries.")

    save_token_tracker()

    if args.run_review and not args.skip_writeup:
        from ai_scientist.llm import create_client
        from ai_scientist.perform_llm_review import perform_review, load_paper
        from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
        from ai_scientist.vlm import create_client as create_vlm_client

        def find_pdf_path_for_review(base: str):
            try:
                names = os.listdir(base)
            except OSError:
                return None
            pdf_files = sorted(f for f in names if f.endswith(".pdf"))
            reflection_pdfs = [f for f in pdf_files if "reflection" in f]
            if not reflection_pdfs:
                return None
            import re as _re

            final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
            if final_pdfs:
                return osp.join(base, final_pdfs[0])
            reflection_nums = []
            for f in reflection_pdfs:
                m = _re.search(r"reflection[_.]?(\d+)", f)
                if m:
                    reflection_nums.append((int(m.group(1)), f))
            if reflection_nums:
                highest = max(reflection_nums, key=lambda x: x[0])
                return osp.join(base, highest[1])
            return osp.join(base, reflection_pdfs[0])

        pdf_path = find_pdf_path_for_review(idea_dir)
        if pdf_path and osp.exists(pdf_path):
            logger.info("Reflection PDF: %s", pdf_path)
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)
            vlm_client, vlm_model = create_vlm_client(vlm_resolved)
            review_img_cap_ref = perform_imgs_cap_ref_review(
                vlm_client, vlm_model, pdf_path
            )
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            logger.info("Paper review completed.")
        else:
            logger.info(
                "Skipping review: no reflection PDF in %s", idea_dir
            )

    # Optional cleanup (same as launch_scientist_bfts.py tail)
    logger.info("Start cleaning up child processes (optional)")
    try:
        import psutil  # type: ignore
        import signal

        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.send_signal(signal.SIGTERM)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _, alive = psutil.wait_procs(children, timeout=3)
        for process in alive:
            try:
                process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        logger.warning(
            "psutil not installed — skipped process cleanup. pip install psutil"
        )

    logger.info("after_experiments.py finished.")
    return 0 if writeup_success or args.skip_writeup else 1


if __name__ == "__main__":
    sys.exit(main())
