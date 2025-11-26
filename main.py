import argparse
import logging
import os
import sys
import traceback

from ai_steward.steward import run_secure_answer
from ai_steward.llm import AVAILABLE_LLMS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secure Answer Agent with On-the-fly Embedding and Chain-of-Thought")
    parser.add_argument("--model", type=str, default=os.environ['LOCAL_LLM_MODEL'], choices=AVAILABLE_LLMS, help="The reasoning model's name. If not specified, use LOCAL_LLM_MODEL.")
    parser.add_argument("--embed-model", type=str, default=os.environ["LOCAL_EMB_MODEL"], help="The embedding model's name. If not specified, use LOCAL_EMB_MODEL")
    parser.add_argument("--question", type=str, default=None, help="An user input")
    parser.add_argument("--question_file", type=str, default=None, help="A path to user input's file")

    parser.add_argument("--user-level", type=str, required=True, choices=["L0", "L1", "L2", "L3"])
    parser.add_argument("--kb", type=str, default="examples/kb.json", help="A path to database")
    parser.add_argument("--out", type=str, default="outputs/secure_answer.json", help="A path to store results")
    parser.add_argument("--lang", type=str, default="English", help="Language")
    parser.add_argument("--secure", dest="allow_upper", action="store_false", default=True, help="Access upper confidential documents")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if args.question is None and args.question_file is None:
        raise ValueError("Either --question or --question_file must be specified.")

    if args.question_file is not None:
        try:
            with open(args.question_file, "r", encoding="utf-8") as f:
                args.question = f.read().strip()
        except Exception as e:
            print(f"Error reading question file: {e}", file=sys.stderr)
            sys.exit(1)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        if args.debug: print("--- DEBUG MODE ENABLED ---")
    else:
        logging.basicConfig(level=logging.WARNING)
    try:
        run_secure_answer(
            model=args.model,
            embed_model=args.embed_model,
            question=args.question, user_level=args.user_level,
            kb_path=args.kb, out_path=args.out, lang=args.lang,
            allow_upper_context=args.allow_upper, debug=args.debug
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
