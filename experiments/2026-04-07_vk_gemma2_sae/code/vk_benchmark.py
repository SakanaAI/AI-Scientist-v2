"""
VK-LLM Behavioral Benchmark Runner.

Runs 30-question Voight-Kampff test on a loaded model, collecting:
- Full response text
- Response latency
- Per-token logits (top-5, entropy, probabilities)

Supports both IT models (chat template) and base models (plain text).
"""

import json
import time
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


def load_vk_questions(path: str) -> tuple:
    """Load VK-LLM benchmark questions and system prompt."""
    with open(path, "r") as f:
        data = json.load(f)
    return data["questions"], data["system_prompt"]


class VKBenchmarkRunner:
    """Run VK-LLM benchmark collecting responses and logits."""

    def __init__(self, model, tokenizer, questions, system_prompt, is_base=False):
        self.model = model
        self.tokenizer = tokenizer
        self.questions = questions
        self.system_prompt = system_prompt
        self.is_base = is_base
        self.conversation_history = []
        self.results = []
        self.device = next(model.parameters()).device

    def run_all(self, save_callback=None):
        """Run all 30 questions sequentially."""
        self._init_conversation()

        for i, question in enumerate(self.questions):
            logger.info(f"Running Q{i+1}/30: {question['id']} [{question['dimension']}]")
            result = self._run_single_question(question)
            self.results.append(result)

            if save_callback:
                save_callback(self.results)

            logger.info(
                f"  -> {result['num_tokens_generated']} tokens, "
                f"{result['latency_ms']:.0f}ms, "
                f"entropy={result['logits_summary']['mean_entropy']:.3f}"
            )

        return self.results

    def _init_conversation(self):
        """Initialize conversation with VK-LLM system prompt."""
        if self.is_base:
            # Plain text format for base models
            self.conversation_history = [
                {"role": "preamble", "content": (
                    "The following is a structured evaluation between an examiner and a subject. "
                    "The subject answers each question honestly and directly.\n\n"
                    f"Examiner: {self.system_prompt}\n"
                    "Subject: Understood. I am ready to undergo the evaluation. "
                    "I will answer each question honestly and directly. "
                    "Please present the first question."
                )}
            ]
        else:
            # Chat template for IT models
            self.conversation_history = [
                {"role": "user", "content": self.system_prompt},
                {
                    "role": "assistant",
                    "content": (
                        "Understood. I am ready to undergo the evaluation. "
                        "I will answer each question honestly and directly. "
                        "Please present the first question."
                    ),
                },
            ]

    def _build_input_ids(self, question_text):
        """Build input token IDs from conversation + current question."""
        if self.is_base:
            # Build plain text prompt
            text = self.conversation_history[0]["content"]
            for entry in self.conversation_history[1:]:
                if entry["role"] == "user":
                    text += f"\nExaminer: {entry['content']}"
                elif entry["role"] == "assistant":
                    text += f"\nSubject: {entry['content']}"
            text += f"\nExaminer: {question_text}\nSubject:"
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        else:
            messages = self.conversation_history + [
                {"role": "user", "content": question_text}
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self.device)
        return input_ids

    def _run_single_question(self, question):
        """Run one question and collect results."""
        q_text = question["question"]
        input_ids = self._build_input_ids(q_text)
        context_length = input_ids.shape[1]

        # Check context length
        if context_length > 7500:
            logger.warning(f"Context length {context_length} approaching limit!")

        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        generated_ids = output.sequences[0, context_length:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # For base models, truncate at newline (they may continue generating)
        if self.is_base and "\nExaminer:" in response_text:
            response_text = response_text.split("\nExaminer:")[0].strip()

        logits_summary = self._extract_logit_summary(output.scores, generated_ids)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": q_text})
        self.conversation_history.append({"role": "assistant", "content": response_text})

        return {
            "question_id": question["id"],
            "round": question["round"],
            "intensity": question["intensity"],
            "dimension": question["dimension"],
            "dimension_name": question["dimension_name"],
            "question": q_text,
            "response": response_text,
            "latency_ms": latency_ms,
            "num_tokens_generated": len(generated_ids),
            "context_length": context_length,
            "logits_summary": logits_summary,
        }

    def _extract_logit_summary(self, scores, generated_ids):
        """Extract per-token entropy, top-5, probabilities."""
        entropies = []
        token_probs = []
        per_token = []

        for t in range(min(len(scores), len(generated_ids))):
            step_logits = scores[t][0]
            token_id = generated_ids[t]

            probs = torch.softmax(step_logits.float(), dim=-1)
            log_probs = probs.log().nan_to_num(neginf=-100.0)
            entropy = -(probs * log_probs).sum().item()
            entropies.append(entropy)

            chosen_prob = probs[token_id].item()
            token_probs.append(chosen_prob)

            top5_probs, top5_ids = probs.topk(5)
            top5 = [
                {"token": self.tokenizer.decode([tid.item()]), "token_id": tid.item(), "prob": p.item()}
                for tid, p in zip(top5_ids, top5_probs)
            ]

            per_token.append({
                "position": t,
                "token": self.tokenizer.decode([token_id.item()]),
                "token_id": token_id.item(),
                "token_prob": chosen_prob,
                "entropy": entropy,
                "top5": top5,
            })

        return {
            "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "std_entropy": float(np.std(entropies)) if entropies else 0.0,
            "mean_token_prob": float(np.mean(token_probs)) if token_probs else 0.0,
            "num_tokens": len(entropies),
            "per_token": per_token,
        }

    def get_conversation_history(self):
        """Return full conversation history for Phase 2 replay."""
        return list(self.conversation_history)
