# AI Scientist v2

Fully autonomous scientific research system that generates ideas, runs experiments via best-first tree search, writes papers, and performs peer review.

## Claude Skills

This project includes Claude Code skills for orchestrating the entire pipeline. **All stages run natively in Claude Code — no external LLM API keys required.** Only `S2_API_KEY` (Semantic Scholar) is recommended for better literature search.

| Skill | Command | Description |
|-------|---------|-------------|
| `/ai-scientist-ideate` | `/ai-scientist-ideate ai_scientist/ideas/topic.md` | Generate research ideas using Claude + Semantic Scholar |
| `/ai-scientist-experiment` | `/ai-scientist-experiment ai_scientist/ideas/topic.json` | Run 4-stage experiments (implement → tune → improve → ablate) |
| `/ai-scientist-writeup` | `/ai-scientist-writeup experiments/dir/` | Generate LaTeX paper from experiment results |
| `/ai-scientist-review` | `/ai-scientist-review experiments/dir/` | Peer review a generated paper (NeurIPS format) |
| `/ai-scientist` | `/ai-scientist ai_scientist/ideas/topic.json` | Run the full pipeline end-to-end |

### API Key Requirements

- **`S2_API_KEY`** — Semantic Scholar API key for literature search (recommended; falls back to web search if absent)
- **No other API keys required** — Claude Code handles all LLM tasks directly (code generation, analysis, writeup, review)

### Legacy Mode (External API Keys)

The original Python scripts (`launch_scientist_bfts.py`) are still available for running the pipeline with external LLM APIs:
- `OPENAI_API_KEY` — For OpenAI models (writeup, review, feedback)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` — For Bedrock (experiment coding)

### Typical Workflow

```
# 1. Generate ideas from a research topic
/ai-scientist-ideate ai_scientist/ideas/my_topic.md --max-num-generations 5

# 2. Run full pipeline on idea #0
/ai-scientist ai_scientist/ideas/my_topic.json --idea_idx 0 --load_code

# Or run stages independently:
/ai-scientist-experiment ai_scientist/ideas/my_topic.json --idea_idx 0
/ai-scientist-writeup experiments/2025-01-01_my_idea_attempt_0/
/ai-scientist-review experiments/2025-01-01_my_idea_attempt_0/

# Skip experiments if results already exist:
/ai-scientist ai_scientist/ideas/my_topic.json --skip_experiments --idea_idx 0
```

## Key Files

- `launch_scientist_bfts.py` — Legacy entry point (requires external API keys)
- `bfts_config.yaml` — Experiment configuration (for legacy mode)
- `ai_scientist/ideas/` — Research topic files and generated ideas
- `experiments/` — Output directory for results
- `.claude/skills/` — Claude Code skill definitions
