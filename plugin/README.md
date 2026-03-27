# AI Scientist v2 — Claude Code Plugin

A Claude Code / Claude Co-Work plugin for autonomous scientific research. Generates ideas, runs ML experiments, writes LaTeX papers, and performs peer review — all natively within Claude Code.

## Installation

### From GitHub

```
/plugin marketplace add strelkon/AI-Scientist-v2
/plugin install ai-scientist@strelkon-AI-Scientist-v2
```

### From Local Path (development)

```bash
claude --plugin-dir ./plugin
```

## Setup

When you install the plugin, you'll be prompted for:

1. **`PROJECT_DIR`** (required): Path to your AI-Scientist-v2 repository checkout
2. **`S2_API_KEY`** (recommended): Semantic Scholar API key for literature search — get one at https://www.semanticscholar.org/product/api

### System Requirements

- Python 3.10+ with PyTorch (CUDA recommended)
- LaTeX (`pdflatex`, `bibtex`, `chktex`) for paper generation
- Project dependencies: `pip install -r requirements.txt`

## Skills

| Skill | Command | Description |
|-------|---------|-------------|
| **pipeline** | `/ai-scientist:pipeline topic.md` | Full end-to-end pipeline |
| **ideate** | `/ai-scientist:ideate topic.md` | Generate research ideas |
| **experiment** | `/ai-scientist:experiment ideas.json` | Run 4-stage experiments |
| **writeup** | `/ai-scientist:writeup experiments/dir/` | Generate LaTeX paper |
| **review** | `/ai-scientist:review experiments/dir/` | NeurIPS-format peer review |

## Usage

```bash
# Generate ideas from a research topic
/ai-scientist:ideate ai_scientist/ideas/my_topic.md --max-num-generations 5

# Run full pipeline on idea #0
/ai-scientist:pipeline ai_scientist/ideas/my_topic.json --idea_idx 0

# Run stages independently
/ai-scientist:experiment ai_scientist/ideas/my_topic.json --idea_idx 0
/ai-scientist:writeup experiments/2025-01-01_my_idea_attempt_0/
/ai-scientist:review experiments/2025-01-01_my_idea_attempt_0/

# Skip experiments if results already exist
/ai-scientist:pipeline ideas.json --skip_experiments --idea_idx 0
```

## API Keys

- **No external LLM API keys required** — Claude Code handles all LLM tasks
- **`S2_API_KEY`** recommended for Semantic Scholar (falls back to web search)
- The original Python scripts (`launch_scientist_bfts.py`) remain available as legacy mode if you prefer using external LLM APIs

## License

Apache-2.0 (same as AI-Scientist-v2)
