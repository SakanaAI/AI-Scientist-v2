# AI-Scientist-v3

**Autonomous AI Research Environment for Claude Code**

AI-Scientist-v3 transforms your repository into a research environment where Claude Code acts as an autonomous AI research scientist. It generates ideas, runs experiments, creates publication-quality figures, writes LaTeX papers, and conducts peer reviews — all with configurable human oversight.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/danielxmed/AI-Scientist-v3.git
cd AI-Scientist-v3

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Claude Code (if not already installed)
# See: https://docs.anthropic.com/en/docs/claude-code

# 4. Configure your research topic
# Edit research.yaml with your topic, venue, and constraints

# 5. Start researching
claude
# Then type: /research
```

## How It Works

This is NOT a Python program that calls an LLM. Claude Code IS the researcher. The repository provides:

- **CLAUDE.md** — Research scientist identity and pipeline overview
- **Skills** (`.claude/skills/`) — Each research phase as an invocable workflow
- **Subagents** (`.claude/agents/`) — Specialized workers (experimentalist, reviewer, literature-scout)
- **Methodology** (`methodology/`) — Domain knowledge for each phase
- **Templates** (`templates/`) — LaTeX templates for ICLR and ICML
- **Utilities** (`lib/`) — Minimal Python helpers for compilation, search, execution

## Research Pipeline

| Phase | Skill | Description |
|-------|-------|-------------|
| 1. Ideation | `/ideate` | Generate research ideas, search literature, check novelty |
| 2. Experiments | `/experiment` | Write ML code, run experiments, iterate on results |
| 3. Plotting | `/plot` | Generate publication-quality figures |
| 4. Writing | `/write-paper` | Write LaTeX paper, gather citations, compile PDF |
| 5. Review | `/review-paper` | Structured peer review with scores |
| Full | `/research` | Run all phases with configurable human checkpoints |

## Human-in-the-Loop

Configure checkpoint behavior in `research.yaml`:

```yaml
checkpoints:
  after_ideation: true       # Review ideas before experimenting
  after_experiments: true    # Review results before writing
  after_writing: false       # Auto-proceed to review
  after_review: false        # Auto-finish
```

Set all to `false` for fully autonomous operation, or `true` for maximum control.

## Project Structure

```
AI-Scientist-v3/
├── CLAUDE.md                  # Research environment identity
├── research.yaml              # Research configuration
├── .claude/
│   ├── skills/                # Research phase skills
│   ├── agents/                # Specialized subagents
│   ├── rules/                 # Methodology constraints
│   └── settings.json          # Permissions
├── methodology/               # Domain knowledge guides
├── templates/                 # LaTeX templates
├── lib/                       # Python utilities
├── ideas/                     # Research topic definitions
├── fewshot-examples/          # Example reviews and ideas
└── experiments/               # Output directory (gitignored)
```

## Configuration

### research.yaml

```yaml
topic: "Your research question here"
venue: icbinb           # icbinb (4-page) or icml (8-page)
constraints:
  max_experiment_runtime_hours: 8
  gpu_required: true
  num_seeds: 3
experiment:
  starter_code: "ideas/your_starter.py"   # Optional
  datasets: ["YourDataset"]
```

### Adding a New Research Topic

1. Create a topic description in `ideas/your_topic.md`
2. Optionally create starter code in `ideas/your_topic.py`
3. Optionally pre-generate ideas in `ideas/your_topic.json`
4. Update `research.yaml` to point to your files

## Requirements

- **Claude Code** (latest version)
- **Python 3.10+** with packages in `requirements.txt`
- **LaTeX** (`texlive` with `pdflatex` and `bibtex`)
- **GPU** (optional, for ML experiments)

## Acknowledgments

Based on [AI-Scientist](https://github.com/SakanaAI/AI-Scientist) by Sakana AI. Redesigned as a Claude Code-native environment by Daniel Nobrega Medeiros.
