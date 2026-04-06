# CLAUDE.md -- AI-Scientist-v3

## Repository Overview

This is **AI-Scientist-v3**, a fork of [Sakana AI's AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist_v2). v2 is an autonomous scientific research system that generates hypotheses, runs ML experiments via agentic tree search, writes LaTeX papers, and conducts peer review -- all autonomously.

**v3's mission**: Replace all direct model-provider API calls (OpenAI, Anthropic, Bedrock, etc.) with **coding agent orchestration**. The coding agent (Claude Code, Codex, etc.) uses its own native capabilities instead of raw SDK calls. No API keys needed.

## Architecture

### Pipeline Stages (execution order)

```
1. Ideation (perform_ideation_temp_free.py)
   -> Generates research ideas using LLM + SemanticScholar search
   -> Output: idea JSON with Name, Title, Hypothesis, Experiments, etc.

2. Experiments (treesearch/perform_experiments_bfts_with_agentmanager.py)
   -> Best-First Tree Search: agent writes code, runs it, evaluates results
   -> AgentManager orchestrates multi-stage exploration (stage1-4)
   -> ParallelAgent manages GPU workers for parallel experiment execution
   -> Interpreter executes generated Python in sandboxed subprocesses
   -> Output: experiment logs, .npy data files, plots

3. Plot Aggregation (perform_plotting.py)
   -> Aggregates experiment plots into final paper figures
   -> Output: figures/ directory

4. Paper Writing (perform_writeup.py / perform_icbinb_writeup.py)
   -> Generates LaTeX paper from experiment results + idea
   -> Gathers citations via SemanticScholar
   -> Compiles LaTeX to PDF (pdflatex + bibtex)
   -> Output: PDF paper

5. Review (perform_llm_review.py / perform_vlm_review.py)
   -> LLM reads paper text and scores it (review_text.txt)
   -> VLM reviews figures/captions/references (review_img_cap_ref.json)
   -> Output: review scores and feedback
```

### Two LLM Call Paths (Both Being Replaced)

**Path A -- Pipeline Stage Calls** (`ai_scientist/llm.py` + `ai_scientist/vlm.py`):
```python
# Current v2 pattern:
client, model = create_client("gpt-4o-2024-11-20")  # returns SDK client
content, history = get_response_from_llm(prompt, client, model, system_msg)
# VLM variant in vlm.py adds base64 image encoding

# v3 target:
agent, model = create_client("agent")  # returns AgentBackend handle
content, history = get_response_from_llm(prompt, agent, model, system_msg)  # routes through agent
```

Consumers: `launch_scientist_bfts.py`, `perform_ideation_temp_free.py`, `perform_writeup.py`, `perform_icbinb_writeup.py`, `perform_llm_review.py`, `perform_vlm_review.py`, `perform_plotting.py`, `treesearch/log_summarization.py`

**Path B -- Tree Search Agent Calls** (`ai_scientist/treesearch/backend/`):
```python
# Current v2 pattern:
output = query(system_message, user_message, model="gpt-4o", temperature=0.7, func_spec=my_spec)
# Dispatches to backend_openai.py or backend_anthropic.py

# v3 target:
output = query(system_message, user_message, model="agent", temperature=0.7, func_spec=my_spec)
# Routes through backend_agent.py -> agent_backend.py
```

Consumers: `treesearch/parallel_agent.py`, `treesearch/agent_manager.py`, `treesearch/journal.py`, `treesearch/journal2report.py`

### Key Classes

- **`AgentManager`** (`treesearch/agent_manager.py`): Orchestrates multi-stage experiments. Uses `FunctionSpec` for structured LLM output (stage config, progress evaluation, completion evaluation).
- **`ParallelAgent`** (`treesearch/parallel_agent.py`): Manages parallel GPU workers. Generates code, executes it, evaluates results. The core BFTS loop.
- **`Journal` / `Node`** (`treesearch/journal.py`): Tree data structure tracking all experiment attempts. Nodes represent code solutions with metrics.
- **`Interpreter`** (`treesearch/interpreter.py`): Sandboxed Python execution. Runs agent-generated code in subprocesses with timeout. **This stays mostly unchanged in v3.**
- **`FunctionSpec`** (`treesearch/backend/utils.py`): Defines JSON schemas for structured output. In v3, schemas become prompt templates instead of API tool_choice params.
- **`TokenTracker`** (`utils/token_tracker.py`): Global singleton tracking API usage. Being adapted for agent-based cost tracking.

## Key Files

| File | Purpose | v3 Status |
|------|---------|-----------|
| `launch_scientist_bfts.py` | Main entry point / orchestrator | Needs CLI arg updates |
| `ai_scientist/llm.py` | LLM abstraction (Path A) | Being replaced |
| `ai_scientist/vlm.py` | Vision LLM abstraction | Being replaced |
| `ai_scientist/agent_backend.py` | **NEW** -- Unified agent interface | To be created |
| `ai_scientist/agent_utils.py` | **NEW** -- Agent helper utilities | To be created |
| `ai_scientist/treesearch/backend/__init__.py` | Backend router | Being updated |
| `ai_scientist/treesearch/backend/backend_openai.py` | OpenAI backend | Being replaced |
| `ai_scientist/treesearch/backend/backend_anthropic.py` | Anthropic backend | Being replaced |
| `ai_scientist/treesearch/backend/backend_agent.py` | **NEW** -- Agent backend | To be created |
| `ai_scientist/treesearch/parallel_agent.py` | BFTS parallel agent (~2300 lines) | Consumer, minimal changes |
| `ai_scientist/treesearch/agent_manager.py` | Experiment stage manager | Consumer, minimal changes |
| `ai_scientist/treesearch/interpreter.py` | Code execution sandbox | Stays as-is |
| `ai_scientist/tools/semantic_scholar.py` | Paper search via S2 API | Being replaced with agent search |
| `ai_scientist/utils/token_tracker.py` | Cost/usage tracking | Being adapted |
| `bfts_config.yaml` | Tree search configuration | Model settings being updated |

## Configuration

### bfts_config.yaml
Controls the tree search agent:
- `agent.code.model` / `agent.feedback.model` -- LLM models for code gen and feedback
- `agent.stages.stage{1-4}_max_iters` -- iterations per experiment stage
- `agent.num_workers` -- parallel GPU workers (currently 4)
- `agent.search.*` -- tree search parameters (debug depth, draft count)
- `report.model` -- model for journal-to-report conversion
- `exec.timeout` -- code execution timeout (3600s)

### Data Flow
```
ideas/*.json          -> Idea definitions (Name, Title, Hypothesis, Experiments)
ideas/*.py            -> Optional starter code for experiments
bfts_config.yaml      -> Tree search parameters
experiments/<date>/   -> Output: logs, results, plots, paper PDF, reviews
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python launch_scientist_bfts.py --load_ideas ai_scientist/ideas/i_cant_believe_its_not_better.json --idea_idx 0

# Run experiments only
python launch_scientist_bfts.py --load_ideas ai_scientist/ideas/i_cant_believe_its_not_better.json --skip_writeup --skip_review

# Run tests
python -m pytest tests/ -v
```

## v2 -> v3 Transformation Status

Track progress in `.ralph/fix_plan.md`. High-level phases:
- Phase 0: Foundation (agent_backend.py) -- **COMPLETE**
- Phase 1: Tree search backend replacement -- **Not started**
- Phase 2: llm.py / vlm.py replacement -- **Not started**
- Phase 3: Pipeline stage updates -- **Not started**
- Phase 4: External tool replacement -- **Not started**
- Phase 5: Token tracking adaptation -- **Not started**
- Phase 6: Config cleanup -- **Not started**
