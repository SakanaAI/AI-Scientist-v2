# Running AI Scientist-v2 on Apple Silicon

The original AI Scientist-v2 codebase is optimized for Linux and NVIDIA GPUs (CUDA). However, you can run it on macOS with Apple Silicon by adapting the installation steps and configuration.

## 1. Prerequisites

### Miniforge (Conda)

You will need Miniforge, the community-driven distribution of Conda optimized for Apple Silicon.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

## 2. Installation

### Step 1: Set up Python Environment

The project requires **Python 3.11**.

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
```

### Step 2: Install System Tools

You need `poppler` (for PDF processing) and `chktex` (for LaTeX linting).

```bash
conda install -c conda-forge poppler chktex
```

### Step 3: Install Python Dependencies

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 3. Configuration & API Keys

See [Supported Models and API Keys](README.md#supported-models-and-api-keys) in the main README for detailed information on configuring API keys for OpenAI, Gemini, Claude (AWS Bedrock), and Semantic Scholar.

## 4. Running Experiments

Once installation is complete, follow the [Generate Research Ideas](README.md#generate-research-ideas) and [Run AI Scientist-v2 Paper Generation Experiments](README.md#run-ai-scientist-v2-paper-generation-experiments) sections in the main README.

## 5. Troubleshooting

- **"Torch not compiled with CUDA enabled"**: This means a script is explicitly trying to use `cuda`. The patched `parallel_agent.py` should prevent this, but if it happens, check the generated code in `experiments/`.
