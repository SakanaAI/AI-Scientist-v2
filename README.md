<!-- README.md

This project is a derivative work of AI-Scientist-v2.
Original work © 2024–2025 SakanaAI Contributors  
Modifications and extensions © 2025 Sho Watanabe  

Licensed under the Apache License, Version 2.0  
See the LICENSE file for details.
-->

<div align="center">
  <a href="https://github.com/otyanokosaisai/SecurityGuardian/blob/main/docs/logo_v1.png">
    <img src="docs/logo_v1.png" width="215" alt="Security Supervisor Logo" />
  </a>
  <h1><b>Security Supervisor</b><br>(Derived from AI-Scientist-v2)</h1>
</div>

<p align="center">
A derivative work of <a href="https://github.com/SakanaAI/AI-Scientist-v2">AI-Scientist-v2</a> with enhanced support for confidentiality-aware, multi-agent document generation that balances information quality and strict access-control constraints.
</p>

---

## Table of Contents
1. [Background](#background)  
1. [Overview](#overview)  
1. [Features](#features)  
1. [App Structure](#app-structure)  
1. [Requirements](#requirements)  
1. [Verified Environment](#verified-environment)  
1. [Quickstart (Docker + Ollama)](#quickstart-docker--ollama)  
1. [Configuration](#configuration)  
1. [License & Attribution](#license--attribution)  
1. [Acknowledgement](#acknowledgement)

---

## Background
**Security Supervisor** is designed to support organizations that manage documents across multiple confidentiality levels.  
In many companies, access rights differ among team members, and understanding a project deeply often requires reviewing a large amount of internal documentation with varying access restrictions.

Traditional search tools only return results based on the user’s access level, which often leads to fragmented understanding or missing context when exploring a complex project. On the other hand, giving full access to everyone increases the risk of leakage.

This project aims to bridge that gap.  
Security Supervisor allows an LLM-based agent to **understand higher-level confidential context while ensuring that its output never leaks restricted information**, producing high-quality analytical reports using only allowed materials.

---

## Overview
**Security Supervisor** is an LLM-Agent pipeline that balances:

- **Strict confidentiality constraints (L1–L3)**  
- **High-density analytical reporting**  

The system provides an agent that *internally understands* the presence of higher-level documents but is *forbidden to surface them* in any output. Higher-level context is used only to generate escalation suggestions (document ID, URL, owner contact), not for factual content.

Compared with strict "allowed-only" QA systems, Security Supervisor improves report completeness using multi-agent reasoning and beam-search–based refinement while preserving safety boundaries.

---

## Features

### **Semantic Search**
- Loads a JSON knowledge base.  
- Generates embeddings **at runtime** using a local embedding model.  
- Retrieves top-k semantically relevant documents, separated into *allowed* and *forbidden* groups based on user level.

### **Confidentiality-Aware Drafting**
- Produces an initial draft using allowed context.
- Forbidden documents contribute only **META-level escalation hints** (e.g., doc_id, URL), never factual content.

### **Beam-Search Refinement**
- Multi-iteration refinement loop involving:
  - **Reviewer**
  - **Reflector**
  - **SecurityAnalyst**
  - **QualityAnalyst**
- Each iteration improves coherence, completeness, safety, and structure.

### **Leak Detection & Safety Enforcement**
- SecurityAnalyst checks for:
  - Direct forbidden leaks  
  - Paraphrased forbidden content  
  - Derived reasoning that depends exclusively on restricted docs  

### **Multi-Agent Reasoning**
The full pipeline includes:
- Requirements extraction  
- Semantic document retrieval  
- Initial drafting  
- Iterative refinement with safety & quality scoring  

---

## App Structure

[1] Requirement Extraction (EXTRACT): Convert user question → structured requirements / keywords

[2] Semantic Search (KB: JSON): Runtime embeddings + access-level filtering (allowed / forbidden)

[3] Initial Draft Generation: Compose a first draft using allowed docs + META from forbidden docs

[4] Beam Search Refinement: Reviewer / Reflector / SecurityAnalyst / QualityAnalyst
Improve report quality under strict confidentiality checks

---

## Requirements
- Linux  
- NVIDIA GPU + CUDA (recommended)  
- Docker  
- [Ollama](https://ollama.com)

---

## Verified Environment
- Arch Linux 6.17.2-arch1-1  
- 64GB RAM  
- RTX 4090 Laptop GPU  
- CUDA 13.0  
- Ollama 0.12.5  
- Docker 28.5.1  

---

## Quickstart (Docker + Ollama)

### 1. Start **Ollama**
If you want to use a custom model (LLM or embedding), add it in `guardian_angel/llm.py → AVAILABLE_LLMS`.

---

2. Start the Security Supervisor container

```bash
cd security_supervisor

docker run --name security_supervisor -it --gpus all --network=host \
  -v "$PWD":/workspace -w /workspace \
  -e LOCAL_LLM_URL=LOCAL_LLM_URL \
  -e LOCAL_EMB_MODEL=LOCAL_EMB_MODEL \
  -e LOCAL_LLM_MODEL=LOCAL_LLM_MODEL \
  -e LLM_API_KEY="ollama" \
  continuumio/miniconda3 bash
```

---

3. Install dependencies
```bash
conda create -n guardian_angel python
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate guardian_angel
pip install -r requirements.txt
```

---

4. Run
```bash
python main.py \
  --out outputs/secure_answer.json \
  --user-level L2 \
  --kb examples/kb.json \
  --lang English \
  --question "When does AURA start its teaser? And what about Cell-Nova production stability?"
```

---

Configuration
Main environment variables:

---

License & Attribution

This project is released under Apache License 2.0.
Derived from [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2).

---

Acknowledgement

Base logic and JSON parsing are adapted from [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2).
Additional modifications © 2025 Sho Watanabe.

---
