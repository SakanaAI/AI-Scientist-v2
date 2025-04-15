<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist_v2/blob/main/docs/logo_v1.jpg">
    <img src="docs/logo_v1.png" width="215" alt="AI Scientist v2 Logo" />
  </a>
  <h1>
    <b>The AI Scientist-v2: Workshop-Level Automated</b><br>
    <b>Scientific Discovery via Agentic Tree Search</b>
  </h1>
</div>

<p align="center">
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist-first-publication/"> [Blog Post]</a> |
  📂 <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment"> [ICLR2025 Workshop Experiment]</a>
</p>

Fully autonomous scientific research systems are becoming increasingly capable, with AI playing a pivotal role in transforming how scientific discoveries are made.
We are excited to introduce The AI Scientist-v2, a generalized end-to-end agentic system that has generated the first workshop paper written entirely by AI and accepted through peer review.

This system autonomously generates hypotheses, runs experiments, analyzes data, and writes scientific manuscripts. Unlike [its predecessor](https://github.com/SakanaAI/AI-Scientist), the AI Scientist-v2 removes reliance on human-authored templates, generalizes across ML domains, and employs a progressive agentic tree-search guided by an experiment manager agent.

> **Note:**
> The AI Scientist-v2 doesn’t necessarily produce better papers than v1, especially when a strong starting template is available. V1 follows well-defined templates, leading to high success rates, while v2 takes a broader, more exploratory approach with lower success rates. V1 works best for tasks with clear objectives and a solid foundation, whereas v2 is designed for open-ended scientific exploration.

> **Caution!**
> This codebase will execute LLM-written code. There are various risks and challenges associated with this autonomy, including the use of potentially dangerous packages, web access, and potential spawning of processes. Use at your own discretion.

## Table of Contents

1.  <a href="#requirements">Requirements</a>
    *   <a href="#installation">Installation</a>
    *   <a href="#supported-models-and-api-keys">Supported Models and API Keys</a>
2.  <a href="#run-ai-scientist-v2-paper-generation-experiments">Run AI Scientist-v2 Paper Generation Experiments</a>
3.  <a href="#citing-the-ai-scientist-v2">Citing The AI Scientist-v2</a>
4.  <a href="#frequently-asked-questions">Frequently Asked Questions</a>
5.  <a href="#acknowledgement">Acknowledgement</a>

## Requirements

This code is designed to run on Linux with NVIDIA GPUs using CUDA and PyTorch.

### Installation

<pre><code>conda create -n ai_scientist python=3.11
conda activate ai_scientist
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install anaconda::poppler # for pdftotext
conda install conda-forge::chktex # for writeup

pip install -r requirements.txt
</code></pre>

You may also need to install pdflatex like here https://gist.github.com/rain1024/98dd5e2c6c8c28f9ea9d 

### Supported Models and API Keys

#### OpenAI models

By default, this uses the `OPENAI_API_KEY` environment variable. You can see other environment variables in `.env.example`.

#### Claude Models via Bedrock

For Claude models provided by Amazon Bedrock, please install these additional packages:
<pre><code>pip install anthropic&#91;bedrock&#93;
</code></pre>
Next, specify a set of valid [AWS Credentials](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html) and the target [AWS Region](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html) by setting the environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`.

#### Semantic Scholar API (Literature Search)

Our code can also optionally use a Semantic Scholar API Key (`S2_API_KEY`) for higher throughput [if you have one](https://www.semanticscholar.org/product/api), though it should work without it in principle. If you have problems with Semantic Scholar, you can skip the literature search and citation phases of paper generation.

Be sure to provide the key for the model used for your runs, e.g.:
<pre><code>export OPENAI_API_KEY="YOUR KEY HERE"
export S2_API_KEY="YOUR KEY HERE"
</code></pre>

## Run AI Scientist-v2 Paper Generation Experiments

You can specify models for the writeup and review phases using command-line arguments.
The best-first tree search configuration can be found in `bfts_config.yaml`. Make any necessary changes to that file.

Relevant tree search config parameters are:

- `agent` config:
  - Set `num_workers` and `steps`. `steps` is the maximum number of nodes to explore. e.g. If you set `num_workers=3`, `steps=21`, the tree will grow up to 21 nodes, 3 nodes at a time. Other agent params like `k_fold_validation`, `expose_prediction`, and `data_preview` are not used in the current version.
  - `num_seeds`: should be the same as num_workers if num_workers < 3. Otherwise, set it to be 3.

- `search` config:
  - `max_debug_depth`: the agent tries to debug a node this many times until it gives up the whole chain.
  - `debug_prob`: probability of debugging nodes
  - `num_drafts`: the number of root nodes at the beginning (the number of trees to grow) during Stage 1.

An example command to run AI-Scientist-v2 on one of the generated ideas (please take a look at bfts_config.yaml for tree search parameters, which includes claude-3-5-sonnet):
<pre><code>python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/i_cant_believe_its_not_better.json" \
 --load_code \
 --add_dataset_ref \
 --model_writeup o1-preview-2024-09-12 \
 --model_citation gpt-4o-2024-11-20 \
 --model_review gpt-4o-2024-11-20 \
 --model_agg_plots o3-mini-2025-01-31 \
 --num_cite_rounds 20
</code></pre>

Once stage 1 is complete, you should see a timestamped log folder inside the `experiments` directory. Inside that folder, navigate to `"timestamp_ideaname"/logs/0-run` to find `unified_tree_viz.html`.

### Running with Gemini
Set `GEMINI_API_KEY` environment variable to your Gemini API key, and set `LLM_MODEL` to a gemini model like `gemini-2.0-flash-exp`. You can use the following command to run the experiment with the Gemini model:

```bash
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/automated_concept_sae_eval.json" \
 --add_dataset_ref \
 --model_writeup "gemini-2.0-flash-exp" \
 --model_citation "gemini-2.0-flash-exp" \
 --model_review "gemini-2.0-flash-exp" \
 --model_agg_plots "gemini-2.0-flash-exp" \
 --num_cite_rounds 20 \
--config_path "bfts_config_gemini.yaml" 
```
## Citing The AI Scientist-v2

If you use **The AI Scientist-v2** in your research, please cite it as follows:

<pre><code>@misc{aiscientist_v2,
 title = {The {AI} Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
 author = {Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Foerster, Jakob and Clune, Jeff and Ha, David},
 year = {2025},
 month = {April},
 howpublished = {Sakana AI Blog Post. Available online: \url{https://pub.sakana.ai/ai-scientist-v2/paper}},
}
</code></pre>

## Frequently Asked Questions

**Why has a PDF or a review not been generated?**

The AI Scientist-v2 finishes an idea with a success rate that depends on the template, the base foundation model, and the complexity of the idea. The highest success rates are observed with Claude Sonnet 3.5 for experiments.

**What is the cost of each experiment?**

It usually costs about $15–20 per experiment when using Claude Sonnet 3.5. Writing adds another ~$5 when using the default models, as shown in the example above.
Using GPT-4o is strongly recommended for `model_citation` to help lower writing costs.

**How do I run The AI Scientist-v2 for different subject fields?**

You can add a new idea JSON file to the `ai_scientist/ideas/` directory. For reference, take a look at the example: `ai_scientist/ideas/i_cant_believe_its_not_better.json`.

**What if I have problems accessing the Semantic Scholar API?**

We use the Semantic Scholar API to check ideas for novelty and collect citations for the paper write-up. You may be able to skip these phases if you don't have an API key or the API is slow to access.

**Got an error that says "CUDA Out of Memory"**

This usually means the AI Scientist-v2 is trying to load a model that's too large for your GPU.
You can update your idea JSON file to suggest using a smaller model.

## Acknowledgement

The tree search component inside `ai_scientist` is built on top of [AIDE](https://github.com/WecoAI/aideml). We thank the AIDE developers for their valuable contributions and for sharing their work.
