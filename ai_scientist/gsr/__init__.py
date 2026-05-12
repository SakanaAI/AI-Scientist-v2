"""
GSR (Generate-Select-Refine) integration for AI Scientist v2.

Implements the open-ended Bayesian optimization framework from
"Open-Ended Task Discovery via Bayesian Optimization" (Adachi et al., 2026)
on top of the AI Scientist v2 pipeline. Each "task" is a research idea;
within-task optimization is the BFTS experiment pipeline.
"""
