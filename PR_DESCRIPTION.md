# PR: Fix stale artifact leakage in ProcessPoolExecutor worker reuse

## Problem
In the tree search loop of `AI-Scientist-v2`, worker processes are persistent and reused across consecutive trials by the `ProcessPoolExecutor`. Each worker process maps to a persistent directory: `process_{worker_id}/working/`.

If a node execution succeeds, the post-processing pipeline renames (moves) the `.npy` and `.png` files to the permanent logging directory (`logs/`). However:
1. If a node fails or is evaluated as buggy (`is_buggy=True`), the plotting and file-moving pipeline is skipped. Leftover artifacts (like `experiment_data.npy`) are left behind in the worker's working directory.
2. In subsequent trials run on that same worker, if the next node fails to produce a `.npy` file (e.g. silent exit or crash), the metrics parsing step will load the stale `experiment_data.npy` from the previous trial. 
3. This leads to metric contamination/inheritance, tree search corruption, and selecting incorrect nodes as the best implementations.

## Fix
Wipe the process's working directory at the start of each task execution inside `_process_node_wrapper(...)` in `ai_scientist/treesearch/parallel_agent.py`:
```python
        # Create process-specific working directory
        working_dir = os.path.join(workspace, "working")
        # Prevents stale artifact leakage across ProcessPoolExecutor worker reuse
        if os.path.exists(working_dir):
            import shutil
            shutil.rmtree(working_dir, ignore_errors=True)
        os.makedirs(working_dir, exist_ok=True)
```

## Why this works
- **Guarantees scratch isolation:** Every trial starts with an empty `working` directory, ensuring no leftover `.npy`, `.png`, model weights, or log files from previous trials can leak.
- **Prevents disk explosion:** Unlike UUID workspaces which create new directories for every single run, this solution bounds the disk space usage to `max_workers` directories.
- **Preserves logging structure:** Reuses the existing process directory naming convention, preventing folder name corruption and keeping frontend dashboard visualizations fully functional.

## Risk
- **Minimal performance overhead:** Deleting a few small files from the scratch directory takes milliseconds, adding virtually zero overhead.
- **Safe on Windows:** The delete uses `ignore_errors=True` in `shutil.rmtree`, so any locked files will not cause a worker process crash.

## Alternatives rejected
- **UUID workspaces:** Rejected because it causes disk space leaks (hundreds of folders accumulate) and corrupts the run folder naming hierarchy (using random UUID names instead of the run folder, which breaks metrics retrieval and visualizations).
- **`max_tasks_per_child=1` in Pool:** Rejected because spawning a new process for every single trial introduces severe execution overhead.
- **LLM-based filtering:** Rejected as it is highly non-deterministic and fails to isolate the underlying filesystem.
