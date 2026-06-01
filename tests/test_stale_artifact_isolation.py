import os
import shutil
import sys
import unittest
from pathlib import Path

# Reconfigure stdout to prevent CP1252 print crashes on Windows when printing metric arrows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Force LLM review mock query
def mock_query(system_message, **kwargs):
    if system_message is None:
        return {"valid_plots_received": True, "plot_analyses": [], "vlm_feedback_summary": []}
        
    intro = system_message.get("Introduction", "")
    if "evaluate the output" in intro:
        code_content = system_message.get("Implementation", "")
        # Node A fails so its files are left behind
        if "node_a" in code_content:
            return {"summary": "Node A execution has bugs.", "is_bug": True}
        # Node B succeeds but generated no data
        return {"summary": "Node B execution succeeded.", "is_bug": False}
    else:
        # Metrics parsing
        return {
            "summary": "Metrics parsed.",
            "is_bug": False, 
            "valid_metrics_received": True, 
            "metric_names": [{
                "metric_name": "val_loss", 
                "lower_is_better": True,
                "data": [{"dataset_name": "dataset1", "final_value": 99.0, "best_value": 99.0}]
            }]
        }

# Inject mock_query into backend BEFORE other imports
import ai_scientist.treesearch.backend as backend
backend.query = mock_query

from ai_scientist.treesearch.parallel_agent import ParallelAgent, MinimalAgent
from ai_scientist.treesearch.journal import Node, Journal
import ai_scientist.treesearch.parallel_agent as pa
import ai_scientist.treesearch.journal as j

# Ensure all namespaces are patched
pa.query = mock_query
j.query = mock_query

# Also mock summary generation to avoid calling the query function at all
Journal.generate_summary = lambda *args, **kwargs: "Mock summary"

# MOCK CONFIGURATIONS
class MockExecCfg:
    timeout = 30
    format_tb_ipython = False
    agent_file_name = "runfile.py"

class MockAgentCfg:
    class MockFeedbackCfg:
        model = "mock"
        temp = 0.0
    feedback = MockFeedbackCfg()
    num_workers = 1
    k_fold_validation = 1
    
    class MockOtherCfg:
        model = "mock"
        temp = 0.0
    code = MockOtherCfg()
    summary = MockOtherCfg()
    select_node = MockOtherCfg()
    
    class MockMultiSeedCfg:
        num_seeds = 1
    multi_seed_eval = MockMultiSeedCfg()

    def get(self, key, default=None):
        return getattr(self, key, default)

class MockConfig:
    workspace_dir = "test_stale_artifact_isolation_workspace"
    exec = MockExecCfg()
    agent = MockAgentCfg()
    experiment = None
    
    def get(self, key, default=None):
        return getattr(self, key, default)

class MockAgent(ParallelAgent):
    def __init__(self, *args, **kwargs):
        self.num_workers = kwargs.pop('num_workers', 1)
        super().__init__(*args, **kwargs)
        
    def _define_global_metrics(self):
        return [{"metric_name": "val_loss", "lower_is_better": True}]

def mock_plan_and_code(self, prompt, **kwargs):
    # Metrics parser script loading experiment_data.npy
    parse_code = """
import numpy as np
import os
try:
    working_dir = os.path.join(os.getcwd(), 'working')
    data = np.load(os.path.join(working_dir, 'experiment_data.npy'), allow_pickle=True).item()
    val = data.get('dataset1', {}).get('val_loss', 99.0)
    print("DATASET: dataset1")
    print("METRIC: val_loss")
    print(f"VALUE: {val}")
except Exception as e:
    raise e
"""
    return "Mock plan", parse_code

MinimalAgent.plan_and_code_query = mock_plan_and_code
MinimalAgent._improve = lambda self, node: Node(code=node.code, plan=node.plan)

# Override VLM plot analysis to prevent nodes from remaining None in good_nodes
def mock_analyze_plots_with_vlm(self, node):
    node.is_buggy_plots = False

MinimalAgent._analyze_plots_with_vlm = mock_analyze_plots_with_vlm

class TestStaleArtifactIsolation(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        shutil.rmtree(self.cfg.workspace_dir, ignore_errors=True)
        os.makedirs(self.cfg.workspace_dir, exist_ok=True)
        self.journal = Journal()
        self.agent = MockAgent(
            task_desc="Stale Isolation Test",
            cfg=self.cfg,
            journal=self.journal,
            stage_name="improve",
            num_workers=1
        )

    def tearDown(self):
        # Shutdown agent executor to release workers
        if hasattr(self, 'agent') and self.agent:
            self.agent.cleanup()
        shutil.rmtree(self.cfg.workspace_dir, ignore_errors=True)

    def test_stale_artifact_isolation(self):
        node_a_code = """# node_a
import numpy as np
import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
data = {'dataset1': {'val_loss': 99.0}}
np.save(os.path.join(working_dir, 'experiment_data.npy'), data)
"""
        node_b_code = """# node_b
import os
# Silently exits without generating data
print("Node B executed successfully but produced no data.")
"""
        node_a = Node(code=node_a_code)
        node_b = Node(code=node_b_code)

        # Run Node A (fails and leaves experiment_data.npy behind)
        self.agent._select_parallel_nodes = lambda: [node_a]
        self.agent.step(lambda e, p: None)
        
        # Verify Node A was marked buggy (as expected under mock_query)
        node_a_res = self.journal.nodes[0]
        self.assertTrue(node_a_res.is_buggy, "Node A should be evaluated as buggy")

        # Run Node B on the same persistent worker process workspace
        self.agent._select_parallel_nodes = lambda: [node_b]
        self.agent.step(lambda e, p: None)

        # Retrieve Node B result
        node_b_res = self.journal.nodes[1]

        # Node B should not have valid metrics, because it produced no data.
        # Since Node A's working dir was wiped before Node B ran, Node B must NOT find experiment_data.npy.
        # Thus, its metrics parsing should not be run, leaving metric value as None or marked buggy.
        metric_val = node_b_res.metric.value if node_b_res.metric else None
        self.assertTrue(
            metric_val is None or node_b_res.is_buggy,
            f"FAIL: Node B inherited stale metric: {node_b_res.metric}"
        )
        if metric_val is not None:
            # If a metric did get created somehow (e.g. from worst metric value), ensure it is not the stale value
            self.assertTrue(
                node_b_res.is_buggy,
                "Node B should be buggy if metrics were parsed in any invalid way"
            )

if __name__ == "__main__":
    unittest.main()
