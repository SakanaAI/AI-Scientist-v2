import os
import tempfile
import unittest

import numpy as np

from ai_scientist.utils.npy_manifest import (
    build_numpy_data_manifest,
    collect_npy_paths,
)


class NpyManifestTest(unittest.TestCase):
    def test_collects_nested_paths_once(self):
        summaries = {
            "BASELINE_SUMMARY": {
                "best node": {"exp_results_npy_files": ["runs/a.npy", "runs/b.npy"]}
            },
            "ABLATION_SUMMARY": [
                {
                    "ablation_name": "no augmentation",
                    "exp_results_npy_files": ["runs/a.npy"],
                }
            ],
        }

        self.assertEqual(collect_npy_paths(summaries), ["runs/a.npy", "runs/b.npy"])

    def test_manifest_describes_keys_and_value_shapes(self):
        with tempfile.TemporaryDirectory() as base_folder:
            run_dir = os.path.join(base_folder, "runs")
            os.makedirs(run_dir)
            np.save(
                os.path.join(run_dir, "experiment_data.npy"),
                {
                    "accuracy": [0.75, 0.8],
                    "config": {"seed": 1},
                    "embedding": np.zeros((2, 3)),
                },
                allow_pickle=True,
            )
            summaries = {
                "RESEARCH_SUMMARY": {
                    "best node": {"exp_results_npy_files": ["runs/experiment_data.npy"]}
                }
            }

            manifest = build_numpy_data_manifest(base_folder, summaries)
            data = manifest["runs/experiment_data.npy"]

        self.assertEqual(data["top_level_type"], "dict")
        self.assertEqual(data["top_level_keys"]["accuracy"]["type"], "list")
        self.assertEqual(data["top_level_keys"]["config"]["keys"], ["seed"])
        self.assertEqual(data["top_level_keys"]["embedding"]["shape"], [2, 3])

    def test_manifest_rejects_paths_outside_experiment_folder(self):
        with tempfile.TemporaryDirectory() as base_folder, tempfile.NamedTemporaryFile() as outside_file:
            summaries = {
                "RESEARCH_SUMMARY": {
                    "best node": {"exp_results_npy_files": [outside_file.name]}
                }
            }

            manifest = build_numpy_data_manifest(base_folder, summaries)

        self.assertIn("outside", manifest[outside_file.name]["error"])


if __name__ == "__main__":
    unittest.main()
