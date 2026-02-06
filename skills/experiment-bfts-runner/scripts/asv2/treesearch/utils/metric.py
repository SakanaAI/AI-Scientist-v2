from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any

import numpy as np
from dataclasses_json import DataClassJsonMixin


@dataclass
@total_ordering
class MetricValue_old(DataClassJsonMixin):
    """
    Represents the value of a metric to be optimized, which can be compared to other metric values.
    Comparisons (and max, min) are based on which value is better, not which is larger.
    """

    value: float | int | np.number | np.floating | np.ndarray | dict | None
    maximize: bool | None = field(default=None, kw_only=True)
    name: str | None = field(
        default=None, kw_only=True
    )  # e.g., "accuracy", "loss", "f1_score"
    description: str | None = field(
        default=None, kw_only=True
    )  # e.g., "Classification accuracy on validation set"

    def __post_init__(self):
        if self.value is not None:
            if isinstance(self.value, dict):
                self.value = {k: float(v) for k, v in self.value.items()}
            else:
                assert isinstance(self.value, (float, int, np.number, np.floating))
                self.value = float(self.value)

    def __gt__(self, other) -> bool:
        """True if self is a _better_ (not necessarily larger) metric value than other"""
        if self.value is None:
            return False
        if other.value is None:
            return True

        assert type(self) is type(other) and (self.maximize == other.maximize)

        # For multi-dataset metrics, use mean for comparison
        self_val = (
            np.mean(list(self.value.values()))
            if isinstance(self.value, dict)
            else self.value
        )
        other_val = (
            np.mean(list(other.value.values()))
            if isinstance(other.value, dict)
            else other.value
        )

        if self_val == other_val:
            return False

        comp = self_val > other_val
        return comp if self.maximize else not comp  # type: ignore

    def __eq__(self, other: Any) -> bool:
        return self.value == other.value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.maximize is None:
            opt_dir = "?"
        elif self.maximize:
            opt_dir = "↑"
        else:
            opt_dir = "↓"
        metric_name = f"({self.name})" if self.name else ""
        if isinstance(self.value_npsafe, dict):
            values_str = ", ".join(f"{k}:{v:.4f}" for k, v in self.value_npsafe.items())
            mean_val = np.mean(list(self.value_npsafe.values()))
            return f"Metric{opt_dir}{metric_name}[{values_str}](mean={mean_val:.4f})"
        else:
            return f"Metric{opt_dir}{metric_name}({self.value_npsafe:.4f})"

    @property
    def is_worst(self):
        """True if the metric value is the worst possible value."""
        return self.value is None

    @property
    def value_npsafe(self):
        if self.value is None:
            return float("nan")
        if isinstance(self.value, dict):
            return {
                k: v if v is not None else float("nan") for k, v in self.value.items()
            }
        return self.value

    def get_dataset_value(self, dataset_name: str) -> float | None:
        """Get the metric value for a specific dataset"""
        if isinstance(self.value, dict):
            return self.value.get(dataset_name)
        return None

    def get_mean_value(self) -> float:
        """Get the mean value across all datasets (or single value if not multi-dataset)"""
        if self.value is None:
            return float("nan")
        if isinstance(self.value, dict):
            return float(np.mean(list(self.value.values())))
        return float(self.value)


@dataclass
@total_ordering
class MetricValue(DataClassJsonMixin):
    """
    Represents one or more metric values to be optimized, which can be compared to other metric values.
    Comparisons (and max, min) are based on which value is better, not which is larger.

    The value can be:
    - A single number (float/int)
    - A dictionary in the format:
      {
        "metric_names": [
          {
            "metric_name": str,
            "lower_is_better": bool,
            "description": str,
            "data": [
                {"dataset_name": str, "final_value": float, "best_value": float},
                {"dataset_name": str, "final_value": float, "best_value": float},
                ...
            ]
          },
          ...
        ]
      }
    """

    value: float | int | np.number | np.floating | dict | None
    maximize: bool | None = field(default=None, kw_only=True)
    name: str | None = field(default=None, kw_only=True)
    description: str | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.value is not None:
            if isinstance(self.value, dict):
                # Check if it's the new format with metric_names list
                if "metric_names" in self.value:
                    # New format - validate and convert values to float
                    for metric in self.value["metric_names"]:
                        for data_point in metric["data"]:
                            if data_point["final_value"] is not None:
                                data_point["final_value"] = float(
                                    data_point["final_value"]
                                )
                            if data_point["best_value"] is not None:
                                data_point["best_value"] = float(
                                    data_point["best_value"]
                                )
                else:
                    # Old format - convert to float
                    self.value = {
                        k: float(v) if v is not None else None
                        for k, v in self.value.items()
                    }
            else:
                # Single value case
                assert isinstance(self.value, (float, int, np.number, np.floating))
                self.value = float(self.value)

    def __gt__(self, other) -> bool:
        if self.value is None:
            return False
        if other.value is None:
            return True

        assert type(self) is type(other)

        # Get mean values for comparison
        self_val = self.get_mean_value()
        other_val = other.get_mean_value()

        if self_val == other_val:
            return False

        # Determine if we should maximize or minimize
        should_maximize = self._should_maximize()
        comp = self_val > other_val
        return comp if should_maximize else not comp

    def _should_maximize(self) -> bool:
        """Determine if we should maximize based on the metric format"""
        if isinstance(self.value, dict):
            # New format
            if "metric_names" in self.value:
                # Use the first metric's lower_is_better value
                try:
                    return not self.value["metric_names"][0]["lower_is_better"]
                except Exception as e:
                    print(f"error during metric value: {e}")
            # Old format
            return bool(self.maximize)
        # Single value case
        return bool(self.maximize)

    def __str__(self) -> str:
        if isinstance(self.value, dict):
            # New format with metric_names list
            if "metric_names" in self.value:
                parts = []
                for metric in self.value["metric_names"]:
                    opt_dir = (
                        "↓"
                        if "lower_is_better" in metric and metric["lower_is_better"]
                        else "↑"
                    )
                    try:
                        values_str = ", ".join(
                            f"{d['dataset_name']}:(final={d['final_value']:.4f}, best={d['best_value']:.4f})"
                            for d in metric["data"]
                        )
                    except Exception as e:
                        print(f"error during metric value: {e}")
                        values_str = "None"
                    parts.append(f"{metric['metric_name']}{opt_dir}[{values_str}]")
                return "Metrics(" + "; ".join(parts) + ")"
            # Old format
            opt_dir = "↓" if not self.maximize else "↑"
            values_str = ", ".join(f"{k}:{v:.4f}" for k, v in self.value.items())
            mean_val = np.mean([v for v in self.value.values() if v is not None])
            return f"Metric{opt_dir}({self.name})[{values_str}](mean={mean_val:.4f})"
        # Single value case
        opt_dir = "?" if self.maximize is None else ("↑" if self.maximize else "↓")
        metric_name = f"({self.name})" if self.name else ""
        return f"Metric{opt_dir}{metric_name}({self.value_npsafe:.4f})"

    def __eq__(self, other: Any) -> bool:
        """Compare equality of metric values"""
        if not isinstance(other, MetricValue):
            raise NotImplementedError
        if self.value is None and other.value is None:
            return True
        if self.value is None or other.value is None:
            return False

        # For new format, compare entire dictionaries
        if isinstance(self.value, dict) and isinstance(other.value, dict):
            # If both are new format with metric_names
            if "metric_names" in self.value and "metric_names" in other.value:
                return self.value == other.value
            # If both are old format (no metric_names)
            elif "metric_names" not in self.value and "metric_names" not in other.value:
                return self.value == other.value
            # Mixed formats should not be equal
            return False
        # Single values
        return self.value == other.value

    def __repr__(self) -> str:
        """Return string representation"""
        return str(self)

    @property
    def value_npsafe(self):
        """Return a NaN-safe version of the value"""
        if self.value is None:
            return float("nan")
        if isinstance(self.value, dict):
            # New format with metric_names list
            if "metric_names" in self.value:
                return {
                    "metric_names": [
                        {
                            **metric,
                            "data": [
                                {
                                    **data_point,
                                    "final_value": (
                                        data_point["final_value"]
                                        if data_point["final_value"] is not None
                                        else float("nan")
                                    ),
                                    "best_value": (
                                        data_point["best_value"]
                                        if data_point["best_value"] is not None
                                        else float("nan")
                                    ),
                                }
                                for data_point in metric["data"]
                            ],
                        }
                        for metric in self.value["metric_names"]
                    ]
                }
            # Old format
            return {
                k: v if v is not None else float("nan") for k, v in self.value.items()
            }
        # Single value case
        return self.value if self.value is not None else float("nan")

    def get_mean_value(self) -> float:
        """Get the mean value across all metrics and datasets"""
        if self.value is None:
            return float("nan")
        if isinstance(self.value, dict):
            # New format
            if "metric_names" in self.value:
                all_values = []
                for metric in self.value["metric_names"]:
                    # Use final_value for comparison
                    values = [
                        d["final_value"]
                        for d in metric["data"]
                        if d["final_value"] is not None
                    ]
                    if values:
                        all_values.extend(values)
                return float(np.mean(all_values)) if all_values else float("nan")
            # Old format
            values = [v for v in self.value.values() if v is not None]
            return float(np.mean(values)) if values else float("nan")
        # Single value case
        return float(self.value)


@dataclass
class WorstMetricValue(MetricValue):
    """
    Represents an invalid metric value, e.g. when the agent creates a buggy solution.
    Always compares worse than any valid metric value.
    """

    value: None = None

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
