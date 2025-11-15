#!/usr/bin/env python3
"""
Simple Performance Profiler for vLLM

This module provides a simple profiling system that collects:
- Input: (context_length, current_length) pairs
- Output: Execution times
- Format: [(context_len_i, current_len_i)]_i, t_j

Perfect for building simple performance models.
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import threading

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ProfilingDataPoint:
    """Single profiling data point for a batch."""
    context_lengths: List[int]
    current_lengths: List[int]
    execution_time: float
    timestamp: float
    batch_size: int

    def to_tuple(self) -> Tuple[List[int], List[int], float]:
        """Convert to (context_lengths, current_lengths, execution_time) tuple."""
        return (self.context_lengths, self.current_lengths, self.execution_time)


class SimpleProfiler:
    """
    Simple profiler that collects batch (context_lens, current_lens) -> execution_time data.

    This is designed for building performance models where you need to predict
    execution time based on context length and current generation length.
    """

    def __init__(self, output_file: str = "vllm_simple_profiles.json"):
        """
        Initialize the simple profiler.

        Args:
            output_file: File to save profiling data
        """
        self.output_file = output_file
        self.data_points: List[ProfilingDataPoint] = []
        self._lock = threading.Lock()
        self.start_time = time.time()

        logger.info(f"Simple profiler initialized. Output file: {output_file}")

    def record_execution(self,
                        context_length: int,
                        current_length: int,
                        execution_time: float,
                        request_id: Optional[str] = None,
                        batch_size: Optional[int] = None) -> None:
        """
        Record a single execution data point (for compatibility, just wraps as a batch of 1).
        """
        self.record_execution_batch([context_length], [current_length], execution_time, batch_size=1)

    def record_execution_batch(self,
                              context_lengths: List[int],
                              query_lens: List[int],
                              execution_time: float,
                              request_ids: Optional[List[str]] = None,
                              batch_size: Optional[int] = None) -> None:
        """
        Record execution data for a batch of sequences (records only one data point for the batch).

        Args:
            context_lengths: List of context lengths for each sequence
            query_lens: List of query lengths for each sequence (current generation length)
            execution_time: Total time taken for the batch execution
            request_ids: Optional list of request identifiers (ignored)
            batch_size: Optional batch size
        """
        with self._lock:
            if len(context_lengths) != len(query_lens):
                context_lengths = context_lengths[:len(query_lens)]
                query_lens = query_lens[:len(context_lengths)]
                logger.warning(f"Mismatched lengths: context_lengths={len(context_lengths)}, query_lens={len(query_lens)}")
                # return

            data_point = ProfilingDataPoint(
                context_lengths=list(context_lengths),
                current_lengths=list(query_lens),
                execution_time=execution_time,
                timestamp=time.time(),
                batch_size=batch_size or len(context_lengths)
            )
            self.data_points.append(data_point)
            logger.debug(f"Recorded batch: {len(context_lengths)} sequences, time={execution_time:.4f}s")

    def get_data_for_model(self) -> List[Tuple[List[int], List[int], float]]:
        """
        Get data in format suitable for ML model training.

        Returns:
            List of (context_lengths, current_lengths, execution_time) tuples
        """
        with self._lock:
            return [dp.to_tuple() for dp in self.data_points]

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics of the collected data."""
        with self._lock:
            if not self.data_points:
                return {}

            times = [dp.execution_time for dp in self.data_points]
            batch_sizes = [dp.batch_size for dp in self.data_points]
            total_samples = sum(batch_sizes)
            all_context_lengths = [cl for dp in self.data_points for cl in dp.context_lengths]
            all_current_lengths = [cl for dp in self.data_points for cl in dp.current_lengths]

            return {
                "total_batches": len(self.data_points),
                "total_samples": total_samples,
                "avg_execution_time": sum(times) / len(times),
                "min_execution_time": min(times),
                "max_execution_time": max(times),
                "avg_context_length": sum(all_context_lengths) / len(all_context_lengths) if all_context_lengths else 0,
                "avg_current_length": sum(all_current_lengths) / len(all_current_lengths) if all_current_lengths else 0,
                "uptime": time.time() - self.start_time
            }

    def save_data(self, filename: Optional[str] = None) -> str:
        """
        Save profiling data to JSON file.

        Args:
            filename: Optional filename, defaults to self.output_file

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = self.output_file

        with self._lock:
            data = {
                "metadata": {
                    "timestamp": time.time(),
                    "total_batches": len(self.data_points),
                    "uptime": time.time() - self.start_time
                },
                "data_points": [asdict(dp) for dp in self.data_points]
            }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.data_points)} batch data points to {filename}")
        return filename

    def export_csv(self, filename: Optional[str] = None) -> str:
        """
        Export data to CSV format for easy analysis.

        Args:
            filename: Optional filename, defaults to output_file with .csv extension

        Returns:
            Path to saved CSV file
        """
        if filename is None:
            filename = self.output_file.replace('.json', '.csv')

        import csv

        with self._lock:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['context_lengths', 'current_lengths', 'execution_time', 'timestamp', 'batch_size'])
                # Write data
                for dp in self.data_points:
                    writer.writerow([
                        repr(dp.context_lengths),
                        repr(dp.current_lengths),
                        dp.execution_time,
                        dp.timestamp,
                        dp.batch_size
                    ])

        logger.info(f"Exported {len(self.data_points)} batch data points to CSV: {filename}")
        return filename

    def clear_data(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self.data_points.clear()
            self.start_time = time.time()

        logger.info("Cleared all profiling data")

    def get_data_by_type(self, execution_type: str = "all") -> List[ProfilingDataPoint]:
        """
        Get data filtered by execution type.

        Args:
            execution_type: "all", "prefill", or "decode"

        Returns:
            Filtered list of data points
        """
        with self._lock:
            if execution_type == "all":
                return self.data_points.copy()
            elif execution_type == "prefill":
                # Only batches where all current_lengths are 0
                return [dp for dp in self.data_points if all(cl == 0 for cl in dp.current_lengths)]
            elif execution_type == "decode":
                return [dp for dp in self.data_points if dp.current_length > 0]
            else:
                raise ValueError(f"Unknown execution type: {execution_type}")


# Global profiler instance
_global_simple_profiler: Optional[SimpleProfiler] = None


def get_simple_profiler() -> SimpleProfiler:
    """Get the global simple profiler instance."""
    global _global_simple_profiler
    if _global_simple_profiler is None:
        # Check for environment variable configuration
        import os
        output_file = os.environ.get("VLLM_PROFILER_OUTPUT_FILE", "vllm_simple_profiles.json")
        _global_simple_profiler = SimpleProfiler(output_file=output_file)
        logger.info(f"Auto-initialized profiler with output file: {output_file}")
    return _global_simple_profiler


def set_simple_profiler(profiler: SimpleProfiler) -> None:
    """Set the global simple profiler instance."""
    global _global_simple_profiler
    _global_simple_profiler = profiler


def record_execution(context_lengths: List[int], query_lens: List[int], execution_time: float, **kwargs) -> None:
    """Record execution using the global simple profiler (records only one data point for the batch)."""
    profiler = get_simple_profiler()
    profiler.record_execution_batch(context_lengths, query_lens, execution_time, **kwargs)


def get_model_data() -> List[Tuple[List[int], List[int], float]]:
    """Get data for ML model training using the global profiler."""
    profiler = get_simple_profiler()
    return profiler.get_data_for_model()


def save_data(filename: Optional[str] = None) -> str:
    """Save data using the global profiler."""
    profiler = get_simple_profiler()
    return profiler.save_data(filename)


def export_csv(filename: Optional[str] = None) -> str:
    """Export data to CSV using the global profiler."""
    profiler = get_simple_profiler()
    return profiler.export_csv(filename)
