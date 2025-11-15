"""
vLLM Performance Profiling Module

This module provides comprehensive performance profiling capabilities for vLLM,
enabling automatic collection of execution time data for building performance models.

The module automatically intercepts key execution points to collect:
- Request context length and generation length
- Execution time for each step (prefill, decode, scheduling, etc.)
- Batch sizes and token counts
- Memory usage and GPU utilization
- Scheduling decisions and timing

Example usage:
    from vllm.profiling.performance_profiler import get_profiler
    
    # Profiling happens automatically when using vLLM
    from vllm import LLM
    llm = LLM("model_name")
    outputs = llm.generate(["prompt"])
    
    # Access profiling data
    profiler = get_profiler()
    summary = profiler.get_performance_summary()
    print(f"Total requests: {summary['total_requests']}")
    
    # Export data for ML training
    profiler.export_performance_model_data("training_data.json")
"""

from .performance_profiler import (
    PerformanceProfiler,
    ProfilingEvent,
    PerformanceProfile,
    get_profiler,
    set_profiler,
    start_request_profile,
    record_event,
    end_request_profile,
    save_profiles,
    export_performance_model_data,
)

from .simple_profiler import (
    SimpleProfiler,
    ProfilingDataPoint,
    get_simple_profiler,
    set_simple_profiler,
    record_execution,
    get_model_data,
    save_data,
    export_csv,
)

__all__ = [
    "PerformanceProfiler",
    "ProfilingEvent", 
    "PerformanceProfile",
    "get_profiler",
    "set_profiler",
    "start_request_profile",
    "record_event",
    "end_request_profile",
    "save_profiles",
    "export_performance_model_data",
    "SimpleProfiler",
    "ProfilingDataPoint",
    "get_simple_profiler",
    "set_simple_profiler",
    "record_execution",
    "get_model_data",
    "save_data",
    "export_csv",
]

__version__ = "1.0.0"
