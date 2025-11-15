#!/usr/bin/env python3
"""
Performance Profiler for vLLM

This module provides comprehensive performance profiling capabilities for vLLM,
intercepting key execution points to collect performance data for building
performance models.

The profiler collects:
- Request context length and generation length
- Execution time for each step
- Batch sizes and token counts
- Memory usage and GPU utilization
- Scheduling decisions and timing
"""

import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import threading
import torch
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ProfilingEvent:
    """Represents a single profiling event."""
    event_type: str
    timestamp: float
    request_id: Optional[str] = None
    batch_size: Optional[int] = None
    context_length: Optional[int] = None
    generation_length: Optional[int] = None
    num_tokens: Optional[int] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Complete performance profile for a request."""
    request_id: str
    context_length: int
    generation_length: int
    total_execution_time: float
    events: List[ProfilingEvent] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)
    
    def add_event(self, event: ProfilingEvent):
        """Add a profiling event to this profile."""
        self.events.append(event)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class PerformanceProfiler:
    """
    Main performance profiler for vLLM.
    
    This class intercepts key execution points and collects comprehensive
    performance data for building performance models.
    """
    
    def __init__(self, 
                 output_dir: str = "vllm_profiles",
                 enable_memory_profiling: bool = True,
                 enable_gpu_profiling: bool = True,
                 max_profiles: int = 10000):
        """
        Initialize the performance profiler.
        
        Args:
            output_dir: Directory to save profiling data
            enable_memory_profiling: Whether to profile memory usage
            enable_gpu_profiling: Whether to profile GPU utilization
            max_profiles: Maximum number of profiles to keep in memory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_gpu_profiling = enable_gpu_profiling
        self.max_profiles = max_profiles
        
        # Thread-safe storage
        self._lock = threading.Lock()
        self.active_profiles: Dict[str, PerformanceProfile] = {}
        self.completed_profiles: deque = deque(maxlen=max_profiles)
        self.global_stats: Dict[str, Any] = defaultdict(list)
        
        # Performance counters
        self.total_requests = 0
        self.total_execution_time = 0.0
        
        # Start time
        self.start_time = time.time()
        
        logger.info(f"Performance profiler initialized. Output directory: {self.output_dir}")
    
    def start_request_profile(self, 
                            request_id: str, 
                            context_length: int, 
                            generation_length: int) -> None:
        """
        Start profiling a new request.
        
        Args:
            request_id: Unique identifier for the request
            context_length: Length of the input context
            generation_length: Expected length of generation
        """
        with self._lock:
            profile = PerformanceProfile(
                request_id=request_id,
                context_length=context_length,
                generation_length=generation_length,
                total_execution_time=0.0
            )
            self.active_profiles[request_id] = profile
            self.total_requests += 1
            
            # Add start event
            start_event = ProfilingEvent(
                event_type="request_start",
                timestamp=time.time(),
                request_id=request_id,
                context_length=context_length,
                generation_length=generation_length
            )
            profile.add_event(start_event)
            
            logger.debug(f"Started profiling request {request_id} (context: {context_length}, gen: {generation_length})")
    
    def record_event(self, 
                    event_type: str, 
                    request_id: str, 
                    execution_time: Optional[float] = None,
                    batch_size: Optional[int] = None,
                    num_tokens: Optional[int] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a profiling event.
        
        Args:
            event_type: Type of event (e.g., 'prefill', 'decode', 'scheduling')
            request_id: ID of the request being profiled
            execution_time: Time taken for this event
            batch_size: Batch size for this event
            num_tokens: Number of tokens processed
            metadata: Additional metadata for the event
        """
        with self._lock:
            if request_id not in self.active_profiles:
                logger.warning(f"Attempted to record event for unknown request: {request_id}")
                return
            
            profile = self.active_profiles[request_id]
            
            # Get current memory and GPU stats if enabled
            memory_usage = None
            gpu_utilization = None
            if self.enable_memory_profiling:
                memory_usage = self._get_memory_usage()
            if self.enable_gpu_profiling:
                gpu_utilization = self._get_gpu_utilization()
            
            event = ProfilingEvent(
                event_type=event_type,
                timestamp=time.time(),
                request_id=request_id,
                batch_size=batch_size,
                execution_time=execution_time,
                num_tokens=num_tokens,
                memory_usage=memory_usage,
                gpu_utilization=gpu_utilization,
                metadata=metadata or {}
            )
            
            profile.add_event(event)
            
            # Update global stats
            if execution_time is not None:
                self.global_stats[f"{event_type}_times"].append(execution_time)
                self.global_stats[f"{event_type}_batch_sizes"].append(batch_size or 0)
                self.global_stats[f"{event_type}_token_counts"].append(num_tokens or 0)
            
            logger.debug(f"Recorded event {event_type} for request {request_id} (time: {execution_time:.4f}s)")
    
    def end_request_profile(self, request_id: str, total_execution_time: float) -> None:
        """
        End profiling for a request.
        
        Args:
            request_id: ID of the request to end profiling
            total_execution_time: Total execution time for the request
        """
        with self._lock:
            if request_id not in self.active_profiles:
                logger.warning(f"Attempted to end profile for unknown request: {request_id}")
                return
            
            profile = self.active_profiles[request_id]
            profile.total_execution_time = total_execution_time
            
            # Add end event
            end_event = ProfilingEvent(
                event_type="request_end",
                timestamp=time.time(),
                request_id=request_id,
                execution_time=total_execution_time
            )
            profile.add_event(end_event)
            
            # Move to completed profiles
            self.completed_profiles.append(profile)
            del self.active_profiles[request_id]
            
            # Update global stats
            self.total_execution_time += total_execution_time
            
            logger.debug(f"Completed profiling request {request_id} (total time: {total_execution_time:.4f}s)")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected performance data."""
        with self._lock:
            summary = {
                "total_requests": self.total_requests,
                "total_execution_time": self.total_execution_time,
                "active_profiles": len(self.active_profiles),
                "completed_profiles": len(self.completed_profiles),
                "uptime": time.time() - self.start_time,
                "global_stats": dict(self.global_stats)
            }
            
            # Calculate averages for each event type
            for event_type in set(event.event_type for profile in self.completed_profiles for event in profile.events):
                times = self.global_stats.get(f"{event_type}_times", [])
                if times:
                    summary[f"{event_type}_avg_time"] = sum(times) / len(times)
                    summary[f"{event_type}_min_time"] = min(times)
                    summary[f"{event_type}_max_time"] = max(times)
                    summary[f"{event_type}_count"] = len(times)
            
            return summary
    
    def save_profiles(self, filename: Optional[str] = None) -> str:
        """
        Save all completed profiles to a JSON file.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"vllm_profiles_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with self._lock:
            data = {
                "metadata": {
                    "timestamp": time.time(),
                    "total_requests": self.total_requests,
                    "total_execution_time": self.total_execution_time,
                    "uptime": time.time() - self.start_time
                },
                "profiles": [profile.to_dict() for profile in self.completed_profiles],
                "global_stats": dict(self.global_stats)
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.completed_profiles)} profiles to {filepath}")
        return str(filepath)
    
    def export_performance_model_data(self, filename: Optional[str] = None) -> str:
        """
        Export data in a format suitable for building performance models.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"performance_model_data_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with self._lock:
            model_data = []
            for profile in self.completed_profiles:
                # Extract key features for performance modeling
                model_entry = {
                    "request_id": profile.request_id,
                    "context_length": profile.context_length,
                    "generation_length": profile.generation_length,
                    "total_execution_time": profile.total_execution_time,
                    "step_breakdown": {}
                }
                
                # Group events by type and calculate step-level metrics
                step_events = defaultdict(list)
                for event in profile.events:
                    if event.event_type not in ["request_start", "request_end"]:
                        step_events[event.event_type].append(event)
                
                for step_type, events in step_events.items():
                    if events:
                        step_entry = {
                            "count": len(events),
                            "total_time": sum(e.execution_time or 0 for e in events),
                            "avg_time": sum(e.execution_time or 0 for e in events) / len(events),
                            "batch_sizes": [e.batch_size for e in events if e.batch_size is not None],
                            "token_counts": [e.num_tokens for e in events if e.num_tokens is not None]
                        }
                        model_entry["step_breakdown"][step_type] = step_entry
                
                model_data.append(model_entry)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Exported performance model data to {filepath}")
        return str(filepath)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in GB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
        except:
            pass
        return None
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage."""
        try:
            if torch.cuda.is_available():
                # This is a simplified approach - in production you might want
                # to use nvidia-ml-py or similar for more accurate GPU metrics
                return torch.cuda.utilization()
        except:
            pass
        return None
    
    def clear_profiles(self) -> None:
        """Clear all stored profiles."""
        with self._lock:
            self.active_profiles.clear()
            self.completed_profiles.clear()
            self.global_stats.clear()
            self.total_requests = 0
            self.total_execution_time = 0.0
            self.start_time = time.time()
        
        logger.info("Cleared all profiling data")


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def set_profiler(profiler: PerformanceProfiler) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


def start_request_profile(request_id: str, context_length: int, generation_length: int) -> None:
    """Start profiling a request using the global profiler."""
    profiler = get_profiler()
    profiler.start_request_profile(request_id, context_length, generation_length)


def record_event(event_type: str, request_id: str, **kwargs) -> None:
    """Record an event using the global profiler."""
    profiler = get_profiler()
    profiler.record_event(event_type, request_id, **kwargs)


def end_request_profile(request_id: str, total_execution_time: float) -> None:
    """End profiling a request using the global profiler."""
    profiler = get_profiler()
    profiler.end_request_profile(request_id, total_execution_time)


def save_profiles(filename: Optional[str] = None) -> str:
    """Save all profiles using the global profiler."""
    profiler = get_profiler()
    return profiler.save_profiles(filename)


def export_performance_model_data(filename: Optional[str] = None) -> str:
    """Export performance model data using the global profiler."""
    profiler = get_profiler()
    return profiler.export_performance_model_data(filename)
