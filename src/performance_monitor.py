"""Performance monitoring utilities for debugging slow operations."""
import time
from typing import Dict, List, Optional
from collections import defaultdict


class PerformanceMonitor:
    """Simple performance monitor for tracking operation times."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.current_operation: Optional[str] = None
        self.start_time: Optional[float] = None
    
    def start(self, operation: str):
        """Start timing an operation."""
        if not self.enabled:
            return
        if self.current_operation:
            self.end()  # End previous operation
        self.current_operation = operation
        self.start_time = time.time()
    
    def end(self):
        """End timing current operation."""
        if not self.enabled or not self.current_operation or not self.start_time:
            return
        elapsed = time.time() - self.start_time
        self.timings[self.current_operation].append(elapsed)
        self.current_operation = None
        self.start_time = None
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        stats = {}
        for op, times in self.timings.items():
            if times:
                stats[op] = {
                    'mean': sum(times) / len(times),
                    'max': max(times),
                    'min': min(times),
                    'total': sum(times),
                    'count': len(times),
                }
        return stats
    
    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self.current_operation = None
        self.start_time = None
    
    def print_summary(self, top_n: int = 10):
        """Print summary of slowest operations."""
        stats = self.get_stats()
        sorted_ops = sorted(
            stats.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )[:top_n]
        
        print("\n=== Performance Summary ===")
        for op, stat in sorted_ops:
            print(
                f"{op:30s}: "
                f"mean={stat['mean']*1000:.2f}ms, "
                f"max={stat['max']*1000:.2f}ms, "
                f"count={stat['count']}"
            )
