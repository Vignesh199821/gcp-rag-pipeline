"""
Evaluation Module - Track RAG pipeline performance and quality metrics
"""
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


@dataclass
class QueryEval:
    """Evaluation data for a single query."""
    query_id: str
    timestamp: str
    question: str
    answer: str
    num_sources: int
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    sources_preview: List[str]
    feedback: Optional[str] = None  # "positive", "negative", or None
    feedback_comment: Optional[str] = None


class EvalTracker:
    """Track and store evaluation metrics for the RAG pipeline."""
    
    def __init__(self, storage_path: str = "./evals"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.evals_file = self.storage_path / "query_evals.json"
        self.metrics_file = self.storage_path / "metrics.json"
        
        # Load existing evals
        self.evals: List[QueryEval] = self._load_evals()
    
    def _load_evals(self) -> List[QueryEval]:
        """Load existing evaluations from file."""
        if self.evals_file.exists():
            try:
                with open(self.evals_file, 'r') as f:
                    data = json.load(f)
                    return [QueryEval(**e) for e in data]
            except Exception:
                return []
        return []
    
    def _save_evals(self):
        """Save evaluations to file."""
        with open(self.evals_file, 'w') as f:
            json.dump([asdict(e) for e in self.evals], f, indent=2)
    
    def log_query(
        self,
        question: str,
        answer: str,
        sources: List[str],
        retrieval_time: float,
        generation_time: float
    ) -> str:
        """Log a query evaluation."""
        query_id = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        eval_entry = QueryEval(
            query_id=query_id,
            timestamp=datetime.now().isoformat(),
            question=question,
            answer=answer[:500] + "..." if len(answer) > 500 else answer,
            num_sources=len(sources),
            retrieval_time_ms=round(retrieval_time * 1000, 2),
            generation_time_ms=round(generation_time * 1000, 2),
            total_time_ms=round((retrieval_time + generation_time) * 1000, 2),
            sources_preview=[s[:100] + "..." if len(s) > 100 else s for s in sources[:3]]
        )
        
        self.evals.append(eval_entry)
        self._save_evals()
        self._update_metrics()
        
        return query_id
    
    def add_feedback(self, query_id: str, feedback: str, comment: Optional[str] = None) -> bool:
        """Add user feedback for a query."""
        for eval_entry in self.evals:
            if eval_entry.query_id == query_id:
                eval_entry.feedback = feedback
                eval_entry.feedback_comment = comment
                self._save_evals()
                self._update_metrics()
                return True
        return False
    
    def _update_metrics(self):
        """Update aggregate metrics."""
        if not self.evals:
            return
        
        total_queries = len(self.evals)
        avg_retrieval_time = sum(e.retrieval_time_ms for e in self.evals) / total_queries
        avg_generation_time = sum(e.generation_time_ms for e in self.evals) / total_queries
        avg_total_time = sum(e.total_time_ms for e in self.evals) / total_queries
        avg_sources = sum(e.num_sources for e in self.evals) / total_queries
        
        # Feedback stats
        positive = sum(1 for e in self.evals if e.feedback == "positive")
        negative = sum(1 for e in self.evals if e.feedback == "negative")
        no_feedback = sum(1 for e in self.evals if e.feedback is None)
        
        metrics = {
            "last_updated": datetime.now().isoformat(),
            "total_queries": total_queries,
            "avg_retrieval_time_ms": round(avg_retrieval_time, 2),
            "avg_generation_time_ms": round(avg_generation_time, 2),
            "avg_total_time_ms": round(avg_total_time, 2),
            "avg_sources_per_query": round(avg_sources, 2),
            "feedback": {
                "positive": positive,
                "negative": negative,
                "no_feedback": no_feedback,
                "positive_rate": round(positive / (positive + negative) * 100, 1) if (positive + negative) > 0 else 0
            }
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {"total_queries": 0, "message": "No queries yet"}
    
    def get_recent_evals(self, limit: int = 10) -> List[Dict]:
        """Get recent evaluations."""
        recent = self.evals[-limit:]
        return [asdict(e) for e in reversed(recent)]
    
    def export_evals(self) -> Dict:
        """Export all evaluations for analysis."""
        return {
            "exported_at": datetime.now().isoformat(),
            "total_evals": len(self.evals),
            "metrics": self.get_metrics(),
            "evals": [asdict(e) for e in self.evals]
        }



class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time

