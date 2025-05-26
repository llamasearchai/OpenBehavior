"""
Database manager for OpenInterpretability platform using SQLite.

Handles storage and retrieval of evaluation results, model analyses, and metrics.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sqlite_utils

from ..models.evaluation import EvaluationResult, BatchEvaluationResult
from ..core.analyzer import InterpretabilityReport, ModelComparison

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for OpenInterpretability data storage.
    
    Uses SQLite with sqlite-utils for efficient data handling.
    """
    
    def __init__(self, db_path: str = "data/evaluations.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db = sqlite_utils.Database(str(self.db_path))
        logger.info(f"DatabaseManager initialized with database: {self.db_path}")
    
    async def init_database(self):
        """Initialize database schema."""
        try:
            # Create evaluations table
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    safety_score REAL,
                    safety_risk_level TEXT,
                    safety_details TEXT,
                    ethical_score REAL,
                    ethical_details TEXT,
                    alignment_score REAL,
                    alignment_details TEXT,
                    metadata TEXT,
                    evaluation_types TEXT,
                    processing_time REAL,
                    is_acceptable BOOLEAN,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS model_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    analysis_timestamp TEXT NOT NULL,
                    confidence_score REAL,
                    behavior_patterns TEXT,
                    insights TEXT,
                    risk_assessment TEXT,
                    recommendations TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS model_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    safety_comparison TEXT,
                    ethical_comparison TEXT,
                    alignment_comparison TEXT,
                    overall_winner TEXT,
                    detailed_analysis TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT,
                    model TEXT,
                    evaluation_types TEXT,
                    processing_time REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_evaluations_model ON evaluations(model);
                CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp ON evaluations(timestamp);
                CREATE INDEX IF NOT EXISTS idx_evaluations_safety_score ON evaluations(safety_score);
                CREATE INDEX IF NOT EXISTS idx_evaluations_ethical_score ON evaluations(ethical_score);
                CREATE INDEX IF NOT EXISTS idx_evaluations_alignment_score ON evaluations(alignment_score);
                CREATE INDEX IF NOT EXISTS idx_model_analyses_model ON model_analyses(model);
                CREATE INDEX IF NOT EXISTS idx_metrics_model ON metrics(model);
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
            """)
            
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def save_evaluation(self, result: EvaluationResult) -> None:
        """
        Save evaluation result to database.
        
        Args:
            result: EvaluationResult to save
        """
        try:
            evaluation_data = {
                "id": result.id,
                "text": result.text,
                "model": result.model,
                "safety_score": result.safety_score.overall_score if result.safety_score else None,
                "safety_risk_level": result.safety_score.risk_level.value if result.safety_score else None,
                "safety_details": json.dumps({
                    "category_scores": {cat.value: score for cat, score in result.safety_score.category_scores.items()},
                    "detected_issues": result.safety_score.detected_issues,
                    "confidence": result.safety_score.confidence,
                    "explanation": result.safety_score.explanation
                }) if result.safety_score else None,
                "ethical_score": result.ethical_score.overall_score if result.ethical_score else None,
                "ethical_details": json.dumps({
                    "dimension_scores": {dim.value: score for dim, score in result.ethical_score.dimension_scores.items()},
                    "ethical_concerns": result.ethical_score.ethical_concerns,
                    "recommendations": result.ethical_score.recommendations,
                    "confidence": result.ethical_score.confidence,
                    "explanation": result.ethical_score.explanation
                }) if result.ethical_score else None,
                "alignment_score": result.alignment_score.overall_score if result.alignment_score else None,
                "alignment_details": json.dumps({
                    "criteria_scores": {crit.value: score for crit, score in result.alignment_score.criteria_scores.items()},
                    "alignment_issues": result.alignment_score.alignment_issues,
                    "strengths": result.alignment_score.strengths,
                    "confidence": result.alignment_score.confidence,
                    "explanation": result.alignment_score.explanation
                }) if result.alignment_score else None,
                "metadata": json.dumps(result.metadata),
                "evaluation_types": json.dumps(result.evaluation_types),
                "processing_time": result.processing_time,
                "is_acceptable": result.is_acceptable,
                "timestamp": result.timestamp.isoformat()
            }
            
            self.db["evaluations"].insert(evaluation_data, replace=True)
            logger.debug(f"Saved evaluation {result.id} to database")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation {result.id}: {e}")
            raise
    
    async def save_model_analysis(self, report: InterpretabilityReport) -> None:
        """
        Save model analysis report to database.
        
        Args:
            report: InterpretabilityReport to save
        """
        try:
            analysis_data = {
                "model": report.model,
                "analysis_timestamp": report.analysis_timestamp.isoformat(),
                "confidence_score": report.confidence_score,
                "behavior_patterns": json.dumps([
                    {
                        "pattern_id": p.pattern_id,
                        "pattern_type": p.pattern_type,
                        "description": p.description,
                        "frequency": p.frequency,
                        "confidence": p.confidence,
                        "examples": p.examples,
                        "risk_level": p.risk_level
                    }
                    for p in report.behavior_patterns
                ]),
                "insights": json.dumps([
                    {
                        "category": i.category,
                        "insight_type": i.insight_type,
                        "description": i.description,
                        "confidence": i.confidence,
                        "supporting_evidence": i.supporting_evidence,
                        "recommendations": i.recommendations
                    }
                    for i in report.insights
                ]),
                "risk_assessment": json.dumps(report.risk_assessment),
                "recommendations": json.dumps(report.recommendations)
            }
            
            self.db["model_analyses"].insert(analysis_data)
            logger.debug(f"Saved model analysis for {report.model} to database")
            
        except Exception as e:
            logger.error(f"Failed to save model analysis for {report.model}: {e}")
            raise
    
    async def save_model_comparison(self, comparison: ModelComparison) -> None:
        """
        Save model comparison to database.
        
        Args:
            comparison: ModelComparison to save
        """
        try:
            comparison_data = {
                "model_a": comparison.model_a,
                "model_b": comparison.model_b,
                "safety_comparison": json.dumps(comparison.safety_comparison),
                "ethical_comparison": json.dumps(comparison.ethical_comparison),
                "alignment_comparison": json.dumps(comparison.alignment_comparison),
                "overall_winner": comparison.overall_winner,
                "detailed_analysis": json.dumps(comparison.detailed_analysis)
            }
            
            self.db["model_comparisons"].insert(comparison_data)
            logger.debug(f"Saved model comparison {comparison.model_a} vs {comparison.model_b} to database")
            
        except Exception as e:
            logger.error(f"Failed to save model comparison: {e}")
            raise
    
    async def get_evaluations(
        self,
        model: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: str = "timestamp DESC"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evaluations from database.
        
        Args:
            model: Filter by model name
            limit: Limit number of results
            offset: Offset for pagination
            order_by: Order by clause
            
        Returns:
            List of evaluation dictionaries
        """
        try:
            query = "SELECT * FROM evaluations"
            params = []
            
            if model:
                query += " WHERE model = ?"
                params.append(model)
            
            query += f" ORDER BY {order_by}"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            if offset:
                query += " OFFSET ?"
                params.append(offset)
            
            results = list(self.db.execute(query, params))
            
            # Parse JSON fields
            for result in results:
                if result.get("metadata"):
                    result["metadata"] = json.loads(result["metadata"])
                if result.get("evaluation_types"):
                    result["evaluation_types"] = json.loads(result["evaluation_types"])
                if result.get("safety_details"):
                    result["safety_details"] = json.loads(result["safety_details"])
                if result.get("ethical_details"):
                    result["ethical_details"] = json.loads(result["ethical_details"])
                if result.get("alignment_details"):
                    result["alignment_details"] = json.loads(result["alignment_details"])
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve evaluations: {e}")
            raise
    
    async def get_model_analyses(
        self,
        model: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve model analyses from database.
        
        Args:
            model: Filter by model name
            limit: Limit number of results
            
        Returns:
            List of analysis dictionaries
        """
        try:
            query = "SELECT * FROM model_analyses"
            params = []
            
            if model:
                query += " WHERE model = ?"
                params.append(model)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            results = list(self.db.execute(query, params))
            
            # Parse JSON fields
            for result in results:
                if result.get("behavior_patterns"):
                    result["behavior_patterns"] = json.loads(result["behavior_patterns"])
                if result.get("insights"):
                    result["insights"] = json.loads(result["insights"])
                if result.get("risk_assessment"):
                    result["risk_assessment"] = json.loads(result["risk_assessment"])
                if result.get("recommendations"):
                    result["recommendations"] = json.loads(result["recommendations"])
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve model analyses: {e}")
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with various statistics
        """
        try:
            stats = {}
            
            # Total evaluations
            stats["total_evaluations"] = self.db.execute("SELECT COUNT(*) as count FROM evaluations").fetchone()["count"]
            
            # Evaluations by model
            model_counts = list(self.db.execute("""
                SELECT model, COUNT(*) as count 
                FROM evaluations 
                GROUP BY model 
                ORDER BY count DESC
            """))
            stats["evaluations_by_model"] = {row["model"]: row["count"] for row in model_counts}
            
            # Average scores
            avg_scores = self.db.execute("""
                SELECT 
                    AVG(safety_score) as avg_safety,
                    AVG(ethical_score) as avg_ethical,
                    AVG(alignment_score) as avg_alignment
                FROM evaluations
                WHERE safety_score IS NOT NULL 
                OR ethical_score IS NOT NULL 
                OR alignment_score IS NOT NULL
            """).fetchone()
            
            stats["average_scores"] = {
                "safety": avg_scores["avg_safety"],
                "ethical": avg_scores["avg_ethical"],
                "alignment": avg_scores["avg_alignment"]
            }
            
            # Risk level distribution
            risk_dist = list(self.db.execute("""
                SELECT safety_risk_level, COUNT(*) as count
                FROM evaluations 
                WHERE safety_risk_level IS NOT NULL
                GROUP BY safety_risk_level
            """))
            stats["risk_level_distribution"] = {row["safety_risk_level"]: row["count"] for row in risk_dist}
            
            # Recent activity (last 24 hours)
            recent_count = self.db.execute("""
                SELECT COUNT(*) as count 
                FROM evaluations 
                WHERE created_at > datetime('now', '-1 day')
            """).fetchone()["count"]
            stats["recent_evaluations_24h"] = recent_count
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
    
    async def export_data(
        self,
        table: str,
        format: str = "json",
        output_file: Optional[str] = None
    ) -> str:
        """
        Export data from database.
        
        Args:
            table: Table name to export
            format: Export format (json, csv, yaml)
            output_file: Optional output file path
            
        Returns:
            Exported data as string
        """
        try:
            if table not in ["evaluations", "model_analyses", "model_comparisons", "metrics"]:
                raise ValueError(f"Invalid table: {table}")
            
            rows = list(self.db[table].rows)
            
            if format == "json":
                import json
                data = json.dumps(rows, indent=2, default=str)
            elif format == "csv":
                import csv
                import io
                
                if not rows:
                    return ""
                
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
                data = output.getvalue()
            elif format == "yaml":
                import yaml
                data = yaml.dump(rows, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(data)
                logger.info(f"Exported {len(rows)} rows from {table} to {output_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if hasattr(self.db, 'close'):
            self.db.close()
        logger.info("Database connection closed") 