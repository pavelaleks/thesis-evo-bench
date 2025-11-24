"""Pydantic models for all entities in the evolution system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StageType(str, Enum):
    """Enumeration of pipeline stage types."""

    SEMANTIC = "semantic"
    OUTLINE = "outline"
    INTRO = "intro"
    CHAPTER1 = "chapter1"
    REVIEW = "review"
    CROSS_EVAL = "cross_eval"
    MULTI_PASS = "multi_pass"


class Topic(BaseModel):
    """Represents a topic for thesis generation."""

    title: str
    domain: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StageConfig(BaseModel):
    """Configuration for a single pipeline stage."""

    stage_type: StageType
    prompt_profile_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    enabled: bool = True
    depends_on: List[str] = Field(default_factory=list)  # Stage IDs this depends on
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration for an entire pipeline."""

    pipeline_id: str
    stages: List[StageConfig]
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_stage_by_type(self, stage_type: StageType) -> Optional[StageConfig]:
        """Get first stage of given type."""
        for stage in self.stages:
            if stage.stage_type == stage_type and stage.enabled:
                return stage
        return None


class PromptProfile(BaseModel):
    """A prompt template profile."""

    name: str
    category: str  # "generation", "evaluation", "meta_prompt_evolution", "meta_pipeline_evolution", "evolution_control"
    template: str
    description: Optional[str] = None
    expected_output_format: str = "json"  # "json", "text", "structured"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def render(self, **kwargs: Any) -> str:
        """
        Render prompt template with provided variables.
        
        Note: This method uses format() which may raise KeyError.
        For safe rendering with defaults, use render_prompt() from utils.
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            # Log warning but don't crash - use safe rendering fallback
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing template variable {e}, using safe rendering")
            # Import here to avoid circular dependency
            from thesis_evo_bench.core.utils import render_prompt
            return render_prompt(self.template, kwargs)


class StageRunResult(BaseModel):
    """Result of running a single pipeline stage."""

    stage_type: StageType
    prompt_profile_name: str
    topic: Topic
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_content: str
    temperature: float
    model_used: str
    token_usage: Dict[str, int] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result of evaluating a stage output or pipeline."""

    evaluator_name: str
    evaluator_type: str  # "rule_based", "llm_judge", "consistency"
    score: float  # 0.0 to 1.0
    stage_type: Optional[StageType] = None
    topic: Optional[Topic] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Summary of a complete pipeline run."""

    run_id: str
    pipeline_id: str
    topic: Topic
    stage_results: List[StageRunResult] = Field(default_factory=list)
    evaluations: List[EvaluationResult] = Field(default_factory=list)
    aggregated_score: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_stage_result(self, stage_type: StageType) -> Optional[StageRunResult]:
        """Get result for a specific stage type."""
        for result in self.stage_results:
            if result.stage_type == stage_type:
                return result
        return None


class PromptPopulation(BaseModel):
    """Population of prompts for a specific stage."""

    stage_type: StageType
    generation: int
    prompts: List[PromptProfile]
    scores: Dict[str, float] = Field(default_factory=dict)  # prompt_name -> score


class PipelinePopulation(BaseModel):
    """Population of pipeline configurations."""

    generation: int
    pipelines: List[PipelineConfig]
    scores: Dict[str, float] = Field(default_factory=dict)  # pipeline_id -> score
    evaluations: Dict[str, List[EvaluationResult]] = Field(
        default_factory=dict,
    )  # pipeline_id -> evaluations

