"""Evaluation modules for assessing pipeline outputs."""

import json
import logging
from typing import Any, Dict, List, Optional

from thesis_evo_bench.config import get_settings
from thesis_evo_bench.core.models import (
    EvaluationResult,
    RunSummary,
    StageRunResult,
    StageType,
    Topic,
)
from thesis_evo_bench.core.prompts import get_prompt_profile
from thesis_evo_bench.core.utils import render_prompt
from thesis_evo_bench.llm import call_llm_sync

logger = logging.getLogger(__name__)


class RuleBasedEvaluator:
    """Rule-based evaluator using heuristics and rules."""

    def __init__(self, name: str = "rule_based"):
        """Initialize rule-based evaluator."""
        self.name = name

    def evaluate(
        self,
        content: str,
        stage_type: Optional[StageType] = None,
        topic: Optional[Topic] = None,
    ) -> EvaluationResult:
        """
        Evaluate content using rule-based heuristics.

        Args:
            content: Content to evaluate
            stage_type: Type of stage that produced content
            topic: Topic context

        Returns:
            EvaluationResult with score and details
        """
        score = 0.0
        details: Dict[str, Any] = {}

        # Length check
        word_count = len(content.split())
        if word_count < 50:
            length_score = 0.2
        elif word_count < 200:
            length_score = 0.5
        elif word_count < 1000:
            length_score = 0.8
        else:
            length_score = 1.0
        details["length_score"] = length_score
        details["word_count"] = word_count

        # Structure check (look for JSON or structured content)
        try:
            json.loads(content)
            structure_score = 1.0
            details["is_json"] = True
        except (json.JSONDecodeError, ValueError):
            # Check for basic structure indicators
            has_sections = any(
                marker in content.lower()
                for marker in ["section", "chapter", "introduction", "conclusion"]
            )
            structure_score = 0.6 if has_sections else 0.3
            details["is_json"] = False
            details["has_sections"] = has_sections

        # Completeness check (basic keyword presence)
        completeness_keywords = ["topic", "analysis", "method", "result"]
        found_keywords = sum(1 for kw in completeness_keywords if kw in content.lower())
        completeness_score = found_keywords / len(completeness_keywords)
        details["completeness_score"] = completeness_score
        details["found_keywords"] = found_keywords

        # Calculate weighted average
        score = (
            length_score * 0.3
            + structure_score * 0.4
            + completeness_score * 0.3
        )

        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type="rule_based",
            score=score,
            stage_type=stage_type,
            topic=topic,
            details=details,
            reasoning=f"Rule-based evaluation: length={length_score:.2f}, structure={structure_score:.2f}, completeness={completeness_score:.2f}",
        )


class LLMJudgeEvaluator:
    """LLM-based evaluator using prompt-driven judgment."""

    def __init__(
        self,
        name: str = "llm_judge",
        prompt_profile_name: str = "eval_general_quality_v1",
    ):
        """
        Initialize LLM judge evaluator.

        Args:
            name: Evaluator name
            prompt_profile_name: Name of prompt profile to use
        """
        self.name = name
        self.prompt_profile_name = prompt_profile_name
        self.settings = get_settings()

    def evaluate(
        self,
        content: str,
        stage_type: Optional[StageType] = None,
        topic: Optional[Topic] = None,
        context: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate content using LLM judgment.

        Args:
            content: Content to evaluate
            stage_type: Type of stage
            topic: Topic context
            context: Additional context for evaluation

        Returns:
            EvaluationResult with LLM-generated score
        """
        try:
            prompt_profile = get_prompt_profile(self.prompt_profile_name)
            # Use safe rendering to avoid crashes on missing variables
            eval_context = {
                "content": content,
                "context": context or "",
                "stage_type": stage_type.value if stage_type else "unknown",
                "topic_title": topic.title if topic else "",
                "topic_domain": topic.domain if topic else "",
            }
            prompt = render_prompt(prompt_profile.template, eval_context)

            response = call_llm_sync(
                model=self.settings.default_model,
                prompt=prompt,
                temperature=0.3,  # Lower temperature for evaluation
            )

            # Parse JSON response
            response_text = response["content"]
            try:
                eval_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    eval_data = json.loads(response_text[json_start:json_end].strip())
                else:
                    # Fallback: extract score from text
                    score = 0.5
                    eval_data = {"score": score, "reasoning": response_text}

            score = float(eval_data.get("score", 0.5))
            reasoning = eval_data.get("reasoning", "LLM evaluation completed")
            details = {k: v for k, v in eval_data.items() if k not in ["score", "reasoning"]}

            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type="llm_judge",
                score=score,
                stage_type=stage_type,
                topic=topic,
                details=details,
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fallback to rule-based score
            fallback = RuleBasedEvaluator(name=f"{self.name}_fallback")
            result = fallback.evaluate(content, stage_type, topic)
            result.evaluator_name = self.name
            result.evaluator_type = "llm_judge"
            result.reasoning = f"LLM evaluation failed, using fallback: {str(e)}"
            return result


class ConsistencyEvaluator:
    """Evaluator for cross-stage consistency."""

    def __init__(
        self,
        name: str = "consistency",
        prompt_profile_name: str = "eval_cross_consistency_v1",
    ):
        """
        Initialize consistency evaluator.

        Args:
            name: Evaluator name
            prompt_profile_name: Name of prompt profile to use
        """
        self.name = name
        self.prompt_profile_name = prompt_profile_name
        self.settings = get_settings()

    def evaluate(
        self,
        current_section: str,
        previous_sections: List[str],
        topic: Optional[Topic] = None,
    ) -> EvaluationResult:
        """
        Evaluate consistency between current and previous sections.

        Args:
            current_section: Current section content
            previous_sections: List of previous section contents
            topic: Topic context

        Returns:
            EvaluationResult with consistency score
        """
        if not previous_sections:
            # No previous sections to compare
            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type="consistency",
                score=1.0,
                topic=topic,
                details={"previous_sections_count": 0},
                reasoning="No previous sections to compare",
            )

        try:
            prompt_profile = get_prompt_profile(self.prompt_profile_name)
            # Use safe rendering to avoid crashes on missing variables
            eval_context = {
                "current_section": current_section,
                "previous_sections": "\n\n---\n\n".join(previous_sections),
                "topic_title": topic.title if topic else "",
            }
            prompt = render_prompt(prompt_profile.template, eval_context)

            response = call_llm_sync(
                model=self.settings.default_model,
                prompt=prompt,
                temperature=0.3,
            )

            response_text = response["content"]
            try:
                eval_data = json.loads(response_text)
            except json.JSONDecodeError:
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    eval_data = json.loads(response_text[json_start:json_end].strip())
                else:
                    eval_data = {"consistency_score": 0.5, "reasoning": response_text}

            score = float(eval_data.get("consistency_score", 0.5))
            reasoning = eval_data.get("reasoning", "Consistency evaluation completed")
            details = {
                k: v
                for k, v in eval_data.items()
                if k not in ["consistency_score", "reasoning"]
            }
            details["previous_sections_count"] = len(previous_sections)

            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type="consistency",
                score=score,
                topic=topic,
                details=details,
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"Consistency evaluation failed: {e}")
            # Fallback: simple keyword overlap check
            current_words = set(current_section.lower().split())
            previous_words = set(" ".join(previous_sections).lower().split())
            overlap = len(current_words & previous_words) / max(len(current_words), 1)
            score = min(overlap * 2, 1.0)  # Normalize

            return EvaluationResult(
                evaluator_name=self.name,
                evaluator_type="consistency",
                score=score,
                topic=topic,
                details={"overlap_ratio": overlap, "fallback": True},
                reasoning=f"Consistency evaluation failed, using keyword overlap: {str(e)}",
            )


def evaluate_pipeline_results(
    stage_results: List[StageRunResult],
    evaluators: List[Any],
    topic: Optional[Topic] = None,
) -> List[EvaluationResult]:
    """
    Evaluate pipeline results using multiple evaluators.

    Args:
        stage_results: List of stage run results
        evaluators: List of evaluator instances
        topic: Topic context

    Returns:
        List of evaluation results
    """
    all_evaluations: List[EvaluationResult] = []

    for stage_result in stage_results:
        for evaluator in evaluators:
            if isinstance(evaluator, ConsistencyEvaluator):
                # For consistency, need previous sections
                previous_results = [
                    r.output_content
                    for r in stage_results
                    if r.stage_type != stage_result.stage_type
                ]
                eval_result = evaluator.evaluate(
                    current_section=stage_result.output_content,
                    previous_sections=previous_results,
                    topic=topic,
                )
            else:
                eval_result = evaluator.evaluate(
                    content=stage_result.output_content,
                    stage_type=stage_result.stage_type,
                    topic=topic,
                )
            all_evaluations.append(eval_result)

    return all_evaluations

