"""Pipeline execution engine for running thesis generation pipelines."""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from thesis_evo_bench.config import get_settings
from thesis_evo_bench.core.evaluators import (
    ConsistencyEvaluator,
    LLMJudgeEvaluator,
    RuleBasedEvaluator,
    evaluate_pipeline_results,
)
from thesis_evo_bench.core.models import (
    EvaluationResult,
    PipelineConfig,
    RunSummary,
    StageConfig,
    StageRunResult,
    StageType,
    Topic,
)
from thesis_evo_bench.core.prompts import get_prompt_profile
from thesis_evo_bench.core.utils import render_prompt
from thesis_evo_bench.llm import call_llm_sync

logger = logging.getLogger(__name__)


def run_pipeline_for_topic(
    topic: Topic,
    pipeline: PipelineConfig,
    prompt_profiles: Dict[str, Any],  # Dict[str, PromptProfile] but avoiding circular import
    evaluators: Optional[List[Any]] = None,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> RunSummary:
    """
    Run a complete pipeline for a given topic.

    Args:
        topic: Topic to generate thesis for
        pipeline: Pipeline configuration
        prompt_profiles: Dictionary of prompt profiles (name -> PromptProfile)
        evaluators: Optional list of evaluators to use
        output_dir: Directory to save outputs
        run_id: Optional run ID (generated if not provided)

    Returns:
        RunSummary with all stage results and evaluations
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    if output_dir is None:
        output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if evaluators is None:
        evaluators = [
            RuleBasedEvaluator(),
            LLMJudgeEvaluator(),
            ConsistencyEvaluator(),
        ]

    settings = get_settings()
    start_time = time.time()
    stage_results: List[StageRunResult] = []
    accumulated_context: Dict[str, Any] = {
        "topic_title": topic.title,
        "topic_domain": topic.domain,
        "topic_description": topic.description or "",
        # Initialize common context variables with empty strings to prevent crashes
        "semantic_analysis": "",
        "outline_structure": "",
        "introduction_text": "",
        "previous_context": "",
        "content_to_review": "",
        "full_context": "",
    }

    # Execute stages in order
    for stage_config in pipeline.stages:
        if not stage_config.enabled:
            logger.info(f"Skipping disabled stage: {stage_config.stage_type}")
            continue

        # Check dependencies
        if stage_config.depends_on:
            # TODO: Implement dependency checking
            pass

        logger.info(f"Running stage: {stage_config.stage_type}")

        # Get prompt profile
        prompt_profile_name = stage_config.prompt_profile_name
        if prompt_profile_name not in prompt_profiles:
            # Try to get from registry
            try:
                prompt_profile = get_prompt_profile(prompt_profile_name)
            except ValueError:
                logger.error(f"Prompt profile '{prompt_profile_name}' not found")
                # Create error result and continue
                stage_result = StageRunResult(
                    stage_type=stage_config.stage_type,
                    prompt_profile_name=prompt_profile_name,
                    topic=topic,
                    input_data=accumulated_context.copy(),
                    output_content="ERROR: Prompt profile not found",
                    temperature=stage_config.temperature,
                    model_used=settings.default_model,
                    metadata={"error": f"Prompt profile '{prompt_profile_name}' not found"},
                )
                stage_results.append(stage_result)
                # Set empty output in context to prevent later stages from crashing
                accumulated_context[f"{stage_config.stage_type.value}_output"] = ""
                accumulated_context[f"{stage_config.stage_type.value}_result"] = ""
                continue
        else:
            prompt_profile = prompt_profiles[prompt_profile_name]

        # Render prompt with accumulated context using safe rendering
        try:
            prompt_text = render_prompt(prompt_profile.template, accumulated_context)
        except Exception as e:
            logger.warning(f"Failed to render prompt (using template as-is): {e}")
            prompt_text = prompt_profile.template

        # Call LLM
        stage_start_time = time.time()
        output_content = ""
        token_usage: Dict[str, int] = {}
        stage_success = False
        stage_error: Optional[str] = None
        response_model = settings.default_model

        try:
            response = call_llm_sync(
                model=settings.default_model,
                prompt=prompt_text,
                temperature=stage_config.temperature,
                max_tokens=stage_config.max_tokens,
            )
            stage_execution_time = time.time() - stage_start_time

            output_content = response.get("content", "")
            token_usage = response.get("usage", {})
            response_model = response.get("model", settings.default_model)
            stage_success = True

        except Exception as e:
            logger.warning(f"Stage {stage_config.stage_type} LLM call failed: {e}")
            stage_execution_time = time.time() - stage_start_time
            output_content = f"ERROR: {str(e)}"
            stage_error = str(e)
            # Continue execution with empty/error output

        # Always create stage result (even on failure)
        stage_result = StageRunResult(
            stage_type=stage_config.stage_type,
            prompt_profile_name=prompt_profile_name,
            topic=topic,
            input_data=accumulated_context.copy(),
            output_content=output_content,
            temperature=stage_config.temperature,
            model_used=response_model,
            token_usage={
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            },
            execution_time_seconds=stage_execution_time,
            metadata={"success": stage_success, "error": stage_error},
        )
        stage_results.append(stage_result)

        # Update accumulated context for next stages (even if this stage failed)
        # This ensures later stages don't crash due to missing variables
        accumulated_context[f"{stage_config.stage_type.value}_output"] = output_content
        accumulated_context[f"{stage_config.stage_type.value}_result"] = output_content

        # Add common aliases for context variables
        if stage_config.stage_type == StageType.SEMANTIC:
            accumulated_context["semantic_analysis"] = output_content
        elif stage_config.stage_type == StageType.OUTLINE:
            accumulated_context["outline_structure"] = output_content
        elif stage_config.stage_type == StageType.INTRO:
            accumulated_context["introduction_text"] = output_content
        elif stage_config.stage_type == StageType.CHAPTER1:
            # Build previous context from all previous stages
            previous_parts = [
                r.output_content
                for r in stage_results[:-1]  # All except current
                if r.output_content and not r.output_content.startswith("ERROR:")
            ]
            accumulated_context["previous_context"] = "\n\n---\n\n".join(previous_parts)
        elif stage_config.stage_type == StageType.REVIEW:
            accumulated_context["content_to_review"] = output_content
            # Build full context from all previous stages
            full_context_parts = [
                r.output_content
                for r in stage_results
                if r.stage_type != StageType.REVIEW and r.output_content and not r.output_content.startswith("ERROR:")
            ]
            accumulated_context["full_context"] = "\n\n---\n\n".join(full_context_parts)

        # Try to parse JSON and add structured data
        if stage_success and output_content and not output_content.startswith("ERROR:"):
            try:
                parsed = json.loads(output_content)
                if isinstance(parsed, dict):
                    # Add parsed fields to context, but don't overwrite critical ones
                    for key, value in parsed.items():
                        if key not in accumulated_context or not accumulated_context[key]:
                            accumulated_context[key] = value
            except (json.JSONDecodeError, ValueError):
                pass

        # Save stage output
        try:
            stage_output_file = (
                output_dir / f"{run_id}_{stage_config.stage_type.value}.json"
            )
            with open(stage_output_file, "w", encoding="utf-8") as f:
                json.dump(stage_result.model_dump(mode="json"), f, indent=2, default=str)
        except Exception as save_error:
            logger.warning(f"Failed to save stage output: {save_error}")

    # Evaluate results
    evaluations: List[EvaluationResult] = []
    if stage_results:
        evaluations = evaluate_pipeline_results(stage_results, evaluators, topic)

        # Save evaluations
        eval_dir = Path("data/evaluations")
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_file = eval_dir / f"{run_id}_evaluations.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(
                [e.model_dump(mode="json") for e in evaluations],
                f,
                indent=2,
                default=str,
            )

    # Calculate aggregated score
    if evaluations:
        aggregated_score = sum(e.score for e in evaluations) / len(evaluations)
    else:
        aggregated_score = 0.0

    execution_time = time.time() - start_time

    # Create run summary
    summary = RunSummary(
        run_id=run_id,
        pipeline_id=pipeline.pipeline_id,
        topic=topic,
        stage_results=stage_results,
        evaluations=evaluations,
        aggregated_score=aggregated_score,
        execution_time_seconds=execution_time,
    )

    # Save run summary
    runs_dir = Path("data/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_file = runs_dir / f"{run_id}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(mode="json"), f, indent=2, default=str)

    return summary

