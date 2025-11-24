"""Prompt evolution engine for Mode 1: evolving prompts for fixed pipeline structure."""

import json
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from thesis_evo_bench.config import get_settings
from thesis_evo_bench.core.evaluators import (
    ConsistencyEvaluator,
    LLMJudgeEvaluator,
    RuleBasedEvaluator,
)
from thesis_evo_bench.core.models import (
    PipelineConfig,
    PromptPopulation,
    PromptProfile,
    RunSummary,
    StageType,
    Topic,
)
from thesis_evo_bench.core.pipeline_executor import run_pipeline_for_topic
from thesis_evo_bench.core.prompts import PROMPT_REGISTRY, get_prompt_profile
from thesis_evo_bench.core.utils import parse_json_response, render_prompt
from thesis_evo_bench.llm import call_llm_sync

logger = logging.getLogger(__name__)


def run_prompt_evolution(
    base_pipeline: PipelineConfig,
    topics: List[Topic],
    prompt_profiles: Dict[str, PromptProfile],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run prompt evolution for fixed pipeline structure.

    Args:
        base_pipeline: Fixed pipeline structure
        topics: List of topics to evaluate on
        prompt_profiles: Initial prompt profiles dictionary
        config: Evolution configuration with:
            - generations: number of generations
            - population_size_per_stage: prompts per stage per generation
            - selection_top_k: top k to select each generation
            - mutation_rate: probability of mutation
            - random_seed: random seed

    Returns:
        Dictionary with evolution results and best prompts
    """
    random.seed(config.get("random_seed", 42))

    generations = config.get("generations", 5)
    population_size = config.get("population_size_per_stage", 10)
    top_k = config.get("selection_top_k", 3)
    mutation_rate = config.get("mutation_rate", 0.3)

    settings = get_settings()
    evolution_id = str(uuid.uuid4())

    # Get stages that need prompt evolution
    stages_to_evolve = [
        stage.stage_type
        for stage in base_pipeline.stages
        if stage.enabled
    ]

    # Initialize populations per stage
    populations: Dict[StageType, List[PromptProfile]] = {}
    for stage_type in stages_to_evolve:
        # Start with base prompt for this stage
        base_prompt_name = base_pipeline.get_stage_by_type(stage_type).prompt_profile_name
        base_prompt = prompt_profiles.get(base_prompt_name) or get_prompt_profile(base_prompt_name)
        populations[stage_type] = [base_prompt]

        # Generate initial population variants
        for i in range(population_size - 1):
            variant = _mutate_prompt(base_prompt, stage_type, mutation_rate)
            variant.name = f"{base_prompt.name}_gen0_variant{i}"
            populations[stage_type].append(variant)

    all_generations: List[Dict[str, Any]] = []

    # Evolution loop
    for generation in range(generations):
        logger.info(f"Generation {generation + 1}/{generations}")

        generation_data: Dict[str, Any] = {
            "generation": generation,
            "populations": {},
            "scores": {},
            "best_prompts": {},
        }

        # Evaluate all prompts in current population
        for stage_type in stages_to_evolve:
            logger.info(f"Evaluating prompts for stage: {stage_type}")

            stage_population = populations[stage_type]
            prompt_scores: Dict[str, float] = {}

            for prompt in tqdm(stage_population, desc=f"Evaluating {stage_type}"):
                # Evaluate prompt on all topics
                topic_scores = []
                for topic in topics:
                    # Create temporary pipeline with this prompt
                    temp_pipeline = _create_pipeline_with_prompt(
                        base_pipeline,
                        stage_type,
                        prompt.name,
                    )

                    # Run pipeline
                    try:
                        summary = run_pipeline_for_topic(
                            topic=topic,
                            pipeline=temp_pipeline,
                            prompt_profiles={prompt.name: prompt, **prompt_profiles},
                            evaluators=[
                                RuleBasedEvaluator(),
                                LLMJudgeEvaluator(),
                                ConsistencyEvaluator(),
                            ],
                        )
                        topic_scores.append(summary.aggregated_score)
                    except Exception as e:
                        logger.error(f"Evaluation failed for {prompt.name}: {e}")
                        topic_scores.append(0.0)

                # Average score across topics
                avg_score = sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
                prompt_scores[prompt.name] = avg_score

            generation_data["scores"][stage_type.value] = prompt_scores

            # Select top k
            sorted_prompts = sorted(
                stage_population,
                key=lambda p: prompt_scores.get(p.name, 0.0),
                reverse=True,
            )
            top_prompts = sorted_prompts[:top_k]
            generation_data["best_prompts"][stage_type.value] = [
                p.name for p in top_prompts
            ]

            # Generate next generation
            next_population: List[PromptProfile] = []

            # Keep top k
            next_population.extend(top_prompts)

            # Generate new variants via meta-prompts
            best_prompt = top_prompts[0]
            for i in range(population_size - top_k):
                if random.random() < 0.5:
                    # Use meta-improve prompt
                    improved = _improve_prompt_via_meta(
                        best_prompt,
                        stage_type,
                        prompt_scores,
                        topics,
                    )
                    if improved:
                        improved.name = f"{best_prompt.name}_gen{generation + 1}_improved{i}"
                        next_population.append(improved)
                    else:
                        # Fallback to mutation
                        mutated = _mutate_prompt(best_prompt, stage_type, mutation_rate)
                        mutated.name = f"{best_prompt.name}_gen{generation + 1}_mut{i}"
                        next_population.append(mutated)
                else:
                    # Direct mutation
                    mutated = _mutate_prompt(best_prompt, stage_type, mutation_rate)
                    mutated.name = f"{best_prompt.name}_gen{generation + 1}_mut{i}"
                    next_population.append(mutated)

            populations[stage_type] = next_population
            generation_data["populations"][stage_type.value] = [
                {
                    "name": p.name,
                    "template": p.template[:200] + "..." if len(p.template) > 200 else p.template,
                }
                for p in next_population
            ]

        # Save generation
        gen_dir = Path(f"data/generations/prompt_mode/generation_{generation}")
        gen_dir.mkdir(parents=True, exist_ok=True)

        gen_file = gen_dir / f"generation_{generation}.yaml"
        with open(gen_file, "w", encoding="utf-8") as f:
            yaml.dump(generation_data, f, default_flow_style=False)

        all_generations.append(generation_data)

    # Final best prompts
    final_best: Dict[str, PromptProfile] = {}
    for stage_type in stages_to_evolve:
        final_population = populations[stage_type]
        if final_population:
            # Re-evaluate to get final scores
            final_scores = {}
            for prompt in final_population:
                topic_scores = []
                for topic in topics:
                    temp_pipeline = _create_pipeline_with_prompt(
                        base_pipeline,
                        stage_type,
                        prompt.name,
                    )
                    try:
                        summary = run_pipeline_for_topic(
                            topic=topic,
                            pipeline=temp_pipeline,
                            prompt_profiles={prompt.name: prompt, **prompt_profiles},
                        )
                        topic_scores.append(summary.aggregated_score)
                    except Exception:
                        topic_scores.append(0.0)
                final_scores[prompt.name] = (
                    sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
                )

            best = max(final_population, key=lambda p: final_scores.get(p.name, 0.0))
            final_best[stage_type.value] = best

    return {
        "evolution_id": evolution_id,
        "generations": all_generations,
        "final_best_prompts": {
            stage: {
                "name": prompt.name,
                "template": prompt.template,
                "category": prompt.category,
            }
            for stage, prompt in final_best.items()
        },
    }


def _create_pipeline_with_prompt(
    base_pipeline: PipelineConfig,
    stage_type: StageType,
    prompt_name: str,
) -> PipelineConfig:
    """Create a pipeline with a specific prompt for a stage."""
    new_stages = []
    for stage in base_pipeline.stages:
        if stage.stage_type == stage_type:
            new_stage = stage.model_copy()
            new_stage.prompt_profile_name = prompt_name
            new_stages.append(new_stage)
        else:
            new_stages.append(stage)

    return PipelineConfig(
        pipeline_id=f"{base_pipeline.pipeline_id}_temp",
        stages=new_stages,
        name=base_pipeline.name,
        description=base_pipeline.description,
    )


def _mutate_prompt(
    base_prompt: PromptProfile,
    stage_type: StageType,
    mutation_rate: float,
) -> PromptProfile:
    """Create a mutated variant of a prompt."""
    template = base_prompt.template

    # Simple mutations: add/remove instructions, modify structure hints
    mutations = [
        lambda t: t.replace("JSON format", "structured JSON format"),
        lambda t: t.replace("Provide JSON", "Provide detailed JSON"),
        lambda t: t + "\n\nEnsure all outputs are well-structured and complete.",
        lambda t: t.replace("list of", "comprehensive list of"),
    ]

    if random.random() < mutation_rate:
        mutation = random.choice(mutations)
        template = mutation(template)

    return PromptProfile(
        name=f"{base_prompt.name}_mutated",
        category=base_prompt.category,
        template=template,
        description=base_prompt.description,
        expected_output_format=base_prompt.expected_output_format,
    )


def _improve_prompt_via_meta(
    prompt: PromptProfile,
    stage_type: StageType,
    scores: Dict[str, float],
    topics: List[Topic],
) -> Optional[PromptProfile]:
    """Use meta-prompt to improve a prompt."""
    try:
        # Get appropriate meta-improve prompt
        meta_prompt_name = f"meta_improve_{stage_type.value}_prompt_v1"
        if meta_prompt_name not in PROMPT_REGISTRY:
            meta_prompt_name = "meta_analyze_prompt_weaknesses_v1"

        meta_prompt = get_prompt_profile(meta_prompt_name)

        # Prepare context
        results_summary = f"Average score: {scores.get(prompt.name, 0.0):.3f}"
        weaknesses = "Performance could be improved"  # TODO: Extract from evaluations

        # Use safe rendering
        meta_context = {
            "current_prompt": prompt.template,
            "results": results_summary,
            "evaluation_scores": str(scores),
            "weaknesses": weaknesses,
        }
        prompt_text = render_prompt(meta_prompt.template, meta_context)

        response = call_llm_sync(
            model=get_settings().default_model,
            prompt=prompt_text,
            temperature=0.7,
        )

        data = parse_json_response(response["content"])
        improved_template = data.get("improved_prompt")

        if improved_template and improved_template != prompt.template:
            return PromptProfile(
                name=f"{prompt.name}_improved",
                category=prompt.category,
                template=improved_template,
                description=prompt.description,
                expected_output_format=prompt.expected_output_format,
            )
    except Exception as e:
        logger.warning(f"Meta-improvement failed: {e}")

    return None

