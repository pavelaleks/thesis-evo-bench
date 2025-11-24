"""Pipeline evolution engine for Mode 2: evolving pipeline structures with fixed prompts."""

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
    EvaluationResult,
    PipelineConfig,
    PipelinePopulation,
    RunSummary,
    StageConfig,
    StageType,
    Topic,
)
from thesis_evo_bench.core.pipeline_executor import run_pipeline_for_topic
from thesis_evo_bench.core.prompts import get_prompt_profile
from thesis_evo_bench.core.utils import parse_json_response, render_prompt, save_pipeline_to_yaml
from thesis_evo_bench.llm import call_llm_sync

logger = logging.getLogger(__name__)


def run_pipeline_evolution(
    base_prompts: Dict[str, str],  # stage_type -> prompt_profile_name
    initial_pipelines: List[PipelineConfig],
    topics: List[Topic],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run pipeline evolution with fixed optimized prompts.

    Args:
        base_prompts: Dictionary mapping stage types to prompt profile names
        initial_pipelines: Initial population of pipeline configurations
        topics: List of topics to evaluate on
        config: Evolution configuration with:
            - generations: number of generations
            - population_size: pipelines per generation
            - selection_top_k: top k to select each generation
            - mutation_rate: probability of mutation
            - crossover_rate: probability of crossover
            - random_seed: random seed

    Returns:
        Dictionary with evolution results and best pipeline
    """
    random.seed(config.get("random_seed", 42))

    generations = config.get("generations", 5)
    population_size = config.get("population_size", 15)
    top_k = config.get("selection_top_k", 5)
    mutation_rate = config.get("mutation_rate", 0.3)
    crossover_rate = config.get("crossover_rate", 0.4)

    settings = get_settings()
    evolution_id = str(uuid.uuid4())

    # Initialize population
    population = initial_pipelines[:population_size]
    if len(population) < population_size:
        # Generate additional pipelines by mutating existing ones
        while len(population) < population_size:
            base = random.choice(initial_pipelines)
            mutated = _mutate_pipeline(base, base_prompts, mutation_rate)
            mutated.pipeline_id = f"pipeline_gen0_{len(population)}"
            population.append(mutated)

    all_generations: List[Dict[str, Any]] = []

    # Evolution loop
    for generation in range(generations):
        logger.info(f"Generation {generation + 1}/{generations}")

        generation_data: Dict[str, Any] = {
            "generation": generation,
            "pipelines": [],
            "scores": {},
            "best_pipelines": [],
        }

        # Evaluate all pipelines
        pipeline_scores: Dict[str, float] = {}
        pipeline_evaluations: Dict[str, List[EvaluationResult]] = {}

        for pipeline in tqdm(population, desc="Evaluating pipelines"):
            topic_scores = []
            all_evals: List[EvaluationResult] = []

            for topic in topics:
                try:
                    summary = run_pipeline_for_topic(
                        topic=topic,
                        pipeline=pipeline,
                        prompt_profiles={
                            name: get_prompt_profile(name)
                            for name in base_prompts.values()
                        },
                        evaluators=[
                            RuleBasedEvaluator(),
                            LLMJudgeEvaluator(),
                            ConsistencyEvaluator(),
                        ],
                    )
                    topic_scores.append(summary.aggregated_score)
                    all_evals.extend(summary.evaluations)
                except Exception as e:
                    logger.error(f"Pipeline {pipeline.pipeline_id} failed: {e}")
                    topic_scores.append(0.0)

            avg_score = (
                sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
            )
            pipeline_scores[pipeline.pipeline_id] = avg_score
            pipeline_evaluations[pipeline.pipeline_id] = all_evals

        generation_data["scores"] = pipeline_scores

        # Select top k
        sorted_pipelines = sorted(
            population,
            key=lambda p: pipeline_scores.get(p.pipeline_id, 0.0),
            reverse=True,
        )
        top_pipelines = sorted_pipelines[:top_k]
        generation_data["best_pipelines"] = [p.pipeline_id for p in top_pipelines]

        # Check for drift
        if generation > 0:
            drift_result = _check_pipeline_drift(
                current_generation=generation,
                current_scores=pipeline_scores,
                previous_generations=all_generations,
            )
            generation_data["drift_check"] = drift_result

        # Generate next generation
        next_population: List[PipelineConfig] = []

        # Keep top k
        next_population.extend(top_pipelines)

        # Generate new pipelines
        while len(next_population) < population_size:
            if random.random() < crossover_rate and len(top_pipelines) >= 2:
                # Crossover
                parent1 = random.choice(top_pipelines)
                parent2 = random.choice([p for p in top_pipelines if p != parent1])
                offspring = _crossover_pipelines(
                    parent1,
                    parent2,
                    base_prompts,
                )
                offspring.pipeline_id = f"pipeline_gen{generation + 1}_cross{len(next_population)}"
                next_population.append(offspring)
            elif random.random() < 0.3:
                # Generate new via meta-prompt
                new_pipeline = _generate_new_pipeline_via_meta(
                    top_pipelines,
                    pipeline_scores,
                    base_prompts,
                )
                if new_pipeline:
                    new_pipeline.pipeline_id = f"pipeline_gen{generation + 1}_new{len(next_population)}"
                    next_population.append(new_pipeline)
                else:
                    # Fallback to mutation
                    base = random.choice(top_pipelines)
                    mutated = _mutate_pipeline(base, base_prompts, mutation_rate)
                    mutated.pipeline_id = f"pipeline_gen{generation + 1}_mut{len(next_population)}"
                    next_population.append(mutated)
            else:
                # Mutation
                base = random.choice(top_pipelines)
                mutated = _mutate_pipeline(base, base_prompts, mutation_rate)
                mutated.pipeline_id = f"pipeline_gen{generation + 1}_mut{len(next_population)}"
                next_population.append(mutated)

        population = next_population

        # Save generation
        gen_dir = Path(f"data/generations/pipeline_mode/generation_{generation}")
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save each pipeline
        for pipeline in population:
            pipeline_file = gen_dir / f"{pipeline.pipeline_id}.yaml"
            save_pipeline_to_yaml(pipeline, pipeline_file)

        # Save generation summary
        gen_summary = {
            "generation": generation,
            "pipeline_ids": [p.pipeline_id for p in population],
            "scores": pipeline_scores,
            "best_pipelines": [p.pipeline_id for p in top_pipelines],
        }
        gen_file = gen_dir / "generation_summary.yaml"
        with open(gen_file, "w", encoding="utf-8") as f:
            yaml.dump(gen_summary, f, default_flow_style=False)

        generation_data["pipelines"] = [
            {
                "pipeline_id": p.pipeline_id,
                "name": p.name,
                "num_stages": len(p.stages),
            }
            for p in population
        ]
        all_generations.append(generation_data)

    # Final best pipeline
    final_scores = {}
    for pipeline in population:
        topic_scores = []
        for topic in topics:
            try:
                summary = run_pipeline_for_topic(
                    topic=topic,
                    pipeline=pipeline,
                    prompt_profiles={
                        name: get_prompt_profile(name)
                        for name in base_prompts.values()
                    },
                )
                topic_scores.append(summary.aggregated_score)
            except Exception:
                topic_scores.append(0.0)
        final_scores[pipeline.pipeline_id] = (
            sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
        )

    best_pipeline = max(population, key=lambda p: final_scores.get(p.pipeline_id, 0.0))

    return {
        "evolution_id": evolution_id,
        "generations": all_generations,
        "final_best_pipeline": {
            "pipeline_id": best_pipeline.pipeline_id,
            "name": best_pipeline.name,
            "stages": [
                {
                    "stage_type": stage.stage_type.value,
                    "prompt_profile_name": stage.prompt_profile_name,
                    "temperature": stage.temperature,
                }
                for stage in best_pipeline.stages
            ],
            "final_score": final_scores.get(best_pipeline.pipeline_id, 0.0),
        },
    }


def _mutate_pipeline(
    base_pipeline: PipelineConfig,
    base_prompts: Dict[str, str],
    mutation_rate: float,
) -> PipelineConfig:
    """Create a mutated variant of a pipeline."""
    new_stages = []

    for stage in base_pipeline.stages:
        if random.random() < mutation_rate:
            # Mutate this stage
            new_stage = stage.model_copy()

            # Possible mutations
            mutations = [
                lambda s: setattr(s, "temperature", max(0.1, min(2.0, s.temperature + random.uniform(-0.2, 0.2)))),
                lambda s: setattr(s, "enabled", not s.enabled) if random.random() < 0.2 else None,
            ]

            mutation = random.choice(mutations)
            mutation(new_stage)

            new_stages.append(new_stage)
        else:
            new_stages.append(stage)

    # Add/remove stages
    if random.random() < mutation_rate * 0.5:
        # Add a stage
        stage_types = [StageType.SEMANTIC, StageType.OUTLINE, StageType.INTRO, StageType.CHAPTER1, StageType.REVIEW]
        new_stage_type = random.choice(stage_types)
        if not any(s.stage_type == new_stage_type for s in new_stages):
            new_stage = StageConfig(
                stage_type=new_stage_type,
                prompt_profile_name=base_prompts.get(new_stage_type.value, "semantic_v1"),
                temperature=0.7,
            )
            new_stages.append(new_stage)

    return PipelineConfig(
        pipeline_id=base_pipeline.pipeline_id + "_mutated",
        stages=new_stages,
        name=base_pipeline.name,
        description=base_pipeline.description,
    )


def _crossover_pipelines(
    parent1: PipelineConfig,
    parent2: PipelineConfig,
    base_prompts: Dict[str, str],
) -> PipelineConfig:
    """Combine two pipelines via crossover."""
    # Combine stages from both parents
    all_stages = {}
    for stage in parent1.stages:
        all_stages[stage.stage_type] = stage

    # Add stages from parent2, preferring better configurations
    for stage in parent2.stages:
        if stage.stage_type not in all_stages:
            all_stages[stage.stage_type] = stage
        elif random.random() < 0.5:
            # Replace with parent2's version
            all_stages[stage.stage_type] = stage

    return PipelineConfig(
        pipeline_id="crossover",
        stages=list(all_stages.values()),
        name=f"Crossover of {parent1.pipeline_id} and {parent2.pipeline_id}",
    )


def _generate_new_pipeline_via_meta(
    top_pipelines: List[PipelineConfig],
    scores: Dict[str, float],
    base_prompts: Dict[str, str],
) -> Optional[PipelineConfig]:
    """Use meta-prompt to generate a new pipeline."""
    try:
        meta_prompt = get_prompt_profile("meta_generate_new_pipeline_v1")

        # Prepare context
        base_pipelines_data = [
            {
                "pipeline_id": p.pipeline_id,
                "stages": [s.stage_type.value for s in p.stages],
                "score": scores.get(p.pipeline_id, 0.0),
            }
            for p in top_pipelines[:3]
        ]

        # Use safe rendering
        meta_context = {
            "base_pipelines": str(base_pipelines_data),
            "analysis": "Top performing pipelines analysis",
            "requirements": "Generate effective thesis generation pipeline",
            "num_variants": 1,
        }
        prompt_text = render_prompt(meta_prompt.template, meta_context)

        response = call_llm_sync(
            model=get_settings().default_model,
            prompt=prompt_text,
            temperature=0.8,
        )

        data = parse_json_response(response["content"])
        new_pipelines = data.get("new_pipelines", [])

        if new_pipelines:
            pipeline_data = new_pipelines[0]
            stages_data = pipeline_data.get("stages", [])

            stages = []
            for stage_data in stages_data:
                stage_type = StageType(stage_data.get("stage_type", "semantic"))
                prompt_name = base_prompts.get(stage_type.value, "semantic_v1")
                stages.append(
                    StageConfig(
                        stage_type=stage_type,
                        prompt_profile_name=prompt_name,
                        temperature=stage_data.get("temperature", 0.7),
                    ),
                )

            return PipelineConfig(
                pipeline_id="meta_generated",
                stages=stages,
                name="Meta-generated pipeline",
            )
    except Exception as e:
        logger.warning(f"Meta-generation failed: {e}")

    return None


def _check_pipeline_drift(
    current_generation: int,
    current_scores: Dict[str, float],
    previous_generations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check for pipeline drift using meta-prompt."""
    try:
        meta_prompt = get_prompt_profile("meta_check_pipeline_drift_v1")

        # Calculate average scores per generation
        gen_averages = []
        for gen_data in previous_generations[-3:]:  # Last 3 generations
            scores = gen_data.get("scores", {})
            if scores:
                gen_averages.append(sum(scores.values()) / len(scores))

        current_avg = sum(current_scores.values()) / len(current_scores) if current_scores else 0.0

        # Use safe rendering
        meta_context = {
            "current_generation": current_generation,
            "previous_generations": str(gen_averages),
            "performance_trends": f"Current: {current_avg:.3f}, Previous: {gen_averages[-1] if gen_averages else 0.0:.3f}",
        }
        prompt_text = render_prompt(meta_prompt.template, meta_context)

        response = call_llm_sync(
            model=get_settings().default_model,
            prompt=prompt_text,
            temperature=0.3,
        )

        data = parse_json_response(response["content"])
        return data
    except Exception as e:
        logger.warning(f"Drift check failed: {e}")
        return {"drift_detected": False, "error": str(e)}

