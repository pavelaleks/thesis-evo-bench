"""Command-line interface for thesis evolution benchmark."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import yaml

from thesis_evo_bench.core.evolution_pipeline_mode import run_pipeline_evolution
from thesis_evo_bench.core.evolution_prompt_mode import run_prompt_evolution
from thesis_evo_bench.core.models import PipelineConfig, Topic
from thesis_evo_bench.core.pipeline_executor import run_pipeline_for_topic
from thesis_evo_bench.core.prompts import PROMPT_REGISTRY
from thesis_evo_bench.core.reporting import (
    build_pipeline_evolution_report,
    build_prompt_evolution_report,
)
from thesis_evo_bench.core.utils import (
    load_pipeline_from_yaml,
    load_pipelines_from_yaml,
    load_topics_from_yaml,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_prompt_evolution_cmd(args: argparse.Namespace) -> None:
    """Run prompt evolution command."""
    logger.info("Starting prompt evolution...")

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load topics
    topics = load_topics_from_yaml(Path(args.topics))
    logger.info(f"Loaded {len(topics)} topics")

    # Load pipeline
    pipeline = load_pipeline_from_yaml(Path(args.pipeline))
    logger.info(f"Loaded pipeline: {pipeline.pipeline_id}")

    # Get prompt profiles
    prompt_profiles = PROMPT_REGISTRY.copy()

    # Run evolution
    results = run_prompt_evolution(
        base_pipeline=pipeline,
        topics=topics,
        prompt_profiles=prompt_profiles,
        config=config,
    )

    # Generate report
    report_dir = Path("data/reports/prompt_evolution")
    report = build_prompt_evolution_report(results, report_dir)
    logger.info(f"Evolution complete. Report saved to {report_dir}")
    logger.info(f"Best prompts: {list(report['final_best_prompts'].keys())}")


def run_pipeline_evolution_cmd(args: argparse.Namespace) -> None:
    """Run pipeline evolution command."""
    logger.info("Starting pipeline evolution...")

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load topics
    topics = load_topics_from_yaml(Path(args.topics))
    logger.info(f"Loaded {len(topics)} topics")

    # Load initial pipelines
    initial_pipelines = load_pipelines_from_yaml(Path(args.initial_population))
    logger.info(f"Loaded {len(initial_pipelines)} initial pipelines")

    # Get base prompts (from first pipeline or config)
    base_prompts = {}
    if initial_pipelines:
        for stage in initial_pipelines[0].stages:
            base_prompts[stage.stage_type.value] = stage.prompt_profile_name

    # Run evolution
    results = run_pipeline_evolution(
        base_prompts=base_prompts,
        initial_pipelines=initial_pipelines,
        topics=topics,
        config=config,
    )

    # Generate report
    report_dir = Path("data/reports/pipeline_evolution")
    report = build_pipeline_evolution_report(results, report_dir)
    logger.info(f"Evolution complete. Report saved to {report_dir}")
    logger.info(f"Best pipeline: {report['final_best_pipeline']['pipeline_id']}")
    logger.info(f"Final score: {report['final_best_pipeline']['final_score']:.3f}")


def run_single_pipeline_cmd(args: argparse.Namespace) -> None:
    """Run single pipeline command."""
    logger.info("Running single pipeline...")

    # Load pipeline
    pipeline = load_pipeline_from_yaml(Path(args.pipeline))
    logger.info(f"Loaded pipeline: {pipeline.pipeline_id}")

    # Create topic
    topic = Topic(
        title=args.topic,
        domain=args.domain or "general",
        description=args.description,
    )

    # Get prompt profiles
    prompt_profiles = PROMPT_REGISTRY.copy()

    # Run pipeline
    summary = run_pipeline_for_topic(
        topic=topic,
        pipeline=pipeline,
        prompt_profiles=prompt_profiles,
    )

    logger.info(f"Pipeline execution complete. Run ID: {summary.run_id}")
    logger.info(f"Aggregated score: {summary.aggregated_score:.3f}")
    logger.info(f"Execution time: {summary.execution_time_seconds:.2f}s")
    logger.info(f"Results saved to data/runs/{summary.run_id}_summary.json")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Thesis Evolution Benchmark - Evolutionary optimization for thesis generation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prompt evolution command
    prompt_parser = subparsers.add_parser(
        "run-prompt-evolution",
        help="Run prompt evolution (Mode 1)",
    )
    prompt_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evolution configuration YAML",
    )
    prompt_parser.add_argument(
        "--topics",
        type=str,
        required=True,
        help="Path to topics YAML file",
    )
    prompt_parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Path to fixed pipeline YAML file",
    )

    # Pipeline evolution command
    pipeline_parser = subparsers.add_parser(
        "run-pipeline-evolution",
        help="Run pipeline evolution (Mode 2)",
    )
    pipeline_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evolution configuration YAML",
    )
    pipeline_parser.add_argument(
        "--topics",
        type=str,
        required=True,
        help="Path to topics YAML file",
    )
    pipeline_parser.add_argument(
        "--initial-population",
        type=str,
        required=True,
        help="Path to initial pipeline population YAML file",
    )

    # Single pipeline command
    single_parser = subparsers.add_parser(
        "run-single-pipeline",
        help="Run a single pipeline for a topic",
    )
    single_parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Path to pipeline YAML file",
    )
    single_parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic title",
    )
    single_parser.add_argument(
        "--domain",
        type=str,
        help="Topic domain (optional)",
    )
    single_parser.add_argument(
        "--description",
        type=str,
        help="Topic description (optional)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "run-prompt-evolution":
            run_prompt_evolution_cmd(args)
        elif args.command == "run-pipeline-evolution":
            run_pipeline_evolution_cmd(args)
        elif args.command == "run-single-pipeline":
            run_single_pipeline_cmd(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

