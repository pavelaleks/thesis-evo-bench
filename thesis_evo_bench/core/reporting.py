"""Reporting utilities for evolution results."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def build_pipeline_score_matrix(
    pipeline_ids: List[str],
    topics: List[str],
    scores: Dict[str, Dict[str, float]],  # pipeline_id -> topic -> score
) -> pd.DataFrame:
    """
    Build a score matrix DataFrame.

    Args:
        pipeline_ids: List of pipeline IDs
        topics: List of topic identifiers
        scores: Nested dictionary of scores

    Returns:
        DataFrame with pipelines as rows and topics as columns
    """
    data = []
    for pipeline_id in pipeline_ids:
        row = {"pipeline_id": pipeline_id}
        for topic in topics:
            row[topic] = scores.get(pipeline_id, {}).get(topic, 0.0)
        data.append(row)

    df = pd.DataFrame(data)
    df.set_index("pipeline_id", inplace=True)
    return df


def summarize_generation(
    generation_data: Dict[str, Any],
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Summarize a generation's results.

    Args:
        generation_data: Generation data dictionary
        output_file: Optional file to save summary to

    Returns:
        Summary dictionary
    """
    generation = generation_data.get("generation", 0)
    scores = generation_data.get("scores", {})

    summary = {
        "generation": generation,
        "num_candidates": len(scores),
        "best_score": max(scores.values()) if scores else 0.0,
        "worst_score": min(scores.values()) if scores else 0.0,
        "average_score": sum(scores.values()) / len(scores) if scores else 0.0,
        "top_candidates": sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5],
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    return summary


def save_matrix_to_csv(
    matrix: pd.DataFrame,
    output_file: Path,
    include_stats: bool = True,
) -> None:
    """
    Save score matrix to CSV file.

    Args:
        matrix: DataFrame to save
        output_file: Output file path
        include_stats: Whether to include summary statistics
    """
    if include_stats:
        # Add summary statistics
        matrix_with_stats = matrix.copy()
        matrix_with_stats["mean"] = matrix.mean(axis=1)
        matrix_with_stats["std"] = matrix.std(axis=1)
        matrix_with_stats["min"] = matrix.min(axis=1)
        matrix_with_stats["max"] = matrix.max(axis=1)
        matrix_with_stats.to_csv(output_file)
    else:
        matrix.to_csv(output_file)


def generate_evolution_report(
    all_generations: List[Dict[str, Any]],
    output_dir: Path,
    report_name: str = "evolution_report",
) -> Dict[str, Any]:
    """
    Generate comprehensive evolution report.

    Args:
        all_generations: List of generation data dictionaries
        output_dir: Directory to save reports
        report_name: Base name for report files

    Returns:
        Report dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract trends
    generation_numbers = []
    best_scores = []
    average_scores = []

    for gen_data in all_generations:
        gen_num = gen_data.get("generation", 0)
        scores = gen_data.get("scores", {})

        generation_numbers.append(gen_num)
        if scores:
            best_scores.append(max(scores.values()))
            average_scores.append(sum(scores.values()) / len(scores))
        else:
            best_scores.append(0.0)
            average_scores.append(0.0)

    # Create trends DataFrame
    trends_df = pd.DataFrame({
        "generation": generation_numbers,
        "best_score": best_scores,
        "average_score": average_scores,
    })

    # Save trends CSV
    trends_file = output_dir / f"{report_name}_trends.csv"
    trends_df.to_csv(trends_file, index=False)

    # Generate summary
    report = {
        "total_generations": len(all_generations),
        "final_best_score": best_scores[-1] if best_scores else 0.0,
        "improvement": best_scores[-1] - best_scores[0] if len(best_scores) > 1 else 0.0,
        "trends": {
            "generations": generation_numbers,
            "best_scores": best_scores,
            "average_scores": average_scores,
        },
    }

    # Save report JSON
    report_file = output_dir / f"{report_name}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def build_prompt_evolution_report(
    evolution_results: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Build report for prompt evolution.

    Args:
        evolution_results: Results from prompt evolution
        output_dir: Directory to save report

    Returns:
        Report dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    generations = evolution_results.get("generations", [])
    final_best = evolution_results.get("final_best_prompts", {})

    # Build per-stage reports
    stage_reports = {}
    for stage, prompt_data in final_best.items():
        stage_reports[stage] = {
            "best_prompt": prompt_data["name"],
            "template_preview": prompt_data["template"][:500],
        }

    # Extract scores per generation per stage
    stage_scores: Dict[str, List[float]] = {}
    for gen_data in generations:
        scores = gen_data.get("scores", {})
        for stage, prompt_scores in scores.items():
            if stage not in stage_scores:
                stage_scores[stage] = []
            if prompt_scores:
                best = max(prompt_scores.values())
                stage_scores[stage].append(best)

    # Create DataFrame
    if stage_scores:
        max_len = max(len(scores) for scores in stage_scores.values())
        data = {}
        for stage, scores in stage_scores.items():
            # Pad to max length
            padded = scores + [0.0] * (max_len - len(scores))
            data[stage] = padded

        df = pd.DataFrame(data)
        df.index.name = "generation"
        csv_file = output_dir / "prompt_evolution_scores.csv"
        df.to_csv(csv_file)

    report = {
        "evolution_id": evolution_results.get("evolution_id"),
        "total_generations": len(generations),
        "final_best_prompts": final_best,
        "stage_scores": stage_scores,
    }

    report_file = output_dir / "prompt_evolution_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def build_pipeline_evolution_report(
    evolution_results: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Build report for pipeline evolution.

    Args:
        evolution_results: Results from pipeline evolution
        output_dir: Directory to save report

    Returns:
        Report dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    generations = evolution_results.get("generations", [])
    final_best = evolution_results.get("final_best_pipeline", {})

    # Extract scores per generation
    gen_scores = []
    for gen_data in generations:
        scores = gen_data.get("scores", {})
        if scores:
            gen_scores.append({
                "generation": gen_data.get("generation", 0),
                "best_score": max(scores.values()),
                "average_score": sum(scores.values()) / len(scores),
                "num_pipelines": len(scores),
            })

    if gen_scores:
        df = pd.DataFrame(gen_scores)
        csv_file = output_dir / "pipeline_evolution_scores.csv"
        df.to_csv(csv_file, index=False)

    report = {
        "evolution_id": evolution_results.get("evolution_id"),
        "total_generations": len(generations),
        "final_best_pipeline": final_best,
        "generation_scores": gen_scores,
    }

    report_file = output_dir / "pipeline_evolution_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report

