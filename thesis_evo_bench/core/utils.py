"""Utility functions for the evolution system."""

import json
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List

from thesis_evo_bench.core.models import (
    PipelineConfig,
    StageConfig,
    StageType,
    Topic,
)


def render_prompt(template: str, context: Dict[str, Any]) -> str:
    """
    Safely render a prompt template with context variables.

    Replaces {variable} placeholders with values from context.
    If a variable is missing, replaces it with an empty string.

    Args:
        template: Template string with {variable} placeholders
        context: Dictionary of variable values

    Returns:
        Rendered prompt string
    """
    # Find all {variable} patterns
    pattern = r"\{([^}]+)\}"
    matches = re.findall(pattern, template)

    # Build replacement dict with safe defaults
    replacements = {}
    for var_name in matches:
        if var_name in context:
            replacements[var_name] = str(context[var_name])
        else:
            # Missing variable - use empty string
            replacements[var_name] = ""

    # Perform replacements
    result = template
    for var_name, value in replacements.items():
        result = result.replace(f"{{{var_name}}}", value)

    return result


def load_topics_from_yaml(file_path: Path) -> List[Topic]:
    """Load topics from YAML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    topics = []
    if isinstance(data, list):
        for item in data:
            topics.append(Topic(**item))
    elif isinstance(data, dict) and "topics" in data:
        for item in data["topics"]:
            topics.append(Topic(**item))

    return topics


def load_pipeline_from_yaml(file_path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    stages = []
    for stage_data in data.get("stages", []):
        stage_data["stage_type"] = StageType(stage_data["stage_type"])
        stages.append(StageConfig(**stage_data))

    return PipelineConfig(
        pipeline_id=data.get("pipeline_id", "default"),
        stages=stages,
        name=data.get("name"),
        description=data.get("description"),
        metadata=data.get("metadata", {}),
    )


def load_pipelines_from_yaml(file_path: Path) -> List[PipelineConfig]:
    """Load multiple pipeline configurations from YAML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    pipelines = []
    if isinstance(data, list):
        for pipeline_data in data:
            stages = []
            for stage_data in pipeline_data.get("stages", []):
                stage_data["stage_type"] = StageType(stage_data["stage_type"])
                stages.append(StageConfig(**stage_data))
            pipelines.append(
                PipelineConfig(
                    pipeline_id=pipeline_data.get("pipeline_id", f"pipeline_{len(pipelines)}"),
                    stages=stages,
                    name=pipeline_data.get("name"),
                    description=pipeline_data.get("description"),
                    metadata=pipeline_data.get("metadata", {}),
                ),
            )
    elif isinstance(data, dict) and "pipelines" in data:
        for pipeline_data in data["pipelines"]:
            stages = []
            for stage_data in pipeline_data.get("stages", []):
                stage_data["stage_type"] = StageType(stage_data["stage_type"])
                stages.append(StageConfig(**stage_data))
            pipelines.append(
                PipelineConfig(
                    pipeline_id=pipeline_data.get("pipeline_id", f"pipeline_{len(pipelines)}"),
                    stages=stages,
                    name=pipeline_data.get("name"),
                    description=pipeline_data.get("description"),
                    metadata=pipeline_data.get("metadata", {}),
                ),
            )

    return pipelines


def save_pipeline_to_yaml(pipeline: PipelineConfig, file_path: Path) -> None:
    """Save pipeline configuration to YAML file."""
    data = {
        "pipeline_id": pipeline.pipeline_id,
        "name": pipeline.name,
        "description": pipeline.description,
        "stages": [
            {
                "stage_type": stage.stage_type.value,
                "prompt_profile_name": stage.prompt_profile_name,
                "temperature": stage.temperature,
                "max_tokens": stage.max_tokens,
                "enabled": stage.enabled,
                "depends_on": stage.depends_on,
                "metadata": stage.metadata,
            }
            for stage in pipeline.stages
        ],
        "metadata": pipeline.metadata,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def parse_json_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling code blocks."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            if json_end > json_start:
                return json.loads(response_text[json_start:json_end].strip())
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            if json_end > json_start:
                return json.loads(response_text[json_start:json_end].strip())

        # Last resort: try to find JSON object in text
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response_text[start:end])

        raise ValueError("Could not parse JSON from response")
