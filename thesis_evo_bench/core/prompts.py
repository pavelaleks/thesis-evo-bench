"""Registry of 25 prompt profiles for generation, evaluation, and evolution."""

from thesis_evo_bench.core.models import PromptProfile

# ============================================================================
# (1) 5 GENERATION PROMPTS
# ============================================================================

SEMANTIC_V1 = PromptProfile(
    name="semantic_v1",
    category="generation",
    description="Generate semantic analysis of topic",
    template="""Analyze the following topic semantically and provide a structured analysis in JSON format.

Topic: {topic_title}
Domain: {topic_domain}
Description: {topic_description}

Provide JSON with:
- key_concepts: list of main concepts
- relationships: list of concept relationships
- scope: scope boundaries
- complexity_level: estimated complexity (1-10)
- semantic_structure: hierarchical structure of concepts""",
)

OUTLINE_V1 = PromptProfile(
    name="outline_v1",
    category="generation",
    description="Generate thesis outline structure",
    template="""Generate a comprehensive thesis outline in JSON format based on the semantic analysis.

Semantic Analysis: {semantic_analysis}
Topic: {topic_title}

Provide JSON with:
- title: thesis title
- abstract: brief abstract
- sections: list of sections with:
  - section_title
  - section_type (intro/chapter/review/conclusion)
  - subsections: list of subsections
  - estimated_length: word count estimate""",
)

INTRO_V1 = PromptProfile(
    name="intro_v1",
    category="generation",
    description="Generate thesis introduction",
    template="""Write a thesis introduction section in JSON format.

Topic: {topic_title}
Outline: {outline_structure}
Semantic Analysis: {semantic_analysis}

Provide JSON with:
- introduction_text: full introduction text
- problem_statement: clear problem statement
- objectives: list of research objectives
- methodology_overview: brief methodology overview
- structure_preview: preview of thesis structure""",
)

CHAPTER1_V1 = PromptProfile(
    name="chapter1_v1",
    category="generation",
    description="Generate first main chapter",
    template="""Write the first main chapter of the thesis in JSON format.

Topic: {topic_title}
Outline: {outline_structure}
Introduction: {introduction_text}
Previous Context: {previous_context}

Provide JSON with:
- chapter_title: chapter title
- chapter_content: full chapter text
- key_points: list of main points covered
- citations_needed: list of areas needing citations
- figures_tables: list of suggested figures/tables""",
)

REVIEW_V1 = PromptProfile(
    name="review_v1",
    category="generation",
    description="Generate review and revision suggestions",
    template="""Review the following thesis content and provide revision suggestions in JSON format.

Topic: {topic_title}
Content to Review: {content_to_review}
Full Context: {full_context}

Provide JSON with:
- overall_quality_score: score 0-1
- strengths: list of strengths
- weaknesses: list of weaknesses
- specific_revisions: list of specific revision suggestions with:
  - location: where to revise
  - issue: what needs fixing
  - suggestion: how to fix
- consistency_check: consistency with previous sections
- completeness_check: missing elements""",
)

# ============================================================================
# (2) 4 EVALUATION PROMPTS
# ============================================================================

EVAL_GENERAL_QUALITY_V1 = PromptProfile(
    name="eval_general_quality_v1",
    category="evaluation",
    description="Evaluate general quality of generated content",
    template="""Evaluate the general quality of the following content on a scale of 0.0 to 1.0.

Content: {content}
Context: {context}
Stage: {stage_type}

Provide JSON with:
- score: float between 0.0 and 1.0
- reasoning: explanation of score
- criteria_scores: object with scores for:
  - clarity: clarity of writing
  - coherence: logical coherence
  - completeness: completeness of content
  - relevance: relevance to topic
  - structure: structural quality""",
)

EVAL_HALLUCINATION_RISK_V1 = PromptProfile(
    name="eval_hallucination_risk_v1",
    category="evaluation",
    description="Evaluate risk of hallucinations or factual errors",
    template="""Evaluate the risk of hallucinations or factual errors in the following content.

Content: {content}
Topic: {topic_title}
Domain: {topic_domain}

Provide JSON with:
- hallucination_risk_score: float 0.0-1.0 (lower is better)
- reasoning: explanation
- flagged_claims: list of claims that may need verification
- confidence_level: overall confidence in factual accuracy
- recommendations: suggestions to reduce risk""",
)

EVAL_STRUCTURE_COHERENCE_V1 = PromptProfile(
    name="eval_structure_coherence_v1",
    category="evaluation",
    description="Evaluate structural coherence and organization",
    template="""Evaluate the structural coherence and organization of the following content.

Content: {content}
Expected Structure: {expected_structure}
Previous Sections: {previous_sections}

Provide JSON with:
- coherence_score: float 0.0-1.0
- reasoning: explanation
- structure_adherence: how well it follows expected structure
- flow_quality: quality of flow between sections
- organization_issues: list of organizational problems
- suggestions: improvement suggestions""",
)

EVAL_CROSS_CONSISTENCY_V1 = PromptProfile(
    name="eval_cross_consistency_v1",
    category="evaluation",
    description="Evaluate consistency across different sections",
    template="""Evaluate consistency between different sections of the thesis.

Current Section: {current_section}
Previous Sections: {previous_sections}
Topic: {topic_title}

Provide JSON with:
- consistency_score: float 0.0-1.0
- reasoning: explanation
- inconsistencies: list of detected inconsistencies
- terminology_consistency: consistency of terminology
- narrative_consistency: consistency of narrative
- recommendations: suggestions to improve consistency""",
)

# ============================================================================
# (3) 8 PROMPT-EVOLUTION META-PROMPTS
# ============================================================================

META_IMPROVE_SEMANTIC_PROMPT_V1 = PromptProfile(
    name="meta_improve_semantic_prompt_v1",
    category="meta_prompt_evolution",
    description="Improve semantic analysis prompt based on results",
    template="""Analyze the following prompt and its results, then generate an improved version.

Current Prompt: {current_prompt}
Results: {results}
Evaluation Scores: {evaluation_scores}
Weaknesses: {weaknesses}

Provide JSON with:
- improved_prompt: improved prompt text
- changes_made: list of changes made
- rationale: why these changes should help
- expected_improvements: what should improve""",
)

META_IMPROVE_OUTLINE_PROMPT_V1 = PromptProfile(
    name="meta_improve_outline_prompt_v1",
    category="meta_prompt_evolution",
    description="Improve outline generation prompt",
    template="""Analyze and improve the outline generation prompt.

Current Prompt: {current_prompt}
Results: {results}
Evaluation Scores: {evaluation_scores}
Weaknesses: {weaknesses}

Provide JSON with:
- improved_prompt: improved prompt text
- changes_made: list of changes
- rationale: explanation
- expected_improvements: expected benefits""",
)

META_IMPROVE_INTRO_PROMPT_V1 = PromptProfile(
    name="meta_improve_intro_prompt_v1",
    category="meta_prompt_evolution",
    description="Improve introduction generation prompt",
    template="""Analyze and improve the introduction generation prompt.

Current Prompt: {current_prompt}
Results: {results}
Evaluation Scores: {evaluation_scores}
Weaknesses: {weaknesses}

Provide JSON with:
- improved_prompt: improved prompt text
- changes_made: list of changes
- rationale: explanation
- expected_improvements: expected benefits""",
)

META_IMPROVE_CHAPTER1_PROMPT_V1 = PromptProfile(
    name="meta_improve_chapter1_prompt_v1",
    category="meta_prompt_evolution",
    description="Improve chapter generation prompt",
    template="""Analyze and improve the chapter generation prompt.

Current Prompt: {current_prompt}
Results: {results}
Evaluation Scores: {evaluation_scores}
Weaknesses: {weaknesses}

Provide JSON with:
- improved_prompt: improved prompt text
- changes_made: list of changes
- rationale: explanation
- expected_improvements: expected benefits""",
)

META_ANALYZE_PROMPT_WEAKNESSES_V1 = PromptProfile(
    name="meta_analyze_prompt_weaknesses_v1",
    category="meta_prompt_evolution",
    description="Analyze weaknesses in a prompt",
    template="""Analyze the weaknesses of the following prompt based on evaluation results.

Prompt: {prompt}
Evaluation Results: {evaluation_results}
Performance Metrics: {performance_metrics}

Provide JSON with:
- weaknesses: list of identified weaknesses
- severity: severity of each weakness (high/medium/low)
- impact: impact on output quality
- root_causes: potential root causes
- improvement_priorities: prioritized list of improvements""",
)

META_GENERATE_PROMPT_MUTATION_V1 = PromptProfile(
    name="meta_generate_prompt_mutation_v1",
    category="meta_prompt_evolution",
    description="Generate mutated variants of a prompt",
    template="""Generate mutated variants of the following prompt to explore different approaches.

Base Prompt: {base_prompt}
Mutation Strategy: {mutation_strategy}
Number of Variants: {num_variants}

Provide JSON with:
- mutations: list of mutated prompts, each with:
  - variant_prompt: mutated prompt text
  - mutation_type: type of mutation applied
  - expected_effect: expected effect of mutation""",
)

META_COMPARE_PROMPTS_V1 = PromptProfile(
    name="meta_compare_prompts_v1",
    category="meta_prompt_evolution",
    description="Compare multiple prompt variants",
    template="""Compare multiple prompt variants and identify the best one.

Prompts: {prompts}
Results for Each: {results}
Evaluation Scores: {evaluation_scores}

Provide JSON with:
- comparison: detailed comparison
- best_prompt: name of best prompt
- ranking: ranked list of prompts
- strengths_weaknesses: strengths and weaknesses of each
- recommendation: recommendation with rationale""",
)

META_SELECT_BEST_PROMPT_V1 = PromptProfile(
    name="meta_select_best_prompt_v1",
    category="meta_prompt_evolution",
    description="Select best prompt from population",
    template="""Select the best prompt from a population based on comprehensive evaluation.

Prompts: {prompts}
Evaluation Scores: {evaluation_scores}
Performance Data: {performance_data}

Provide JSON with:
- best_prompt: name of best prompt
- selection_criteria: criteria used
- justification: why this prompt is best
- top_k: list of top k prompts ranked
- insights: insights about what makes prompts effective""",
)

# ============================================================================
# (4) 5 PIPELINE-EVOLUTION META-PROMPTS
# ============================================================================

META_ANALYZE_PIPELINE_V1 = PromptProfile(
    name="meta_analyze_pipeline_v1",
    category="meta_pipeline_evolution",
    description="Analyze pipeline structure and performance",
    template="""Analyze the following pipeline structure and its performance.

Pipeline: {pipeline_structure}
Evaluation Results: {evaluation_results}
Performance Metrics: {performance_metrics}

Provide JSON with:
- strengths: list of pipeline strengths
- weaknesses: list of weaknesses
- bottlenecks: identified bottlenecks
- improvement_opportunities: opportunities for improvement
- recommendations: specific recommendations""",
)

META_GENERATE_NEW_PIPELINE_V1 = PromptProfile(
    name="meta_generate_new_pipeline_v1",
    category="meta_pipeline_evolution",
    description="Generate new pipeline variants",
    template="""Generate new pipeline structure variants based on analysis.

Base Pipelines: {base_pipelines}
Analysis: {analysis}
Requirements: {requirements}
Number of Variants: {num_variants}

Provide JSON with:
- new_pipelines: list of new pipeline structures, each with:
  - pipeline_id: unique identifier
  - stages: list of stage configurations
  - rationale: why this structure should work
  - expected_benefits: expected benefits""",
)

META_MUTATE_PIPELINE_V1 = PromptProfile(
    name="meta_mutate_pipeline_v1",
    category="meta_pipeline_evolution",
    description="Generate mutated pipeline variants",
    template="""Generate mutated variants of a pipeline structure.

Base Pipeline: {base_pipeline}
Mutation Strategy: {mutation_strategy}
Number of Variants: {num_variants}

Provide JSON with:
- mutations: list of mutated pipelines, each with:
  - pipeline_id: unique identifier
  - stages: mutated stage configuration
  - mutation_type: type of mutation
  - expected_effect: expected effect""",
)

META_CROSSOVER_PIPELINE_V1 = PromptProfile(
    name="meta_crossover_pipeline_v1",
    category="meta_pipeline_evolution",
    description="Combine two pipelines via crossover",
    template="""Combine two pipeline structures via crossover to create new variants.

Parent Pipeline 1: {parent1}
Parent Pipeline 2: {parent2}
Crossover Strategy: {crossover_strategy}
Number of Offspring: {num_offspring}

Provide JSON with:
- offspring: list of crossover pipelines, each with:
  - pipeline_id: unique identifier
  - stages: combined stage configuration
  - inherited_from_parent1: features from parent 1
  - inherited_from_parent2: features from parent 2
  - rationale: why this combination should work""",
)

META_CHECK_PIPELINE_DRIFT_V1 = PromptProfile(
    name="meta_check_pipeline_drift_v1",
    category="meta_pipeline_evolution",
    description="Check for pipeline drift or degradation",
    template="""Check for drift or degradation in pipeline performance across generations.

Current Generation: {current_generation}
Previous Generations: {previous_generations}
Performance Trends: {performance_trends}

Provide JSON with:
- drift_detected: boolean
- drift_type: type of drift if detected
- severity: severity of drift
- causes: potential causes
- recommendations: recommendations to address drift""",
)

# ============================================================================
# (5) 3 EVOLUTION-CONTROL PROMPTS
# ============================================================================

META_EVOLUTION_SUMMARY_V1 = PromptProfile(
    name="meta_evolution_summary_v1",
    category="evolution_control",
    description="Generate summary of evolution progress",
    template="""Generate a comprehensive summary of evolution progress.

Generation: {generation}
Population: {population}
Performance History: {performance_history}
Best Candidates: {best_candidates}

Provide JSON with:
- summary: overall summary
- progress: progress indicators
- trends: identified trends
- insights: key insights
- next_steps: recommended next steps""",
)

META_REGRESSION_CHECK_V1 = PromptProfile(
    name="meta_regression_check_v1",
    category="evolution_control",
    description="Check for performance regression",
    template="""Check for performance regression compared to previous generations.

Current Generation: {current_generation}
Previous Generations: {previous_generations}
Performance Metrics: {performance_metrics}

Provide JSON with:
- regression_detected: boolean
- regression_severity: severity if detected
- affected_areas: areas affected
- causes: potential causes
- recovery_strategy: strategy to recover""",
)

META_GENERATE_EXPERIMENT_REPORT_V1 = PromptProfile(
    name="meta_generate_experiment_report_v1",
    category="evolution_control",
    description="Generate comprehensive experiment report",
    template="""Generate a comprehensive experiment report for the evolution run.

Experiment Config: {experiment_config}
All Generations: {all_generations}
Final Results: {final_results}
Best Candidates: {best_candidates}

Provide JSON with:
- executive_summary: high-level summary
- methodology: methodology used
- results: detailed results
- analysis: analysis of results
- conclusions: conclusions
- recommendations: recommendations for future work""",
)

# ============================================================================
# PROMPT REGISTRY
# ============================================================================

PROMPT_REGISTRY: dict[str, PromptProfile] = {
    # Generation prompts
    "semantic_v1": SEMANTIC_V1,
    "outline_v1": OUTLINE_V1,
    "intro_v1": INTRO_V1,
    "chapter1_v1": CHAPTER1_V1,
    "review_v1": REVIEW_V1,
    # Evaluation prompts
    "eval_general_quality_v1": EVAL_GENERAL_QUALITY_V1,
    "eval_hallucination_risk_v1": EVAL_HALLUCINATION_RISK_V1,
    "eval_structure_coherence_v1": EVAL_STRUCTURE_COHERENCE_V1,
    "eval_cross_consistency_v1": EVAL_CROSS_CONSISTENCY_V1,
    # Prompt evolution meta-prompts
    "meta_improve_semantic_prompt_v1": META_IMPROVE_SEMANTIC_PROMPT_V1,
    "meta_improve_outline_prompt_v1": META_IMPROVE_OUTLINE_PROMPT_V1,
    "meta_improve_intro_prompt_v1": META_IMPROVE_INTRO_PROMPT_V1,
    "meta_improve_chapter1_prompt_v1": META_IMPROVE_CHAPTER1_PROMPT_V1,
    "meta_analyze_prompt_weaknesses_v1": META_ANALYZE_PROMPT_WEAKNESSES_V1,
    "meta_generate_prompt_mutation_v1": META_GENERATE_PROMPT_MUTATION_V1,
    "meta_compare_prompts_v1": META_COMPARE_PROMPTS_V1,
    "meta_select_best_prompt_v1": META_SELECT_BEST_PROMPT_V1,
    # Pipeline evolution meta-prompts
    "meta_analyze_pipeline_v1": META_ANALYZE_PIPELINE_V1,
    "meta_generate_new_pipeline_v1": META_GENERATE_NEW_PIPELINE_V1,
    "meta_mutate_pipeline_v1": META_MUTATE_PIPELINE_V1,
    "meta_crossover_pipeline_v1": META_CROSSOVER_PIPELINE_V1,
    "meta_check_pipeline_drift_v1": META_CHECK_PIPELINE_DRIFT_V1,
    # Evolution control prompts
    "meta_evolution_summary_v1": META_EVOLUTION_SUMMARY_V1,
    "meta_regression_check_v1": META_REGRESSION_CHECK_V1,
    "meta_generate_experiment_report_v1": META_GENERATE_EXPERIMENT_REPORT_V1,
}


def get_prompt_profile(name: str) -> PromptProfile:
    """Get a prompt profile by name."""
    if name not in PROMPT_REGISTRY:
        raise ValueError(f"Prompt profile '{name}' not found in registry")
    return PROMPT_REGISTRY[name]


def list_prompts_by_category(category: str) -> list[PromptProfile]:
    """List all prompts in a given category."""
    return [
        profile
        for profile in PROMPT_REGISTRY.values()
        if profile.category == category
    ]

