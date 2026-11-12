"""Transparent employee-to-project matching with semantic or TF-IDF similarity."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"
WEIGHTS = {
    "skills": 0.45,
    "experience": 0.25,
    "availability": 0.20,
    "career_interest": 0.10,
}  

_SKILL_ALIASES = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "js": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "postgres": "postgresql",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "continuous integration/continuous delivery": "ci/cd",
}


def parse_list(value: Any) -> list[str]:
    """Read a JSON-list CSV cell, with a forgiving delimiter fallback."""
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return [part.strip().strip("\"'") for part in re.split(r"[,;]", raw) if part.strip()]


def normalize_skill(skill: str) -> str:
    value = re.sub(r"\s+", " ", str(skill).strip().lower())
    value = _SKILL_ALIASES.get(value, value)
    return value


def _employee_text(employee: dict[str, Any]) -> str:
    skills = parse_list(employee.get("skills", []))
    return "Technical skills: " + ", ".join(skills)


def _project_text(project: dict[str, Any]) -> str:
    skills = parse_list(project.get("required_skills", []))
    return (
        f"Project domain: {project.get('domain', '')}. "
        f"Project: {project.get('project_name', '')}. "
        f"Requirements: {', '.join(skills)}. {project.get('description', '')}"
    )


class SimilarityIndex:
    """Precomputes one similarity value for every employee/project pair."""

    def __init__(self, employees: list[dict[str, Any]], projects: list[dict[str, Any]]):
        employee_texts = [_employee_text(employee) for employee in employees]
        project_texts = [_project_text(project) for project in projects]
        all_texts = employee_texts + project_texts
        self.backend_name = "SentenceTransformer"
        self.model_error: str | None = None
        self.similarities: np.ndarray

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME)
            vectors = model.encode(
                all_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            employee_vectors = vectors[: len(employees)]
            project_vectors = vectors[len(employees) :]
            self.similarities = np.clip(employee_vectors @ project_vectors.T, 0.0, 1.0)
        except Exception as exc:  # Model download/runtime errors must not block local use.
            self.backend_name = "TF-IDF fallback"
            self.model_error = f"{type(exc).__name__}: {exc}"
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            vectors = vectorizer.fit_transform(all_texts)
            employee_vectors = vectors[: len(employees)]
            project_vectors = vectors[len(employees) :]
            self.similarities = cosine_similarity(employee_vectors, project_vectors)

    def semantic_similarity(self, employee_index: int, project_index: int) -> float:
        return float(np.clip(self.similarities[employee_index, project_index], 0.0, 1.0))


def score_employee_project(
    employee: dict[str, Any],
    project: dict[str, Any],
    semantic_similarity: float,
) -> dict[str, Any]:
    """Score a pair on a 0-100 scale and return an auditable breakdown."""
    employee_skills = parse_list(employee.get("skills", []))
    required_skills = parse_list(project.get("required_skills", []))
    employee_skill_keys = {normalize_skill(skill) for skill in employee_skills}
    required_keys = [normalize_skill(skill) for skill in required_skills]
    exact_keys = {key for key in required_keys if key in employee_skill_keys}
    matched_skills = [skill for skill in required_skills if normalize_skill(skill) in exact_keys]
    missing_skills = [skill for skill in required_skills if normalize_skill(skill) not in exact_keys]
    exact_coverage = len(exact_keys) / len(set(required_keys)) if required_keys else 1.0
    semantic = float(np.clip(semantic_similarity, 0.0, 1.0))
    skill_score = 100.0 * (0.70 * exact_coverage + 0.30 * semantic)

    minimum_experience = max(0.0, float(project.get("minimum_experience", 0) or 0))
    experience = max(0.0, float(employee.get("experience_years", 0) or 0))
    if minimum_experience == 0 or experience >= minimum_experience:
        experience_score = 100.0
    else:
        # Below-minimum candidates get partial credit; meeting the minimum caps the score.
        experience_score = 85.0 * experience / minimum_experience

    available = str(employee.get("availability", "")).strip().casefold() == "available"
    availability_score = 100.0 if available else 0.0
    interests = [str(item).strip().casefold() for item in parse_list(employee.get("career_interests", []))]
    domain = str(project.get("domain", "")).strip().casefold()
    if domain in interests:
        career_score = 100.0
        career_alignment = "Aligned"
    elif any(domain and (domain in item or item in domain) for item in interests):
        career_score = 60.0
        career_alignment = "Partially aligned"
    else:
        career_score = 0.0
        career_alignment = "No direct alignment"

    total = (
        WEIGHTS["skills"] * skill_score
        + WEIGHTS["experience"] * experience_score
        + WEIGHTS["availability"] * availability_score
        + WEIGHTS["career_interest"] * career_score
    )
    explanation = (
        f"Skills {skill_score:.0f}/100 ({len(exact_keys)}/{len(set(required_keys))} required skills "
        f"matched exactly; semantic context {semantic * 100:.0f}%). "
        f"Experience {experience_score:.0f}/100 ({experience:g} years; "
        f"minimum {minimum_experience:g}). Availability {availability_score:.0f}/100. "
        f"Career interest: {career_alignment.lower()}."
    )

    return {
        "score": round(float(np.clip(total, 0.0, 100.0)), 2),
        "skill_score": round(skill_score, 2),
        "exact_skill_coverage": round(exact_coverage * 100.0, 2),
        "semantic_similarity": round(semantic * 100.0, 2),
        "experience_score": round(experience_score, 2),
        "availability_score": round(availability_score, 2),
        "career_score": round(career_score, 2),
        "career_alignment": career_alignment,
        "matching_skills": matched_skills,
        "missing_skills": missing_skills,
        "eligible": available,
        "explanation": explanation,
    }


def rank_employees(
    employees: list[dict[str, Any]],
    project: dict[str, Any],
    project_index: int,
    similarity_index: SimilarityIndex,
    minimum_score: float,
) -> list[dict[str, Any]]:
    ranked = []
    for employee_index, employee in enumerate(employees):
        result = dict(employee)
        result.update(
            score_employee_project(
                employee,
                project,
                similarity_index.semantic_similarity(employee_index, project_index),
            )
        )
        result["meets_threshold"] = result["eligible"] and result["score"] >= minimum_score
        ranked.append(result)
    ranked.sort(
        key=lambda item: (
            not item["eligible"],
            not item["meets_threshold"],
            -item["score"],
            str(item.get("employee_id", "")),
        )
    )
    return ranked
