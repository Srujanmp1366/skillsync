"""Global project allocation and a comparable first-fit greedy baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from matching import SimilarityIndex, parse_list, score_employee_project


def _project_slots(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slots = []
    for project in projects:
        count = max(0, int(project.get("required_employees", 0) or 0))
        for number in range(1, count + 1):
            slots.append({"project": project, "slot_number": number})
    return slots


def _assignment_row(
    employee: dict[str, Any],
    project: dict[str, Any],
    slot_number: int,
    employee_index: int,
    project_index: int,
    similarity_index: SimilarityIndex,
) -> dict[str, Any]:
    score = score_employee_project(
        employee,
        project,
        similarity_index.semantic_similarity(employee_index, project_index),
    )
    return {
        "project_id": project["project_id"],
        "project_name": project["project_name"],
        "slot_number": slot_number,
        "employee_id": employee["employee_id"],
        "employee_name": employee["name"],
        "score": score["score"],
        "matching_skills": score["matching_skills"],
        "missing_skills": score["missing_skills"],
        "explanation": score["explanation"],
    }


def optimize_allocation(
    employees: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    similarity_index: SimilarityIndex,
    minimum_score: float = 65.0,
) -> dict[str, Any]:
    """Maximize the sum of suitable matches using a global assignment solve.

    Each project position is a row. Real employees are unique columns and each
    position has its own zero-value dummy column, allowing it to stay unfilled.
    """
    slots = _project_slots(projects)
    available_indices = [
        index for index, employee in enumerate(employees)
        if str(employee.get("availability", "")).strip().casefold() == "available"
    ]
    total_slots = len(slots)
    assignments: list[dict[str, Any]] = []
    filled_slots: set[int] = set()

    if total_slots:
        forbidden = -1_000_000.0
        weights = np.full((total_slots, len(available_indices) + total_slots), forbidden, dtype=float)
        for slot_index, slot in enumerate(slots):
            project = slot["project"]
            project_index = next(i for i, item in enumerate(projects) if item["project_id"] == project["project_id"])
            for column, employee_index in enumerate(available_indices):
                score = score_employee_project(
                    employees[employee_index],
                    project,
                    similarity_index.semantic_similarity(employee_index, project_index),
                )["score"]
                if score >= minimum_score:
                    weights[slot_index, column] = score
            weights[slot_index, len(available_indices) + slot_index] = 0.0

        row_indices, column_indices = linear_sum_assignment(-weights)
        for slot_index, column in zip(row_indices.tolist(), column_indices.tolist()):
            if column >= len(available_indices) or weights[slot_index, column] <= 0:
                continue
            employee_index = available_indices[column]
            project = slots[slot_index]["project"]
            project_index = next(i for i, item in enumerate(projects) if item["project_id"] == project["project_id"])
            assignments.append(
                _assignment_row(
                    employees[employee_index], project, slots[slot_index]["slot_number"],
                    employee_index, project_index, similarity_index,
                )
            )
            filled_slots.add(slot_index)

    unfilled = []
    for slot_index, slot in enumerate(slots):
        if slot_index not in filled_slots:
            project = slot["project"]
            unfilled.append({
                "project_id": project["project_id"],
                "project_name": project["project_name"],
                "slot_number": slot["slot_number"],
                "required_skills": parse_list(project.get("required_skills", [])),
                "minimum_experience": project.get("minimum_experience", 0),
            })
    assignments.sort(key=lambda row: (str(row["project_id"]), row["slot_number"]))
    total_score = round(sum(row["score"] for row in assignments), 2)
    return {
        "assignments": assignments,
        "unfilled": unfilled,
        "total_positions": total_slots,
        "filled_positions": len(assignments),
        "total_score": total_score,
        "average_score": round(total_score / len(assignments), 2) if assignments else 0.0,
        "minimum_score": float(minimum_score),
    }


def first_fit_allocation(
    employees: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    similarity_index: SimilarityIndex,
    minimum_score: float = 65.0,
) -> dict[str, Any]:
    """First-fit project-by-project baseline under the optimizer's same rules."""
    slots = _project_slots(projects)
    available_indices = {
        index for index, employee in enumerate(employees)
        if str(employee.get("availability", "")).strip().casefold() == "available"
    }
    assignments = []
    unfilled = []
    for slot in slots:
        project = slot["project"]
        project_index = next(i for i, item in enumerate(projects) if item["project_id"] == project["project_id"])
        candidates = []
        for employee_index in available_indices:
            score = score_employee_project(
                employees[employee_index], project,
                similarity_index.semantic_similarity(employee_index, project_index),
            )["score"]
            if score >= minimum_score:
                candidates.append((score, str(employees[employee_index]["employee_id"]), employee_index))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if not candidates:
            unfilled.append({
                "project_id": project["project_id"],
                "project_name": project["project_name"],
                "slot_number": slot["slot_number"],
                "required_skills": parse_list(project.get("required_skills", [])),
                "minimum_experience": project.get("minimum_experience", 0),
            })
            continue
        _, _, employee_index = candidates[0]
        available_indices.remove(employee_index)
        assignments.append(
            _assignment_row(
                employees[employee_index], project, slot["slot_number"], employee_index,
                project_index, similarity_index,
            )
        )
    total_score = round(sum(row["score"] for row in assignments), 2)
    return {
        "assignments": assignments,
        "unfilled": unfilled,
        "total_positions": len(slots),
        "filled_positions": len(assignments),
        "total_score": total_score,
        "average_score": round(total_score / len(assignments), 2) if assignments else 0.0,
        "minimum_score": float(minimum_score),
    }
