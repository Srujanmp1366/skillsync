import unittest

import numpy as np

from matching import score_employee_project
from optimizer import first_fit_allocation, optimize_allocation


class StaticSimilarity:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def semantic_similarity(self, employee_index, project_index):
        return float(self.values[employee_index, project_index])


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.project = {
            "project_id": "P1",
            "project_name": "Analytics API",
            "required_skills": ["Python", "SQL"],
            "minimum_experience": 4,
            "domain": "Analytics",
        }

    def test_available_employee_with_exact_skills(self):
        employee = {
            "employee_id": "E1", "name": "A", "skills": ["Python", "SQL"],
            "experience_years": 5, "availability": "Available", "career_interests": ["Analytics"],
        }
        result = score_employee_project(employee, self.project, 0.8)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["matching_skills"], ["Python", "SQL"])
        self.assertEqual(result["score"], 97.3)

    def test_below_minimum_experience_gets_partial_credit(self):
        employee = {
            "employee_id": "E1", "name": "A", "skills": ["Python", "SQL"],
            "experience_years": 2, "availability": "Available", "career_interests": ["Analytics"],
        }
        result = score_employee_project(employee, self.project, 0.8)
        self.assertLess(result["experience_score"], 100)
        self.assertTrue(result["eligible"])

    def test_assigned_employee_is_ineligible(self):
        employee = {
            "employee_id": "E1", "name": "A", "skills": ["Python", "SQL"],
            "experience_years": 8, "availability": "Assigned", "career_interests": ["Analytics"],
        }
        self.assertFalse(score_employee_project(employee, self.project, 0.8)["eligible"])

    def test_semantic_context_does_not_claim_missing_exact_skill(self):
        employee = {
            "employee_id": "E1", "name": "A", "skills": ["Python", "PostgreSQL"],
            "experience_years": 5, "availability": "Available", "career_interests": [],
        }
        result = score_employee_project(employee, self.project, 0.95)
        self.assertEqual(result["matching_skills"], ["Python"])
        self.assertEqual(result["missing_skills"], ["SQL"])
        self.assertIn("semantic context 95%", result["explanation"])

    def test_global_allocation_is_unique_and_leaves_unsupported_slots_open(self):
        employees = [
            {"employee_id": "E1", "name": "A", "skills": ["Python"], "experience_years": 5,
             "availability": "Available", "career_interests": ["Analytics"]},
            {"employee_id": "E2", "name": "B", "skills": ["Python"], "experience_years": 5,
             "availability": "Assigned", "career_interests": ["Analytics"]},
        ]
        projects = [
            {**self.project, "required_employees": 2},
            {"project_id": "P2", "project_name": "Mainframe", "required_skills": ["COBOL", "IBM MQ"],
             "minimum_experience": 8, "required_employees": 1, "domain": "Legacy"},
        ]
        similarity = StaticSimilarity([[0.8, 0.1], [0.8, 0.1]])
        result = optimize_allocation(employees, projects, similarity, minimum_score=60)
        ids = [item["employee_id"] for item in result["assignments"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("E2", ids)
        self.assertEqual(result["filled_positions"], 1)
        self.assertEqual(len(result["unfilled"]), 2)

    def test_multiple_positions_competing_for_employee_and_baseline_validity(self):
        employees = [
            {"employee_id": "E1", "name": "A", "skills": ["Python", "SQL"], "experience_years": 5,
             "availability": "Available", "career_interests": ["Analytics"]},
            {"employee_id": "E2", "name": "B", "skills": ["Python", "SQL"], "experience_years": 5,
             "availability": "Available", "career_interests": ["Analytics"]},
        ]
        projects = [{**self.project, "required_employees": 2},
                    {"project_id": "P2", "project_name": "Analytics API 2", "required_skills": ["Python"],
                     "minimum_experience": 3, "required_employees": 1, "domain": "Analytics"}]
        similarity = StaticSimilarity([[0.8, 0.8], [0.7, 0.7]])
        for result in (
            optimize_allocation(employees, projects, similarity, minimum_score=60),
            first_fit_allocation(employees, projects, similarity, minimum_score=60),
        ):
            ids = [item["employee_id"] for item in result["assignments"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertLessEqual(result["filled_positions"], len(employees))
            self.assertEqual(result["filled_positions"] + len(result["unfilled"]), result["total_positions"])


if __name__ == "__main__":
    unittest.main()
