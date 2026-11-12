"""Streamlit interface for the SkillSync resource allocation MVP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from matching import SimilarityIndex, parse_list, rank_employees
from optimizer import first_fit_allocation, optimize_allocation


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    employees = pd.read_csv(DATA_DIR / "employees.csv")
    projects = pd.read_csv(DATA_DIR / "projects.csv")
    for column in ("skills", "career_interests"):
        employees[column] = employees[column].map(parse_list)
    projects["required_skills"] = projects["required_skills"].map(parse_list)
    return employees, projects


@st.cache_resource(show_spinner="Loading the matching model (first run may download it)…")
def load_similarity_index(employee_payload: str, project_payload: str) -> SimilarityIndex:
    return SimilarityIndex(json.loads(employee_payload), json.loads(project_payload))


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def display_list(items: Any) -> str:
    values = parse_list(items)
    return ", ".join(values) if values else "—"


def allocation_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result["assignments"]:
        rows.append({
            "Project": item["project_name"],
            "Position": item["slot_number"],
            "Assigned employee": item["employee_name"],
            "Matching score": item["score"],
            "Matching skills": display_list(item["matching_skills"]),
            "Explanation": item["explanation"],
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="SkillSync | Resource Allocation", page_icon="🧭", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; }
    [data-testid="stMetric"] { background: #f5f7fb; border: 1px solid #e8ebf2;
        padding: 0.85rem 1rem; border-radius: 0.7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SkillSync")
st.caption("Employee and project matching for informed resource planning")

employees_frame, projects_frame = load_data()
employees = records(employees_frame)
projects = records(projects_frame)
employee_payload = json.dumps(employees, sort_keys=True)
project_payload = json.dumps(projects, sort_keys=True)
similarity_index = load_similarity_index(employee_payload, project_payload)

with st.sidebar:
    st.header("Matching settings")
    minimum_score = st.slider(
        "Minimum score for allocation",
        min_value=0,
        max_value=100,
        value=65,
        help="Available employees below this score are left unassigned by both allocation methods.",
    )
    st.caption("Score weights: skills 45% · experience 25% · availability 20% · career interests 10%.")
    st.divider()
    st.subheader("Semantic matching")
    st.write(f"Backend: **{similarity_index.backend_name}**")
    if similarity_index.model_error:
        st.warning("The pretrained model could not be loaded. Similarity uses a local TF-IDF fallback.")
        with st.expander("Model loading detail"):
            st.code(similarity_index.model_error)
    st.caption("Semantic similarity is supporting context; it does not verify possession of a skill.")

tabs = st.tabs(["Dashboard", "Employees", "AI Recommendations", "Optimize Allocation"])

with tabs[0]:
    total_positions = int(projects_frame["required_employees"].sum())
    available_count = int((employees_frame["availability"].str.casefold() == "available").sum())
    metric_columns = st.columns(5)
    metric_columns[0].metric("Employees", len(employees_frame))
    metric_columns[1].metric("Available", available_count)
    metric_columns[2].metric("Projects", len(projects_frame))
    metric_columns[3].metric("Required positions", total_positions)
    metric_columns[4].metric("Allocation threshold", f"{minimum_score}/100")

    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Project requirements")
        project_view = projects_frame.copy()
        project_view["required_skills"] = project_view["required_skills"].map(display_list)
        project_view = project_view.rename(columns={
            "project_name": "Project",
            "domain": "Domain",
            "required_skills": "Required skills",
            "minimum_experience": "Minimum experience (years)",
            "required_employees": "Positions",
        })
        st.dataframe(
            project_view[["Project", "Domain", "Required skills", "Minimum experience (years)", "Positions"]],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.subheader("Employee availability")
        availability = employees_frame["availability"].value_counts().rename_axis("Status").to_frame("Employees")
        st.bar_chart(availability, color="#4766d5")
        st.caption("Assigned employees are excluded from full-time recommendations and allocation.")

with tabs[1]:
    st.subheader("Employee directory")
    search = st.text_input("Search employees", placeholder="Name, skill, interest, or employee ID")
    employee_view = employees_frame.copy()
    employee_view["skills"] = employee_view["skills"].map(display_list)
    employee_view["career_interests"] = employee_view["career_interests"].map(display_list)
    employee_view = employee_view.rename(columns={
        "employee_id": "Employee ID",
        "name": "Name",
        "skills": "Skills",
        "experience_years": "Experience (years)",
        "availability": "Availability",
        "career_interests": "Career interests",
    })
    if search.strip():
        query = search.strip().casefold()
        mask = employee_view.astype(str).apply(lambda column: column.str.casefold().str.contains(query, regex=False)).any(axis=1)
        employee_view = employee_view[mask]
    st.dataframe(employee_view, hide_index=True, use_container_width=True)
    st.caption(f"Showing {len(employee_view)} of {len(employees_frame)} fictional employee profiles.")

with tabs[2]:
    st.subheader("Project recommendations")
    project_names = [project["project_name"] for project in projects]
    selected_name = st.selectbox("Select a project", project_names, key="recommendation_project")
    selected_project_index = project_names.index(selected_name)
    selected_project = projects[selected_project_index]
    ranked = rank_employees(
        employees, selected_project, selected_project_index, similarity_index, minimum_score
    )
    st.caption(
        f"Required skills: {display_list(selected_project['required_skills'])} · "
        f"Minimum experience: {selected_project['minimum_experience']} years · "
        f"Positions: {selected_project['required_employees']}"
    )

    eligible = [item for item in ranked if item["eligible"]]
    ineligible = [item for item in ranked if not item["eligible"]]

    def recommendation_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame([{
            "Employee": item["name"],
            "Score": item["score"],
            "Suitability": "Meets threshold" if item["meets_threshold"] else "Below threshold",
            "Matching skills": display_list(item["matching_skills"]),
            "Missing skills": display_list(item["missing_skills"]),
            "Experience": f"{item['experience_years']} years",
            "Availability": item["availability"],
            "Career alignment": item["career_alignment"],
            "Skill fit": item["skill_score"],
            "Experience fit": item["experience_score"],
            "Availability fit": item["availability_score"],
            "Career fit": item["career_score"],
            "Explanation": item["explanation"],
        } for item in items])

    st.markdown("#### Available employees")
    st.caption("Eligible employees can be considered. Only those meeting the selected threshold are used by allocation.")
    st.dataframe(recommendation_frame(eligible), hide_index=True, use_container_width=True)
    with st.expander(f"Ineligible employees ({len(ineligible)} assigned)"):
        st.caption("Assigned employees are shown for context but cannot receive a new full-time assignment.")
        st.dataframe(recommendation_frame(ineligible), hide_index=True, use_container_width=True)
    st.info("The score combines exact skill coverage with semantic context. A semantic match does not confirm a candidate has a missing required skill.")

with tabs[3]:
    st.subheader("Organization-wide allocation")
    st.write(
        "Run both methods against the same project positions, currently available employees, score formula, "
        "and minimum suitability threshold. Assignments here are a planning result and do not change employee records."
    )
    if st.button("Run allocation", type="primary", key="run_allocation"):
        st.session_state["allocation_results"] = {
            "threshold": minimum_score,
            "optimized": optimize_allocation(employees, projects, similarity_index, minimum_score),
            "baseline": first_fit_allocation(employees, projects, similarity_index, minimum_score),
        }

    results = st.session_state.get("allocation_results")
    if results is None:
        st.info("Run the allocation to compare the global optimizer with the first-fit baseline.")
    else:
        if results["threshold"] != minimum_score:
            st.warning(
                f"These results use a {results['threshold']}/100 threshold. "
                "Run allocation again to apply the current threshold."
            )
        optimized = results["optimized"]
        baseline = results["baseline"]
        summary = st.columns(4)
        summary[0].metric("Allocated employees", optimized["filled_positions"])
        summary[1].metric("Unfilled positions", len(optimized["unfilled"]))
        summary[2].metric("Total matching score", f"{optimized['total_score']:.2f}")
        summary[3].metric("Average score per assignment", f"{optimized['average_score']:.2f}")

        st.markdown("#### Global optimized assignments")
        st.dataframe(allocation_table(optimized), hide_index=True, use_container_width=True)
        st.markdown("#### Unfilled positions")
        if optimized["unfilled"]:
            unfilled_frame = pd.DataFrame([{
                "Project": item["project_name"],
                "Position": item["slot_number"],
                "Required skills": display_list(item["required_skills"]),
                "Minimum experience (years)": item["minimum_experience"],
            } for item in optimized["unfilled"]])
            st.dataframe(unfilled_frame, hide_index=True, use_container_width=True)
        else:
            st.success("All project positions met the current threshold.")

        st.markdown("#### First-fit baseline comparison")
        comparison = pd.DataFrame([
            {
                "Method": "First-fit greedy baseline",
                "Filled positions": baseline["filled_positions"],
                "Unfilled positions": len(baseline["unfilled"]),
                "Total matching score": baseline["total_score"],
                "Average assignment score": baseline["average_score"],
            },
            {
                "Method": "Global optimized allocation",
                "Filled positions": optimized["filled_positions"],
                "Unfilled positions": len(optimized["unfilled"]),
                "Total matching score": optimized["total_score"],
                "Average assignment score": optimized["average_score"],
            },
        ])
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        st.caption("Scores shown are the actual sums and averages from these two methods on the current dataset.")

st.divider()
st.caption("SkillSync is decision support for HR and project managers. Human review should confirm skills and assignment fit.")
