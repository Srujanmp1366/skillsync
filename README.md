# SkillSync — HackStellar PS-02 MVP

SkillSync recommends employees for project positions and compares a global assignment optimizer with a first-fit greedy baseline. It uses fictional profiles, transparent weighted scoring, and CSV files for local data.

## Requirements

- Python 3.11 or newer
- Internet access on the first run to download `all-MiniLM-L6-v2` (optional; the app falls back to TF-IDF if the model cannot load)

## Run locally

From this directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Choose a project under **AI Recommendations** to inspect ranked profiles. Under **Optimize Allocation**, set a minimum score and run the allocation to compare the global optimizer and first-fit baseline. Previewing a result does not change employee availability or write assignments to the CSV files.

## Matching and allocation

The score ranges from 0 to 100 and combines:

- 45% skill suitability: 70% exact required-skill coverage and 30% semantic similarity
- 25% experience suitability: partial credit below the minimum and full credit once the minimum is met
- 20% availability: available employees receive full credit; assigned employees receive none and are ineligible for assignment
- 10% career-interest alignment

The semantic backend is the pretrained `all-MiniLM-L6-v2` SentenceTransformer, loaded once through Streamlit resource caching. If import, download, or model loading fails, the app identifies and uses a local TF-IDF similarity fallback. Semantic similarity is contextual evidence only; it does not verify possession of a missing skill.

Allocation expands each required project position into a row and solves a global maximum-weight assignment with SciPy's `linear_sum_assignment`. Each row has a zero-score dummy option, and available employees are unique columns. A configurable score threshold excludes weak matches, so a position can remain unfilled. The baseline uses the same scores, availability rule, threshold, and positions, assigning each project in listed order.

## Data

- `data/employees.csv`: 28 fictional employee profiles
- `data/projects.csv`: 8 fictional projects with overlapping requirements

List fields are stored as JSON arrays inside CSV cells. Edit these files to explore different scenarios; employee availability is never modified by the optimizer.

## Core checks

Run the built-in checks from this directory:

```bash
python -m unittest discover -s tests -v
```
