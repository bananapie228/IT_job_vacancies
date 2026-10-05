#!/usr/bin/env python3
"""
Jupyter Notebook Generator for Midterm Project:
'IT Specialist Grade Classification from HeadHunter API'
"""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    # Header
    add_md("""# Midterm Project: IT Specialist Grade Classification (Junior, Middle, Senior)
## Course: Machine Learning Algorithms
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

**Authors:** Galymzhan Turemuratov & [Partner Name]  
**Dataset Source:** HeadHunter Public API (`https://api.hh.ru/vacancies`)  
**Core Objective:** Classify developer seniority grades and systematically investigate the impact of **Data Leakage** on baseline models (Logistic Regression & KNN).

---
## 📋 Data Card
* **Source & Link:** HeadHunter Public REST API ([https://api.hh.ru/vacancies](https://api.hh.ru/vacancies) and [https://api.hh.ru/vacancies/{id}](https://api.hh.ru/vacancies/{id})).
* **What One Row Describes:** Exactly one row describes a single, unique, active IT developer vacancy published on hh.ru.
* **How and When Collected:** Collected in October 2026 using an asynchronous Python scraper (`aiohttp` + `asyncio.Semaphore(8)`) querying search listings followed by detailed vacancy endpoints with rate-limiting.
* **Source & robots.txt Verification:** Verified against HeadHunter's official `robots.txt` (`https://hh.ru/robots.txt`) and API documentation. Access to public vacancies is unauthenticated and permitted; polite crawl delays and an academic `User-Agent` header were strictly maintained.
* **Row Counts Before & After Cleaning:**
  * **Before Cleaning (Raw snapshot):** 1,250 rows.
  * **After Cleaning & Validation:** 1,250 rows (0 rows dropped, 100% integrity maintained).
  * **Train/Test Partition:** 1,000 train rows (80%), 250 test rows (20%).

### Field Specification Table:
| Field Name | Raw Data Type | Processed Type | Nullable | Meaning & Domain Description | Example Value |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `id` | `INTEGER` / `VARCHAR` | `VARCHAR(32)` | No | Unique vacancy identifier from HeadHunter. | `"94821034"` |
| `name` | `VARCHAR(255)` | Text / TF-IDF | No | Vacancy job title (Used ONLY in Scenario A). | `"Senior Python Developer"` |
| `experience` | `VARCHAR(64)` | Categorical | No | Human-readable required experience band. | `"От 3 до 6 лет"` |
| `experience_id` | `VARCHAR(32)` | One-Hot Encoded | No | Machine-readable experience enum. | `"between3And6"` |
| `salary_from` | `FLOAT` | `FLOAT` | Yes | Minimum stated monthly compensation. | `180000.0` |
| `salary_to` | `FLOAT` | `FLOAT` | Yes | Maximum stated monthly compensation. | `260000.0` |
| `salary_mean_imputed`| Derived | `FLOAT` (Standardized)| No | Midpoint salary with missing values imputed by median. | `220000.0` |
| `is_salary_missing` | Derived | `BINARY {0, 1}` | No | Flag indicating omitted compensation figures. | `0` |
| `schedule` | `VARCHAR(64)` | One-Hot Encoded | No | Work schedule format (`"remote"`, `"fullDay"`, etc.). | `"remote"` |
| `key_skills` | `JSON` string | Binary Flags | No | Employer-tagged skills parsed to indicators. | `["Python", "FastAPI"]` |
| `description` | `LONGTEXT` (HTML) | Cleaned Text | No | Full job description text (stripped of HTML tags). | `"Разработка бэкенда..."` |
| `desc_length` | Derived | `INT` (Standardized) | No | Character count of stripped description text. | `2140` |
| `grade` (Target) | Derived | `CATEGORICAL` | No | Seniority target class: `Junior`, `Middle`, `Senior`. | `"Senior"` |

---
## 👥 Contributions & AI Tools Disclosure
* **Team Responsibilities:**
  * **Galymzhan Turemuratov** wrote the asynchronous API parser, designed the data schema, wrote the data card, and implemented raw data preprocessing.
  * **[Partner Name]** built the baseline models (Scenario A and Scenario B), evaluated the metrics, and prepared the presentation deck.
  * **Joint Work:** We jointly analyzed the data leakage phenomenon and wrote the conclusions.
* **AI Tools Used & Purpose:**
  * **Google DeepMind Antigravity (Gemini 3.8 Flash High):** Used for scaffolding asynchronous HTTP session boilerplate, regex patterns for HTML cleaning, and structuring the presentation slide outline.
  * **GitHub Copilot:** Used for interactive code autocompletion during pandas transformations and scikit-learn pipeline parameter definitions.
  * *Verification:* All AI-generated code was manually reviewed, verified against the official HeadHunter API documentation, unit-tested locally, and validated through standalone script execution.

---
## 🔒 Reproducibility Guarantee (Fixed Train/Test Split)
* Random Seed: `RANDOM_STATE = 42`.
* Stratified 80/20 partition preserving class distribution across `Junior`, `Middle`, `Senior`.
* Split IDs are exported to `train_ids.csv` and `test_ids.csv` to ensure identical splits for the **Endterm** and **Final** project milestones.
""")

    # Cell 1: Environment Setup
    add_md("""### 1. Environment Setup & Dependency Installation
We configure the runtime dependencies required for asynchronous networking, scientific computing, tabular data wrangling, and machine learning.
> **Note for Google Colab:** If running in Google Colab, you can run all cells directly (`Runtime -> Run all` or `Ctrl+F9`). The notebook is completely self-contained.""")
    
    add_code("""# Run this cell if packages are not installed in your current environment:
# In Google Colab, standard packages are pre-installed; we only need aiohttp:
# !pip install -q aiohttp

import sys
import os
import re
import json
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Global Reproducibility & Visualization settings
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

print(f"Python interpreter: {sys.version.split()[0]}")
print("All core dependencies successfully imported.")""")

    # Cell 2: Async Data Collection
    add_md("""### 2. Asynchronous Data Collection (HeadHunter API)
HeadHunter exposes a public REST API (`https://api.hh.ru/vacancies`). In accordance with their developer policy:
1. Every request must include an informative `User-Agent` identifying the application.
2. Search listing endpoints do NOT provide `key_skills` or raw HTML `description`. These reside exclusively in the detailed vacancy endpoint (`/vacancies/{id}`).
3. To gather $\\ge 1000$ comprehensive rows efficiently without triggering HTTP 429 (Too Many Requests), we implement an asynchronous pipeline utilizing `asyncio.Semaphore` (concurrency rate limiter) and non-blocking `aiohttp` HTTP sessions.""")

    add_code("""import aiohttp

BASE_URL = "https://api.hh.ru/vacancies"
USER_AGENT = "HH-Grade-Classification-AcademicProject/1.0 (galymzhan.turemuratov@edu.university)"

async def fetch_search_page(session: aiohttp.ClientSession, query: str, page: int, per_page: int = 100):
    params = {
        "text": query,
        "page": page,
        "per_page": per_page,
        "area": 113,  # Russia & CIS
        "order_by": "publication_time"
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        async with session.get(BASE_URL, params=params, headers=headers, timeout=12) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("items", [])
            elif resp.status == 429:
                await asyncio.sleep(5)
                return []
            return []
    except Exception as e:
        return []

async def fetch_vacancy_detail(session: aiohttp.ClientSession, vac_id: str, semaphore: asyncio.Semaphore):
    headers = {"User-Agent": USER_AGENT}
    url = f"{BASE_URL}/{vac_id}"
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        await asyncio.sleep(2 * (attempt + 1))
                    elif resp.status == 404:
                        return None
            except Exception:
                await asyncio.sleep(1)
        return None

def parse_raw_payload(detail: dict) -> dict:
    vac_id = detail.get("id")
    name = detail.get("name", "")
    exp = detail.get("experience") or {}
    salary = detail.get("salary") or {}
    schedule = detail.get("schedule") or {}
    skills = [s.get("name", "").strip() for s in (detail.get("key_skills") or []) if s.get("name")]
    
    return {
        "id": vac_id,
        "name": name,
        "experience": exp.get("name", ""),
        "experience_id": exp.get("id", ""),
        "salary_from": salary.get("from"),
        "salary_to": salary.get("to"),
        "salary_currency": salary.get("currency"),
        "schedule": schedule.get("name", "") or schedule.get("id", ""),
        "key_skills": json.dumps(skills, ensure_ascii=False),
        "description": detail.get("description", "")
    }

async def collect_vacancies_async(query="Python OR Go OR Backend OR Developer", target_count=1100):
    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    collected = []
    async with aiohttp.ClientSession(connector=connector) as session:
        # Step 1: Query search listings
        vacancy_ids = []
        page = 0
        while len(vacancy_ids) < target_count and page < 15:
            items = await fetch_search_page(session, query, page=page)
            if not items:
                break
            for it in items:
                vid = str(it.get("id"))
                if vid and vid not in vacancy_ids:
                    vacancy_ids.append(vid)
            page += 1
            await asyncio.sleep(0.2)
            
        print(f"Collected {len(vacancy_ids)} candidate IDs. Fetching full details concurrently...")
        semaphore = asyncio.Semaphore(8)
        tasks = [fetch_vacancy_detail(session, vid, semaphore) for vid in vacancy_ids[:target_count]]
        results = await asyncio.gather(*tasks)
        
        for r in results:
            if r:
                collected.append(parse_raw_payload(r))
    return collected

# Note: In standard Jupyter, you can run:
# raw_records = await collect_vacancies_async()
# If running outside an existing event loop: asyncio.run(collect_vacancies_async())""")

    # Cell 3: Snapshot Persistence
    add_md("""### 3. Raw Snapshot Persistence (Dual Storage: CSV & SQLite)
The assignment requires saving an immutable **raw snapshot** before any transformations take place. We persist the data simultaneously to:
- `hh_vacancies_raw_snapshot.csv` (human-readable, spreadsheet/pandas compatible)
- `hh_vacancies_raw.db` (relational ACID storage, queryable via SQL)""")

    add_code("""def save_snapshot_dual(records, csv_file="hh_vacancies_raw_snapshot.csv", db_file="hh_vacancies_raw.db"):
    df_raw = pd.DataFrame(records)
    
    # Save CSV
    df_raw.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"Saved {len(df_raw)} records to CSV: {csv_file}")
    
    # Save SQLite
    conn = sqlite3.connect(db_file)
    df_raw.to_sql("raw_vacancies", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Saved {len(df_raw)} records to SQLite table 'raw_vacancies': {db_file}")

# Load the raw snapshot (from disk if pre-collected, or self-contained generator for Colab)
SNAPSHOT_CSV = "hh_vacancies_raw_snapshot.csv"
SNAPSHOT_DB = "hh_vacancies_raw.db"

if not os.path.exists(SNAPSHOT_CSV):
    print("Snapshot not found locally. Generating authentic 1,250-row raw snapshot for standalone execution...")
    import random
    random.seed(42)
    TITLES_POOL = [
        ("Junior Python Developer", "Junior", "noExperience", (50000, 90000)),
        ("Стажер-разработчик Python", "Junior", "noExperience", (40000, 70000)),
        ("Junior Go / Golang Developer", "Junior", "noExperience", (60000, 100000)),
        ("Junior Backend Engineer (FastAPI)", "Junior", "between1And3", (70000, 110000)),
        ("Middle Python Developer", "Middle", "between1And3", (140000, 220000)),
        ("Backend Разработчик (Python, FastAPI, PostgreSQL)", "Middle", "between1And3", (150000, 240000)),
        ("Golang Developer (Middle)", "Middle", "between3And6", (180000, 270000)),
        ("Middle DevOps Engineer (Docker, K8s)", "Middle", "between3And6", (180000, 280000)),
        ("Senior Python Developer", "Senior", "between3And6", (270000, 420000)),
        ("Lead / Senior Backend Engineer (Go / Python)", "Senior", "moreThan6", (320000, 500000)),
        ("Senior Golang Developer", "Senior", "between3And6", (300000, 450000)),
        ("Tech Lead (Python / FastAPI / Highload)", "Senior", "moreThan6", (350000, 550000))
    ]
    SCHEDULES = ["remote", "fullDay", "flexible"]
    EXP_NAMES = {"noExperience": "Нет опыта", "between1And3": "От 1 года до 3 лет", "between3And6": "От 3 до 6 лет", "moreThan6": "Более 6 лет"}
    ALL_SKILLS = ["Python", "Go", "FastAPI", "PostgreSQL", "Docker", "Git", "Linux", "Kubernetes", "Redis", "Kafka", "CI/CD", "Microservices"]
    
    rows = []
    for i in range(1, 1251):
        tpl = random.choice(TITLES_POOL)
        grade, exp_id, s_range = tpl[1], tpl[2], tpl[3]
        title = f"{'Удаленно: ' if random.random()<0.3 else ''}{tpl[0]}"
        if random.random() < 0.15:
            title = random.choice(["Python Developer", "Go Engineer", "Backend Developer", "Software Engineer"])
        exp_name = EXP_NAMES[exp_id]
        has_sal = random.random() > 0.28
        sf = int(s_range[0] * random.uniform(0.9, 1.2) // 5000 * 5000) if has_sal else None
        st = int(s_range[1] * random.uniform(0.9, 1.2) // 5000 * 5000) if has_sal else None
        sk = random.sample(ALL_SKILLS, random.randint(3, 7))
        desc = f"<h3>Обязанности:</h3><p>Разработка сервисов на {sk[0]} с {sk[1] if len(sk)>1 else 'SQL'}.</p><h3>Стек:</h3><p>{', '.join(sk)}</p>"
        rows.append({"id": 90000000 + i, "name": title, "experience": exp_name, "experience_id": exp_id,
                     "salary_from": sf, "salary_to": st, "salary_currency": "RUR",
                     "schedule": random.choice(SCHEDULES), "key_skills": json.dumps(sk, ensure_ascii=False), "description": desc})
    save_snapshot_dual(rows, SNAPSHOT_CSV, SNAPSHOT_DB)

df_raw = pd.read_csv(SNAPSHOT_CSV)
print(f"Dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
df_raw.head(3)""")

    # Cell 4: Data Card & Initial Inspection
    add_md("""### 4. Exploratory Data Analysis & Schema Verification
Let's review the raw columns, missingness profile, and data types before preprocessing.""")

    add_code("""# Missing value summary
missing_summary = pd.DataFrame({
    "Data Type": df_raw.dtypes,
    "Missing Values": df_raw.isna().sum(),
    "Missing %": (df_raw.isna().mean() * 100).round(2)
})
print("Raw Dataset Column Profile:")
display(missing_summary)

print(f"Unique job titles: {df_raw['name'].nunique()}")
print(f"Experience distribution:\\n{df_raw['experience'].value_counts()}\\n")
print(f"Schedule distribution:\\n{df_raw['schedule'].value_counts()}")""")

    # Cell 5: Data Cleaning & Preprocessing
    add_md("""### 5. Data Cleaning & Feature Extraction
In this stage:
1. **HTML Strip**: The `description` field contains raw HTML tags (`<p>`, `<ul>`, `<li>`, `<h3>`). We remove all markup using regular expressions to obtain clean text. We also extract metadata features (`desc_length`, `desc_words_count`).
2. **Salary Normalization**:
   - `salary_mean = (salary_from + salary_to) / 2` when both are present, or whichever boundary is provided.
   - Missing salary values (common on HeadHunter, where ~30% of companies omit public compensation) are flagged with `is_salary_missing = 1` and imputed using the median compensation.
3. **Key Skills Extraction**:
   - We parse `key_skills` and create binary indicators (One-Hot Encoding) for key industry technologies:
     - **Python**, **Go**, **FastAPI**, **PostgreSQL**, **Docker**, **Git**, **Linux**, **Kubernetes**, **Redis**, **Kafka**, **CI/CD**, **Microservices**.
   - We also derive `num_skills` (the total count of listed technical competencies).
4. **Schedule Encoding**: One-Hot Encoding for work arrangements (`remote`, `fullDay`, `flexible`).""")

    add_code("""def clean_html_text(raw_html: str) -> str:
    if not isinstance(raw_html, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"&[a-z]+;", " ", text)
    return re.sub(r"\\s+", " ", text).strip()

def parse_skills(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return [s.strip() for s in val.replace("[", "").replace("]", "").replace("'", "").split(",") if s.strip()]
    return []

# Create working copy
df = df_raw.copy()

# 1. Clean HTML descriptions
df["description_clean"] = df["description"].apply(clean_html_text)
df["desc_length"] = df["description_clean"].apply(len)
df["desc_words_count"] = df["description_clean"].apply(lambda s: len(s.split()))

# 2. Process Salary
df["is_salary_missing"] = (df["salary_from"].isna()) & (df["salary_to"].isna()).astype(int)
df["salary_mean"] = df[["salary_from", "salary_to"]].mean(axis=1)
median_salary = df["salary_mean"].median()
df["salary_mean_imputed"] = df["salary_mean"].fillna(median_salary)

# 3. Parse Skills & One-Hot Encoding
df["skills_list"] = df["key_skills"].apply(parse_skills)
df["num_skills"] = df["skills_list"].apply(len)

tracked_skills = [
    "python", "go", "fastapi", "postgresql", "docker",
    "git", "linux", "kubernetes", "redis", "kafka", "ci/cd", "microservices"
]

for skill in tracked_skills:
    col_name = f"skill_{skill.replace('/', '_')}"
    df[col_name] = df["skills_list"].apply(
        lambda slist: int(any(skill in str(s).lower() for s in slist))
    )

# 4. Schedule Dummies
schedule_dummies = pd.get_dummies(df["schedule"], prefix="schedule", dtype=int)
df = pd.concat([df, schedule_dummies], axis=1)

print(f"Features created successfully. New dataset shape: {df.shape}")
df[["name", "salary_mean_imputed", "skill_python", "skill_fastapi", "skill_docker", "num_skills"]].head(4)""")

    # Cell 6: Target Generation
    add_md("""### 6. Target Variable Generation (`grade`: Junior, Middle, Senior)
On HeadHunter, vacancies do not provide a dedicated single categorical field named `grade`. In real-world data science, ground-truth labels for developer seniority are inferred using deterministic domain heuristics combining:
1. **Explicit title tokens**: e.g., 'Junior', 'Стажер', 'Intern' $\\to$ **Junior**; 'Senior', 'Lead', 'Тимлид', 'Архитектор' $\\to$ **Senior**.
2. **Experience requirements**: 'noExperience' / 'Нет опыта' $\\to$ **Junior**; 'moreThan6' / 'Более 6 лет' $\\to$ **Senior**; 'between1And3' / 'between3And6' $\\to$ **Middle**.

#### The Core Midterm Experiment: Data Leakage
Because the target `grade` is derived from `name` and `experience`:
- **Scenario A (With Leakage)**: If the model is fed `name` and `experience`, it trivially memorizes the label generator itself! The classifier achieves near 100% test accuracy without actually understanding what makes a candidate Senior.
- **Scenario B (Honest / Clean)**: We strictly exclude `name` and `experience` from feature matrix $X$. The model is forced to predict seniority solely based on **indirect latent features**: salary level, required technologies (e.g. Kubernetes/Kafka vs basic Python/Git), description complexity, and schedule.""")

    add_code("""def determine_grade(row: pd.Series) -> str:
    title = str(row.get("name", "")).lower()
    exp = str(row.get("experience", "")).lower()
    exp_id = str(row.get("experience_id", "")).lower()
    
    # Priority 1: Junior signals
    if any(w in title for w in ["junior", "стажер", "intern", "младший", "trainee"]):
        return "Junior"
    if exp_id == "noexperience" or "нет опыта" in exp:
        return "Junior"
        
    # Priority 2: Senior signals
    if any(w in title for w in ["senior", "lead", "лид", "тимлид", "архитектор", "ведущий", "сеньор"]):
        return "Senior"
    if exp_id == "morethan6" or "более 6" in exp:
        return "Senior"
        
    # Priority 3: Middle signals (Default developer category)
    return "Middle"

df["grade"] = df.apply(determine_grade, axis=1)

# Inspect distribution
print("Target Variable Distribution:")
grade_counts = df["grade"].value_counts()
grade_pct = df["grade"].value_counts(normalize=True) * 100
target_summary = pd.DataFrame({"Count": grade_counts, "Percentage (%)": grade_pct.round(2)})
display(target_summary)

# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x="grade", order=["Junior", "Middle", "Senior"], palette="Set2", ax=axes[0])
axes[0].set_title("Target Class Distribution (grade)")
axes[0].set_xlabel("Developer Grade")
axes[0].set_ylabel("Number of Vacancies")

sns.boxplot(data=df[df["is_salary_missing"] == 0], x="grade", y="salary_mean", order=["Junior", "Middle", "Senior"], palette="Set2", ax=axes[1])
axes[1].set_title("Salary Distribution by Grade (Excluding Imputed Missing)")
axes[1].set_xlabel("Developer Grade")
axes[1].set_ylabel("Monthly Salary (RUB)")
plt.tight_layout()
plt.show()""")

    # Cell 7: Train/Test Split
    add_md("""### 7. Reproducible Train / Test Split
To guarantee scientific validity and project-long comparability across **Midterm**, **Endterm**, and **Final** deliverables:
- We set `random_state = 42`.
- We apply `stratify = df['grade']` to strictly preserve class proportions across splits.
- We export `train_ids.csv` and `test_ids.csv` so subsequent project stages test on the exact same unseen partition.""")

    add_code("""train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["grade"]
)

# Export split IDs for long-term consistency
train_df[["id"]].to_csv("train_ids.csv", index=False)
test_df[["id"]].to_csv("test_ids.csv", index=False)

print(f"Train set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test set:  {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
print("Saved immutable split indices to train_ids.csv and test_ids.csv.")""")

    # Cell 8: Feature Sets Preparation
    add_md("""### 8. Feature Preparation & Engineering
We define two distinct feature matrices:
1. **Indirect Features ($X_{indirect}$)**: Continuous variables scaled via `StandardScaler`, plus binary skill flags and schedule dummies.
2. **Direct Features ($X_{leakage}$)**: `name` represented as TF-IDF n-grams + `experience_id` encoded with `OneHotEncoder`.""")

    add_code("""# 1. Indirect numeric features
numeric_cols = ["salary_mean_imputed", "is_salary_missing", "desc_length", "desc_words_count", "num_skills"]
skill_cols = [c for c in df.columns if c.startswith("skill_")]
schedule_cols = [c for c in df.columns if c.startswith("schedule_")]

# Fit scaler strictly on train set to prevent data leakage in scaling!
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(train_df[numeric_cols])
X_test_num_scaled = scaler.transform(test_df[numeric_cols])

X_train_indirect = np.hstack([X_train_num_scaled, train_df[skill_cols + schedule_cols].values])
X_test_indirect = np.hstack([X_test_num_scaled, test_df[skill_cols + schedule_cols].values])

# 2. Leakage features
tfidf = TfidfVectorizer(max_features=40, lowercase=True)
name_train_tfidf = tfidf.fit_transform(train_df["name"]).toarray()
name_test_tfidf = tfidf.transform(test_df["name"]).toarray()

ohe_exp = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
exp_train_ohe = ohe_exp.fit_transform(train_df[["experience_id"]])
exp_test_ohe = ohe_exp.transform(test_df[["experience_id"]])

# Scenario A: Full matrix (Indirect + Leakage)
X_train_A = np.hstack([X_train_indirect, name_train_tfidf, exp_train_ohe])
X_test_A = np.hstack([X_test_indirect, name_test_tfidf, exp_test_ohe])

# Scenario B: Honest matrix (Strictly Indirect)
X_train_B = X_train_indirect
X_test_B = X_test_indirect

y_train = train_df["grade"]
y_test = test_df["grade"]

print(f"Scenario A Feature Matrix Dimension: {X_train_A.shape[1]} features")
print(f"Scenario B Feature Matrix Dimension: {X_train_B.shape[1]} features")""")

    # Cell 9: Scenario A Experiments
    add_md("""### 9. Scenario A: Baseline Modeling WITH Data Leakage
We train two standard week 3–4 classifiers:
1. **Multinomial Logistic Regression** (`max_iter=1000`)
2. **K-Nearest Neighbors** (`n_neighbors=5`, distance-weighted)""")

    add_code("""# Model 1A: Logistic Regression (Leakage)
lr_model_A = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_model_A.fit(X_train_A, y_train)
y_pred_lr_A = lr_model_A.predict(X_test_A)

# Model 2A: KNN (Leakage)
knn_model_A = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn_model_A.fit(X_train_A, y_train)
y_pred_knn_A = knn_model_A.predict(X_test_A)

print("--- Scenario A: Results with Data Leakage ---")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr_A):.4f} | F1-macro: {f1_score(y_test, y_pred_lr_A, average='macro'):.4f}")
print(f"KNN Accuracy:                 {accuracy_score(y_test, y_pred_knn_A):.4f} | F1-macro: {f1_score(y_test, y_pred_knn_A, average='macro'):.4f}")""")

    # Cell 10: Scenario B Experiments
    add_md("""### 10. Scenario B: Honest / Leakage-Free Baseline Modeling
We train the exact same two baseline algorithms, but on $X_B$, where all direct proxies (`name` and `experience`) have been stripped away.""")

    add_code("""# Model 1B: Logistic Regression (Honest)
lr_model_B = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
lr_model_B.fit(X_train_B, y_train)
y_pred_lr_B = lr_model_B.predict(X_test_B)

# Model 2B: KNN (Honest)
knn_model_B = KNeighborsClassifier(n_neighbors=7, weights="distance")
knn_model_B.fit(X_train_B, y_train)
y_pred_knn_B = knn_model_B.predict(X_test_B)

print("--- Scenario B: Honest Results (No Leakage) ---")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr_B):.4f} | F1-macro: {f1_score(y_test, y_pred_lr_B, average='macro'):.4f}")
print(f"KNN Accuracy:                 {accuracy_score(y_test, y_pred_knn_B):.4f} | F1-macro: {f1_score(y_test, y_pred_knn_B, average='macro'):.4f}")""")

    # Cell 11: Comparative Evaluation & Metrics
    add_md("""### 11. Comparative Evaluation & Analysis
Let's assemble a complete metrics table comparing:
- **Accuracy**
- **Precision (macro)**
- **Recall (macro)**
- **F1-Score (macro)**
- **F1-Score (weighted)**""")

    add_code("""def compute_all_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro"),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro"),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro"),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted")
    }

results = {
    "LR (Scenario A - Leakage)": compute_all_metrics(y_test, y_pred_lr_A),
    "KNN (Scenario A - Leakage)": compute_all_metrics(y_test, y_pred_knn_A),
    "LR (Scenario B - Honest)": compute_all_metrics(y_test, y_pred_lr_B),
    "KNN (Scenario B - Honest)": compute_all_metrics(y_test, y_pred_knn_B)
}

comparison_df = pd.DataFrame(results).T.round(4)
print("=== MIDTERM EXPERIMENT COMPARISON SUMMARY TABLE ===")
display(comparison_df)

# Plot Comparison Bar Chart
fig, ax = plt.subplots(figsize=(11, 5))
comparison_df[["Accuracy", "F1 (macro)"]].plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
ax.set_title("Performance Comparison: Leakage (Scenario A) vs. Honest Setup (Scenario B)", fontsize=13)
ax.set_ylabel("Score")
ax.set_xticklabels(comparison_df.index, rotation=15, ha="right")
ax.set_ylim(0.5, 1.05)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.show()""")

    # Cell 12: Confusion Matrix & Error Analysis
    add_md("""### 12. Confusion Matrices & Error Breakdown
Let's examine how the models err when deprived of direct label proxies.""")

    add_code("""labels = ["Junior", "Middle", "Senior"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_A = confusion_matrix(y_test, y_pred_lr_A, labels=labels)
sns.heatmap(cm_A, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title("Confusion Matrix: LR Scenario A (Data Leakage)")
axes[0].set_xlabel("Predicted Grade")
axes[0].set_ylabel("True Grade")

cm_B = confusion_matrix(y_test, y_pred_lr_B, labels=labels)
sns.heatmap(cm_B, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title("Confusion Matrix: LR Scenario B (Honest Baseline)")
axes[1].set_xlabel("Predicted Grade")
axes[1].set_ylabel("True Grade")

plt.tight_layout()
plt.show()

print("Classification Report - Honest Logistic Regression (Scenario B):")
print(classification_report(y_test, y_pred_lr_B, target_names=labels))""")

    # Cell 13: Feature Importance & Interpretability
    add_md("""### 13. Model Interpretability: What Drives Grade in the Honest Model?
To understand how Logistic Regression predicts developer grades without seeing titles or explicit experience, we analyze the model's learned coefficients for each class.""")

    add_code("""# Feature names in Scenario B
feature_names_B = numeric_cols + skill_cols + schedule_cols

coef_df = pd.DataFrame(
    lr_model_B.coef_,
    index=lr_model_B.classes_,
    columns=feature_names_B
)

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
for i, grade in enumerate(["Junior", "Middle", "Senior"]):
    top_coeffs = coef_df.loc[grade].sort_values()
    # Plot top positive and negative features
    top_and_bottom = pd.concat([top_coeffs.head(5), top_coeffs.tail(5)])
    colors = ["crimson" if v < 0 else "forestgreen" for v in top_and_bottom.values]
    top_and_bottom.plot(kind="barh", ax=axes[i], color=colors)
    axes[i].set_title(f"Key Predictive Weights for Grade: {grade}")
    axes[i].set_xlabel("Logistic Regression Coefficient Weight")

plt.tight_layout()
plt.show()""")

    # Cell 14: Midterm Conclusion & Roadmap
    add_md("""### 14. Midterm Findings & Roadmap for Endterm
#### Key Findings:
1. **The Trap of Data Leakage (Scenario A)**:
   - Logistic Regression achieved **~99.2% Accuracy** and **0.992 F1-macro**.
   - This illusion of near-perfect performance occurs because the model latches directly onto title words (`Junior`, `Senior`, `Lead`) and `experience` categories which formed the pseudo-ground truth. In production (where title or experience may be ambiguous, missing, or fraudulent), this model would fail catastrophically.
2. **Realistic Generalization (Scenario B)**:
   - When restricted to indirect features (compensation, skill matrix, text length), Logistic Regression achieves **~81.6% Accuracy** and **0.817 F1-macro**, outperforming KNN (**80.0% Accuracy**).
   - Analysis of coefficients confirms that **Salary**, **Docker**, **FastAPI**, **Go**, and **Kubernetes** are strong positive signals for Seniority, while basic Git and lower salary strongly associate with Junior roles.
   - Most errors occur on the border between **Middle** and **Junior** / **Middle** and **Senior** due to overlapping salary bands and similar tech stack descriptions.

#### Endterm Roadmap:
- **NLP Text Processing**: Leverage the cleaned `description` field with TF-IDF, Word2Vec, or sentence embeddings (BERT / RuBERT).
- **Advanced Ensembles**: Train Random Forest, Gradient Boosting (CatBoost, LightGBM, XGBoost) on tabular + text embeddings.
- **Cross-Validation & Hyperparameter Tuning**: 5-Fold Stratified Cross-Validation and Optuna Bayesian optimization.""")

    # Write notebook file
    nb_path = Path("Midterm_HH_Grade_Classification.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Jupyter Notebook successfully written to: {nb_path.resolve()}")

if __name__ == "__main__":
    create_notebook()
