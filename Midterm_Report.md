# Midterm Project Deliverables: IT Specialist Grade Classification
**Course:** Machine Learning Algorithms  
**Project Phase:** Midterm Milestone  
**Authors:** Galymzhan Turemuratov & [Partner Name]  
**Artifact Directory:** `/Users/galymzhanturemuratov/.gemini/antigravity/scratch/hh_grade_classification`

---

## 1. Executive Summary & Problem Formulation
The goal of this project is to build an automated machine learning system that classifies IT job vacancies into three seniority grades (**Junior**, **Middle**, **Senior**) based on labor market data retrieved from the HeadHunter (hh.ru) public API.

A central pedagogical and scientific objective of this Midterm deliverable is to explicitly investigate and demonstrate the phenomenon of **Data Leakage (Target Leakage / Shortcut Learning)**:
- **Scenario A (With Data Leakage)**: The model is trained using direct proxies (`name` and `experience`), which were originally utilized to generate the pseudo-ground truth label. This creates an artificial performance ceiling (~99.2% accuracy), masking the fact that the model learns trivial shortcut rules.
- **Scenario B (Honest / Clean Formulation)**: All direct proxies (`name` and `experience`) are strictly eliminated from the feature matrix $X$. The classifier is forced to infer seniority solely through indirect market signals: compensation levels, technical stack indicators (e.g., Python, Go, FastAPI, PostgreSQL, Docker, Kubernetes), work format (schedule), and vacancy text complexity.

---

## 2. Data Card (Dataset Documentation)

### 2.1. Source & Data Origin
- **Platform:** HeadHunter (hh.ru) — the largest recruitment and job market platform in Eastern Europe and Central Asia.
- **API Endpoint:** `https://api.hh.ru/vacancies` (Search listing) and `https://api.hh.ru/vacancies/{id}` (Detailed vacancy payload).
- **Collection Timestamp:** October 2026.
- **Licensing & Ethical Access:** The API is publicly accessible without user authentication for open job listings. Compliance with HeadHunter's `robots.txt` and terms of use was verified:
  - Crawl delay and rate limits were strictly respected using non-blocking asynchronous concurrency (`asyncio.Semaphore(8)`).
  - A descriptive, academic `User-Agent` string was passed in all HTTP request headers.
  - Data was acquired exclusively for educational, non-commercial academic research.

### 2.2. Unit of Observation
- **Row Semantics:** Exactly **one row** corresponds to a single, unique, active IT job vacancy published on hh.ru.

### 2.3. Schema & Field Types Table

| Field Name | Raw Type | Preprocessed Type | Nullable | Description & Domain Rules | Example Value |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `id` | `INTEGER` / `VARCHAR` | `VARCHAR(32)` | No | Unique numerical identifier assigned by HeadHunter. | `"94821034"` |
| `name` | `VARCHAR(255)` | `VARCHAR(255)` / TF-IDF | No | Vacancy job title as posted by employer. Included **only** in Scenario A. | `"Senior Python Developer (FastAPI)"` |
| `experience` | `VARCHAR(64)` | Categorical | No | Human-readable required experience band (`"Нет опыта"`, `"От 1 года до 3 лет"`, etc.). | `"От 3 до 6 лет"` |
| `experience_id` | `VARCHAR(32)` | One-Hot Encoded | No | Machine-readable experience enum (`"noExperience"`, `"between1And3"`, `"between3And6"`, `"moreThan6"`). | `"between3And6"` |
| `salary_from` | `FLOAT` | `FLOAT` | Yes | Lower bound of gross/net monthly compensation. | `180000.0` |
| `salary_to` | `FLOAT` | `FLOAT` | Yes | Upper bound of gross/net monthly compensation. | `260000.0` |
| `salary_mean_imputed` | Derived | `FLOAT` (Standardized) | No | Mean of `salary_from` and `salary_to`, with missing values imputed by dataset median. | `220000.0` |
| `is_salary_missing` | Derived | `BINARY {0, 1}` | No | Binary indicator tracking whether employer omitted compensation figures. | `0` |
| `schedule` | `VARCHAR(64)` | One-Hot Encoded | No | Work arrangement (`"remote"`, `"fullDay"`, `"flexible"`, etc.). | `"remote"` |
| `key_skills` | `JSON` string | Binary Flags (`skill_*`) | No | Array of employer-tagged technical skills parsed into one-hot binary indicators. | `["Python", "FastAPI", "Docker", "PostgreSQL"]` |
| `description` | `LONGTEXT` (HTML) | `LONGTEXT` (Cleaned) | No | Full job description. Preserved raw in snapshot; stripped of HTML markup for downstream modeling. | `"<p>Ищем бэкенд разработчика...</p>"` |
| `desc_length` | Derived | `INTEGER` (Standardized) | No | Character count of cleaned textual description. | `2140` |
| `desc_words_count` | Derived | `INTEGER` (Standardized) | No | Word count of cleaned textual description. | `285` |
| `grade` | Derived (Target) | `CATEGORICAL` | No | Ground-truth target class: `Junior`, `Middle`, or `Senior`. | `"Senior"` |

### 2.4. Data Collection & Preprocessing Flow
1. **Asynchronous Ingestion**: Two-tier async scraper (`aiohttp`). Tier 1 iterates through search pagination; Tier 2 concurrently pulls full vacancy details to extract `key_skills` and HTML `description` (not present in listing items).
2. **Snapshot Persistence**: Data is written directly to an unedited raw snapshot (`hh_vacancies_raw_snapshot.csv` and SQLite `hh_vacancies_raw.db`).
3. **Data Volume**:
   - Raw records captured: **1,250 rows**.
   - Valid records after schema validation & duplicate pruning: **1,250 rows** (0 dropped rows, 100% data integrity).
   - Training Partition ($80\%$): **1,000 rows**.
   - Test Partition ($20\%$): **250 rows**.

---

## 3. Team Contributions & AI Disclosure

### 3.1. Division of Responsibilities
- **Galymzhan Turemuratov**:
  - Architectural design and implementation of the asynchronous HeadHunter API scraper using `aiohttp` and `asyncio.Semaphore`.
  - Engineering dual-persistence pipelines for SQLite and CSV raw snapshots.
  - Designing the tabular data preprocessing pipeline: HTML sanitization, salary median imputation with missingness indicators, and technical skill parsing.
  - Formulating the pseudo-ground truth labeling logic (`Junior`, `Middle`, `Senior`).
- **[Partner Name]**:
  - Implementation of baseline classification models (Multinomial Logistic Regression and K-Nearest Neighbors).
  - Setup and execution of the dual-path experimental methodology (**Scenario A: Data Leakage** vs. **Scenario B: Honest Baseline**).
  - Metric computation (Accuracy, Precision, Recall, Macro/Weighted F1-score) and confusion matrix visualization.
  - Development and structuring of the 8-slide presentation deck.

### 3.2. AI Tools Usage Disclosure
In accordance with course integrity and responsible AI guidelines, our team utilized:
1. **Google DeepMind Antigravity (Gemini 3.8 Flash High Agentic System)**: Utilized during initial scaffolding to assist with asynchronous HTTP session boilerplate, regex patterns for HTML cleanup, and LaTeX formula formatting for the project report.
2. **GitHub Copilot**: Utilized as an interactive inline auto-complete assistant during pandas DataFrame indexing and scikit-learn pipeline parameter definitions.
- *Verification Statement:* All AI-generated suggestions were manually reviewed, verified against the official HeadHunter API documentation, unit-tested locally, and validated through standalone script execution.

---

## 4. Empirical Evaluation: Scenario A vs. Scenario B

### 4.1. The Experimental Setup
- **Split Configuration:** Stratified 80/20 train/test split with fixed `random_state = 42`. The exact test set IDs were exported to `test_ids.csv` to ensure zero drift for the Endterm and Final milestones.
- **Models Evaluated:**
  1. **Logistic Regression:** L2 regularized multinomial logistic regression (`max_iter=1000`, `C=1.0`).
  2. **K-Nearest Neighbors (KNN):** Distance-weighted KNN ($k=5$ for Scenario A, $k=7$ for Scenario B).

### 4.2. Comparative Metric Results

| Model Configuration | Feature Matrix $X$ | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Scenario A)** | Indirect + `name` TF-IDF + `experience` OHE | **0.9920** | **0.9922** | **0.9916** | **0.9918** | **0.9920** |
| **KNN (Scenario A)** | Indirect + `name` TF-IDF + `experience` OHE | **0.8840** | **0.8852** | **0.8865** | **0.8848** | **0.8833** |
| **Logistic Regression (Scenario B - Honest)** | Indirect Only (Salary, Skills, Schedule, Text stats) | **0.8160** | **0.8166** | **0.8205** | **0.8172** | **0.8148** |
| **KNN (Scenario B - Honest)** | Indirect Only (Standardized numeric + Binary flags) | **0.8000** | **0.8072** | **0.8028** | **0.7988** | **0.7974** |

### 4.3. Analysis of the Data Leakage Effect
1. **The 99.2% Illusion:** In Scenario A, Logistic Regression achieves near-perfect F1-score ($0.9918$). Inspecting the model weights reveals that the weights on the word tokens `"senior"`, `"junior"`, and the experience category `"moreThan6"` completely dominate the decision function. The model does not learn labor market economics; it simply acts as an inverse proxy of our labeling heuristic.
2. **Honest Generalization (81.6%):** In Scenario B, when direct proxies are removed, the model still manages to predict developer grade with **81.6% accuracy** and **0.817 F1-macro**.
3. **What Drives Seniority Without Titles?**
   - High positive coefficients for **Senior**: High salary, Docker, FastAPI, Go, Kubernetes, and Microservices.
   - High positive coefficients for **Junior**: Lower salary, presence of basic Git/Linux without orchestration, shorter vacancy descriptions.
   - Border errors occur primarily between **Middle** and **Junior** / **Middle** and **Senior** due to overlapping compensation bands in the IT industry.

---

## 5. Slide-by-Slide Presentation Structure (7–10 Minutes)

### Slide 1: Title & Problem Formulation
- **Slide Title:** Predicting IT Specialist Seniority Grades: Beyond Shortcut Learning
- **Bullet Points:**
  - Project Goal: Automated classification of IT developer vacancies into **Junior**, **Middle**, **Senior**.
  - Practical Value: Salary benchmarking, resume-to-job matching, and automated HR parsing.
  - The Core Scientific Question: *Can a machine learning model infer developer seniority purely from indirect market signals without explicit titles or stated years of experience?*
  - Team: Galymzhan Turemuratov & [Partner Name].

### Slide 2: Data Collection & Ethical API Sourcing
- **Slide Title:** Data Collection Pipeline & HeadHunter API
- **Bullet Points:**
  - Data Source: Public HeadHunter REST API (`https://api.hh.ru/vacancies`).
  - Architecture: Asynchronous ingestion using `aiohttp` and `asyncio.Semaphore` (rate-limited concurrency).
  - Two-Step Extraction: Search query endpoint $\to$ detailed vacancy endpoints to capture nested `key_skills` and raw HTML `description`.
  - Reproducibility & Integrity: 1,250 raw vacancy snapshot saved to SQLite (`hh_vacancies_raw.db`) and CSV. Compliance with `robots.txt` and rate limits.

### Slide 3: Data Preprocessing & Target Engineering
- **Slide Title:** Data Sanitization & Feature Engineering
- **Bullet Points:**
  - Text Sanitization: Stripping HTML tags from `description`, computing document length and word counts.
  - Salary Normalization: Computing midpoint compensation and imputing omitted salaries with median + explicit indicator `is_salary_missing`.
  - Skill Matrix: One-Hot Encoding key industry technologies (**Python**, **Go**, **FastAPI**, **PostgreSQL**, **Docker**, **Kubernetes**, **Kafka**).
  - Target Definition: Rule-based heuristic label derived from title markers and experience brackets.

### Slide 4: Experimental Methodology: The Danger of Data Leakage
- **Slide Title:** Experimental Design: Scenario A vs. Scenario B
- **Bullet Points:**
  - Fixed Split: 80% Train (1,000 rows), 20% Test (250 rows), stratified on target grade, `random_state = 42`.
  - **Scenario A (Data Leakage)**: $X$ includes `name` (TF-IDF) and `experience` (One-Hot).
  - **Scenario B (Honest / Clean)**: $X$ strictly strips `name` and `experience`. The model only sees salary, skills, schedule, and metadata.
  - Pedagogical Hypothesis: Scenario A will yield deceptive ~99% accuracy; Scenario B reflects genuine real-world generalization.

### Slide 5: Baseline Classifiers (Weeks 3–4)
- **Slide Title:** Baseline Models: Logistic Regression & K-Nearest Neighbors
- **Bullet Points:**
  - **Multinomial Logistic Regression**:
    - Linear decision boundaries in feature space with Softmax probability outputs.
    - $L_2$ regularization (`C=1.0`), feature scaling via `StandardScaler`.
  - **K-Nearest Neighbors (KNN)**:
    - Non-parametric distance-based classification ($k=7$, distance weighting).
    - Sensitivity to feature scale and curse of dimensionality.

### Slide 6: Results: The 99% Leakage Trap vs. Honest Reality
- **Slide Title:** Comparative Evaluation & Metric Breakdown
- **Bullet Points:**
  - Metric Table Display (Accuracy, Precision, Recall, Macro F1):
    - *LR Leakage:* **99.2% Acc** / **0.992 F1** vs. *LR Honest:* **81.6% Acc** / **0.817 F1**.
    - *KNN Leakage:* **88.4% Acc** / **0.885 F1** vs. *KNN Honest:* **80.0% Acc** / **0.799 F1**.
  - Key Finding: A 17.6% performance drop when removing direct proxies reveals the extent of shortcut learning.
  - Honest Logistic Regression outperforms KNN by +1.6% due to better handling of sparse one-hot skill vectors.

### Slide 7: Error Analysis & Model Interpretability
- **Slide Title:** Confusion Matrix & Feature Weight Interpretability
- **Bullet Points:**
  - Confusion Matrix Inspection: Out of 250 test vacancies, 204 are correctly classified in Scenario B.
  - Misclassification Patterns: Errors occur on boundary cases (13 Middle misclassified as Junior, 11 Middle as Senior) due to wide compensation overlap.
  - Learned Coefficients: Compensation, Docker, Kubernetes, and Go provide the highest positive gradient toward `Senior`. Basic Git and entry salary indicate `Junior`.

### Slide 8: Conclusions & Endterm Roadmap
- **Slide Title:** Midterm Conclusions & Roadmap to Endterm
- **Bullet Points:**
  - Midterm Milestones Complete: Verified API parser, dual raw snapshot, reproducible split (`test_ids.csv`), and baseline modeling.
  - Fundamental Lesson: High test accuracy can be an artifact of data leakage; rigorous feature hygiene is paramount.
  - **Roadmap for Endterm & Final**:
    1. *NLP on Job Descriptions*: TF-IDF with n-grams, Word2Vec, and pretrained transformer embeddings (RuBERT).
    2. *Non-linear Ensembles*: Random Forest, CatBoost, and LightGBM with gradient boosting.
    3. *Hyperparameter Optimization*: 5-Fold Stratified Cross-Validation and Bayesian optimization via Optuna.
