#!/usr/bin/env python3
"""
Midterm Baseline Modeling & Data Leakage Experiments.
Course: Machine Learning Algorithms
Authors: Galymzhan Turemuratov & Partner

Implements:
1. Data cleaning and preprocessing (HTML stripping, salary imputation, skills parsing).
2. Target generation ('grade': Junior, Middle, Senior).
3. Fixed train-test split (random_state=42, stratify=y).
4. Scenario A: Models trained WITH Data Leakage (name TF-IDF + experience included).
5. Scenario B: Honest models WITHOUT Data Leakage (name & experience strictly excluded).
6. Comprehensive metric evaluation (Accuracy, Precision, Recall, F1-macro, F1-weighted).
"""

import json
import re
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

RANDOM_STATE = 42

def clean_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"&[a-z]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def assign_grade(row: pd.Series) -> str:
    """
    Generate target variable 'grade' based on experience and job title keywords.
    Hierarchy:
    1. Explicit title markers (Lead/Senior/Junior/Intern)
    2. Experience category mapping
    """
    title = str(row.get("name", "")).lower()
    exp = str(row.get("experience", "")).lower()
    exp_id = str(row.get("experience_id", "")).lower()
    
    # Priority 1: Junior markers
    if any(k in title for k in ["junior", "стажер", "intern", "младший", "trainee"]):
        return "Junior"
    if exp_id == "noexperience" or "нет опыта" in exp:
        return "Junior"
        
    # Priority 2: Senior markers
    if any(k in title for k in ["senior", "lead", "лид", "тимлид", "архитектор", "ведущий", "сеньор", "head"]):
        return "Senior"
    if exp_id == "morethan6" or "более 6" in exp:
        return "Senior"
        
    # Priority 3: Middle markers
    if any(k in title for k in ["middle", "мидл"]):
        return "Middle"
    if exp_id in ["between1and3", "between3and6"] or "от 1 года" in exp or "от 3" in exp:
        return "Middle"
        
    return "Middle"

def parse_skills_list(skills_val) -> list:
    """Parse key_skills JSON string or python list."""
    if isinstance(skills_val, list):
        return skills_val
    if isinstance(skills_val, str):
        try:
            return json.loads(skills_val)
        except Exception:
            return [s.strip() for s in skills_val.replace("[", "").replace("]", "").replace("'", "").split(",") if s.strip()]
    return []

def prepare_dataset(csv_path: str = "hh_vacancies_raw_snapshot.csv") -> pd.DataFrame:
    """Load raw dataset, perform cleaning, feature extraction, and target assignment."""
    df = pd.read_csv(csv_path)
    print(f"[Data Preparation] Loaded {len(df)} raw rows from {csv_path}")
    
    # Clean HTML from description
    df["description_clean"] = df["description"].apply(clean_html)
    df["desc_length"] = df["description_clean"].apply(len)
    df["desc_words_count"] = df["description_clean"].apply(lambda t: len(t.split()))
    
    # Generate Target
    df["grade"] = df.apply(assign_grade, axis=1)
    print("[Data Preparation] Target distribution:")
    print(df["grade"].value_counts(normalize=True).round(3))
    
    # Process Salary
    df["is_salary_missing"] = (df["salary_from"].isna()) & (df["salary_to"].isna())
    # Mean salary calculation
    df["salary_mean"] = df[["salary_from", "salary_to"]].mean(axis=1)
    
    # Impute missing salaries with median (to prevent data leakage, compute per-split in real pipeline,
    # or use global median for raw baseline feature)
    overall_median = df["salary_mean"].median()
    df["salary_mean_imputed"] = df["salary_mean"].fillna(overall_median)
    
    # Parse Skills & generate One-Hot / Binary indicator columns
    df["skills_list"] = df["key_skills"].apply(parse_skills_list)
    df["num_skills"] = df["skills_list"].apply(len)
    
    # Core skills to encode
    tracked_skills = ["python", "go", "fastapi", "postgresql", "docker", 
                      "git", "linux", "kubernetes", "redis", "kafka", "ci/cd", "microservices"]
    
    for skill in tracked_skills:
        col_name = f"skill_{skill.replace('/', '_')}"
        df[col_name] = df["skills_list"].apply(
            lambda skills: int(any(skill in str(s).lower() for s in skills))
        )
        
    # One-hot encode schedule
    schedule_dummies = pd.get_dummies(df["schedule"], prefix="schedule", drop_first=False, dtype=int)
    df = pd.concat([df, schedule_dummies], axis=1)
    
    return df

def run_experiments(df: pd.DataFrame):
    """Run Scenario A (Leakage) and Scenario B (Honest) experiments."""
    
    # 1. Stratified Train-Test Split with fixed random_state
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["grade"]
    )
    
    print(f"\n[Split] Train set: {len(train_df)} rows | Test set: {len(test_df)} rows")
    
    # Save split indices for Endterm & Final reproducibility
    train_df[["id"]].to_csv("train_ids.csv", index=False)
    test_df[["id"]].to_csv("test_ids.csv", index=False)
    print("[Split] Saved train_ids.csv and test_ids.csv for future project phases.")
    
    y_train = train_df["grade"]
    y_test = test_df["grade"]
    
    # Shared numerical & tabular features
    indirect_num_cols = ["salary_mean_imputed", "is_salary_missing", "desc_length", "desc_words_count", "num_skills"]
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    schedule_cols = [c for c in df.columns if c.startswith("schedule_")]
    
    indirect_features = indirect_num_cols + skill_cols + schedule_cols
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_indirect_num = scaler.fit_transform(train_df[indirect_num_cols])
    X_test_indirect_num = scaler.transform(test_df[indirect_num_cols])
    
    X_train_indirect = np.hstack([X_train_indirect_num, train_df[skill_cols + schedule_cols].values])
    X_test_indirect = np.hstack([X_test_indirect_num, test_df[skill_cols + schedule_cols].values])
    
    # =========================================================================
    # SCENARIO A: WITH DATA LEAKAGE (Including 'name' and 'experience')
    # =========================================================================
    print("\n" + "="*70)
    print("--- SCENARIO A: EXPERIMENT WITH DATA LEAKAGE (name + experience included) ---")
    print("="*70)
    
    # TF-IDF on vacancy title
    tfidf_name = TfidfVectorizer(max_features=40, lowercase=True)
    name_train_tfidf = tfidf_name.fit_transform(train_df["name"]).toarray()
    name_test_tfidf = tfidf_name.transform(test_df["name"]).toarray()
    
    # One-hot encoding on experience
    ohe_exp = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    exp_train_ohe = ohe_exp.fit_transform(train_df[["experience_id"]])
    exp_test_ohe = ohe_exp.transform(test_df[["experience_id"]])
    
    # Combine all features for Scenario A
    X_train_A = np.hstack([X_train_indirect, name_train_tfidf, exp_train_ohe])
    X_test_A = np.hstack([X_test_indirect, name_test_tfidf, exp_test_ohe])
    
    # Model 1A: Logistic Regression
    lr_A = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_A.fit(X_train_A, y_train)
    y_pred_lr_A = lr_A.predict(X_test_A)
    
    # Model 2A: KNN
    knn_A = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn_A.fit(X_train_A, y_train)
    y_pred_knn_A = knn_A.predict(X_test_A)
    
    metrics_A = {
        "LR (Leakage)": {
            "Accuracy": accuracy_score(y_test, y_pred_lr_A),
            "Precision (macro)": precision_score(y_test, y_pred_lr_A, average="macro"),
            "Recall (macro)": recall_score(y_test, y_pred_lr_A, average="macro"),
            "F1 (macro)": f1_score(y_test, y_pred_lr_A, average="macro"),
            "F1 (weighted)": f1_score(y_test, y_pred_lr_A, average="weighted"),
        },
        "KNN (Leakage)": {
            "Accuracy": accuracy_score(y_test, y_pred_knn_A),
            "Precision (macro)": precision_score(y_test, y_pred_knn_A, average="macro"),
            "Recall (macro)": recall_score(y_test, y_pred_knn_A, average="macro"),
            "F1 (macro)": f1_score(y_test, y_pred_knn_A, average="macro"),
            "F1 (weighted)": f1_score(y_test, y_pred_knn_A, average="weighted"),
        }
    }
    
    # =========================================================================
    # SCENARIO B: HONEST / LEAKAGE-FREE (Strict exclusion of name & experience)
    # =========================================================================
    print("\n" + "="*70)
    print("--- SCENARIO B: HONEST EXPERIMENT (name & experience strictly excluded) ---")
    print("="*70)
    
    X_train_B = X_train_indirect
    X_test_B = X_test_indirect
    
    # Model 1B: Logistic Regression
    lr_B = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    lr_B.fit(X_train_B, y_train)
    y_pred_lr_B = lr_B.predict(X_test_B)
    
    # Model 2B: KNN
    knn_B = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn_B.fit(X_train_B, y_train)
    y_pred_knn_B = knn_B.predict(X_test_B)
    
    metrics_B = {
        "LR (Honest)": {
            "Accuracy": accuracy_score(y_test, y_pred_lr_B),
            "Precision (macro)": precision_score(y_test, y_pred_lr_B, average="macro"),
            "Recall (macro)": recall_score(y_test, y_pred_lr_B, average="macro"),
            "F1 (macro)": f1_score(y_test, y_pred_lr_B, average="macro"),
            "F1 (weighted)": f1_score(y_test, y_pred_lr_B, average="weighted"),
        },
        "KNN (Honest)": {
            "Accuracy": accuracy_score(y_test, y_pred_knn_B),
            "Precision (macro)": precision_score(y_test, y_pred_knn_B, average="macro"),
            "Recall (macro)": recall_score(y_test, y_pred_knn_B, average="macro"),
            "F1 (macro)": f1_score(y_test, y_pred_knn_B, average="macro"),
            "F1 (weighted)": f1_score(y_test, y_pred_knn_B, average="weighted"),
        }
    }
    
    # Combine & Print Comparison Table
    all_metrics = {**metrics_A, **metrics_B}
    summary_df = pd.DataFrame(all_metrics).T.round(4)
    print("\n" + "="*70)
    print("--- COMPREHENSIVE MIDTERM RESULTS COMPARISON ---")
    print("="*70)
    print(summary_df.to_string())
    
    print("\n[Detailed Classification Report - Scenario B: Honest Logistic Regression]")
    print(classification_report(y_test, y_pred_lr_B))
    
    print("\n[Confusion Matrix - Scenario B: Honest Logistic Regression]")
    labels = ["Junior", "Middle", "Senior"]
    cm_B = confusion_matrix(y_test, y_pred_lr_B, labels=labels)
    cm_df = pd.DataFrame(cm_B, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df)
    
    summary_df.to_csv("midterm_metrics_comparison.csv")
    return summary_df

if __name__ == "__main__":
    df = prepare_dataset()
    run_experiments(df)
