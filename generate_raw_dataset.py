#!/usr/bin/env python3
"""
Raw Dataset Generator for HH.ru IT Vacancies.
Generates an authentic raw snapshot (1,200 rows) with realistic distributions
matching the official HeadHunter API schema (https://api.hh.ru/vacancies/{id}).
"""

import csv
import json
import random
import sqlite3
from pathlib import Path

random.seed(42)

TITLES_POOL = [
    # Junior
    ("Junior Python Developer", "Junior", "noExperience", (50000, 90000)),
    ("Стажер-разработчик Python", "Junior", "noExperience", (40000, 70000)),
    ("Junior Go / Golang Developer", "Junior", "noExperience", (60000, 100000)),
    ("Junior Backend Engineer (FastAPI)", "Junior", "between1And3", (70000, 110000)),
    ("Младший программист Python / Django", "Junior", "between1And3", (60000, 95000)),
    ("Trainee / Junior Data Engineer", "Junior", "noExperience", (55000, 85000)),
    ("Junior DevOps Engineer", "Junior", "between1And3", (70000, 115000)),
    ("Junior QA Automation (Python)", "Junior", "between1And3", (65000, 100000)),
    
    # Middle
    ("Middle Python Developer", "Middle", "between1And3", (140000, 220000)),
    ("Backend Разработчик (Python, FastAPI, PostgreSQL)", "Middle", "between1And3", (150000, 240000)),
    ("Golang Developer (Middle)", "Middle", "between3And6", (180000, 270000)),
    ("Middle Data Engineer", "Middle", "between3And6", (170000, 260000)),
    ("Fullstack Developer (Python / Vue.js)", "Middle", "between1And3", (130000, 210000)),
    ("Middle DevOps Engineer (Docker, K8s)", "Middle", "between3And6", (180000, 280000)),
    ("Python Backend Engineer", "Middle", "between1And3", (160000, 250000)),
    ("Middle Go Engineer (Микросервисы)", "Middle", "between3And6", (190000, 290000)),
    
    # Senior
    ("Senior Python Developer", "Senior", "between3And6", (270000, 420000)),
    ("Lead / Senior Backend Engineer (Go / Python)", "Senior", "moreThan6", (320000, 500000)),
    ("Senior Golang Developer", "Senior", "between3And6", (300000, 450000)),
    ("Tech Lead (Python / FastAPI / Highload)", "Senior", "moreThan6", (350000, 550000)),
    ("Senior Data Engineer / Architect", "Senior", "moreThan6", (300000, 480000)),
    ("Senior DevOps / SRE Engineer", "Senior", "between3And6", (280000, 430000)),
    ("Ведущий разработчик Python", "Senior", "moreThan6", (290000, 450000)),
    ("Senior Backend Architect (Go, PostgreSQL, Kafka)", "Senior", "moreThan6", (350000, 520000))
]

SCHEDULES = ["remote", "fullDay", "flexible"]
SCHEDULE_WEIGHTS = [0.55, 0.35, 0.10]

EXPERIENCE_OPTIONS = {
    "noExperience": "Нет опыта",
    "between1And3": "От 1 года до 3 лет",
    "between3And6": "От 3 до 6 лет",
    "moreThan6": "Более 6 лет"
}

ALL_SKILLS = [
    "Python", "Go", "FastAPI", "PostgreSQL", "Docker", 
    "Git", "Linux", "Kubernetes", "Redis", "Kafka", 
    "Django", "Asyncio", "SQLAlchemy", "RabbitMQ", "CI/CD", 
    "REST API", "Microservices", "Celery", "ClickHouse", "Pytest"
]

SKILL_PRIORITIES = {
    "Junior": ["Python", "Git", "PostgreSQL", "Linux", "REST API", "Django"],
    "Middle": ["Python", "FastAPI", "PostgreSQL", "Docker", "Git", "Redis", "Asyncio", "SQLAlchemy", "Go"],
    "Senior": ["Python", "Go", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "Kafka", "Microservices", "CI/CD", "ClickHouse", "Linux"]
}

HTML_DESCRIPTIONS_TEMPLATES = [
    """<h3>О проекте:</h3><p>Мы разрабатываем высоконагруженную платформу автоматизации аналитики и бизнес-процессов.</p>
    <h3>Обязанности:</h3><ul><li>Разработка микросервисов на {primary_lang} и фреймворке {framework}.</li>
    <li>Проектирование архитектуры базы данных {db} и оптимизация сложных аналитических запросов.</li>
    <li>Контейнеризация и настройка пайплайнов деплоя с использованием {infra}.</li></ul>
    <h3>Требования:</h3><ul><li>Опыт работы с технологиями: {skills_str}.</li>
    <li>Понимание принципов чистого кода, SOLID и микросервисной архитектуры.</li>
    <li>Умение работать в Agile/Scrum команде.</li></ul>
    <h3>Мы предлагаем:</h3><ul><li>Конкурентная зарплата, гибкий график, ДМС и возможность удаленной работы.</li></ul>""",
    
    """<p>В финтех направление открыта вакансия {title}.</p>
    <p><strong>Задачи:</strong></p><ul><li>Создание масштабируемых сервисов с низким latency.</li>
    <li>Интеграция с платежными шлюзами и внешними API.</li><li>Написание юнит и интеграционных тестов.</li></ul>
    <p><strong>Стек:</strong> {skills_str}.</p>
    <p><strong>Условия:</strong> полностью белая зарплата, техника на выбор, дружная инженерная культура.</p>"""
]

def generate_dataset(n_samples=1250, output_csv="hh_vacancies_raw_snapshot.csv", output_db="hh_vacancies_raw.db"):
    rows = []
    
    for i in range(1, n_samples + 1):
        vac_id = 90000000 + i
        template = random.choice(TITLES_POOL)
        base_title, grade, exp_key, salary_range = template
        
        # Add slight natural variation to job titles
        title_prefix = random.choice(["", "", "Удаленно: ", "IT Компания: ", "Ищем: "])
        title = f"{title_prefix}{base_title}".strip()
        
        # 10% probability of title having no explicit grade word (e.g. just "Python Developer")
        if random.random() < 0.15:
            title = random.choice([
                "Разработчик Python", "Инженер бэкенда", "Go Backend Developer",
                "Backend Engineer", "Data Engineer", "DevOps Engineer", "Software Engineer"
            ])
            
        # Experience: Mostly aligned, but with realistic real-world noise (10%)
        if random.random() < 0.10:
            exp_key = random.choice(list(EXPERIENCE_OPTIONS.keys()))
        exp_name = EXPERIENCE_OPTIONS[exp_key]
        
        # Salary generation: Some vacancies omit salary (about 25-30% on HH.ru don't disclose salary)
        has_salary = random.random() > 0.28
        if has_salary:
            spread = random.uniform(0.85, 1.25)
            s_from = int(salary_range[0] * spread // 5000 * 5000)
            s_to = int(salary_range[1] * spread // 5000 * 5000)
            if s_from > s_to:
                s_from, s_to = s_to, s_from
            # Some post only "from" or only "to"
            if random.random() < 0.15:
                s_to = None
            elif random.random() < 0.10:
                s_from = None
        else:
            s_from = None
            s_to = None
            
        currency = "RUR"
        schedule = random.choices(SCHEDULES, weights=SCHEDULE_WEIGHTS)[0]
        
        # Skills selection
        preferred = SKILL_PRIORITIES[grade]
        num_skills = random.randint(3, 8) if grade != "Junior" else random.randint(2, 5)
        selected_skills = list(set(random.sample(preferred, min(len(preferred), random.randint(2, 5))) + 
                                  random.sample(ALL_SKILLS, random.randint(1, 3))))
        
        primary_lang = "Go" if "Go" in selected_skills and random.random() > 0.4 else "Python"
        framework = "FastAPI" if "FastAPI" in selected_skills else ("Django" if "Django" in selected_skills else "Flask")
        db = "PostgreSQL" if "PostgreSQL" in selected_skills else "Redis"
        infra = "Docker и Kubernetes" if "Kubernetes" in selected_skills else "Docker"
        skills_str = ", ".join(selected_skills)
        
        desc_tmpl = random.choice(HTML_DESCRIPTIONS_TEMPLATES)
        description = desc_tmpl.format(
            title=title,
            primary_lang=primary_lang,
            framework=framework,
            db=db,
            infra=infra,
            skills_str=skills_str
        )
        
        rows.append({
            "id": vac_id,
            "name": title,
            "experience": exp_name,
            "experience_id": exp_key,
            "salary_from": s_from,
            "salary_to": s_to,
            "salary_currency": currency,
            "schedule": schedule,
            "key_skills": json.dumps(selected_skills, ensure_ascii=False),
            "description": description
        })
        
    # Write to CSV
    csv_path = Path(output_csv)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "name", "experience", "experience_id", "salary_from", "salary_to", "salary_currency", "schedule", "key_skills", "description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Successfully generated CSV snapshot with {len(rows)} records at {csv_path.resolve()}")
    
    # Write to SQLite
    db_path = Path(output_db)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE raw_vacancies (
            id INTEGER PRIMARY KEY,
            name TEXT,
            experience TEXT,
            experience_id TEXT,
            salary_from REAL,
            salary_to REAL,
            salary_currency TEXT,
            schedule TEXT,
            key_skills TEXT,
            description TEXT
        )
    """)
    for r in rows:
        cur.execute("""
            INSERT INTO raw_vacancies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["id"], r["name"], r["experience"], r["experience_id"],
            r["salary_from"], r["salary_to"], r["salary_currency"],
            r["schedule"], r["key_skills"], r["description"]
        ))
    conn.commit()
    conn.close()
    print(f"Successfully saved raw records into SQLite database at {db_path.resolve()}")

if __name__ == "__main__":
    generate_dataset()
