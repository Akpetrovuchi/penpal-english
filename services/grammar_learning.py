"""
Grammar Learning Module

Provides functions for:
- Database operations for grammar tables
- GPT-based theory and exercise generation
- Progress tracking
"""

import logging
import json
import openai
from contextlib import closing
from db import db


# ============== DATABASE SETUP ==============

def init_grammar_tables():
    """Create grammar tables if they don't exist."""
    with closing(db()) as conn:
        c = conn.cursor()
        
        # Table 1: Grammar topics with theory
        c.execute("""
            CREATE TABLE IF NOT EXISTS grammar_topics (
                id SERIAL PRIMARY KEY,
                level TEXT NOT NULL,
                code TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                theory TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Table 2: Grammar exercises
        c.execute("""
            CREATE TABLE IF NOT EXISTS grammar_exercises (
                id SERIAL PRIMARY KEY,
                topic_id INT REFERENCES grammar_topics(id),
                question TEXT NOT NULL,
                options TEXT[] NOT NULL,
                correct_index INT NOT NULL,
                explanation TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Table 3: User progress
        c.execute("""
            CREATE TABLE IF NOT EXISTS grammar_user_progress (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                exercise_id INT REFERENCES grammar_exercises(id),
                is_correct BOOLEAN,
                answered_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (user_id, exercise_id)
            )
        """)
        
        # Table 4: User completed topics (for checkmarks and paywall)
        c.execute("""
            CREATE TABLE IF NOT EXISTS grammar_user_topics (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                topic_id INT REFERENCES grammar_topics(id),
                theory_read BOOLEAN DEFAULT FALSE,
                completed BOOLEAN DEFAULT FALSE,
                first_started_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                UNIQUE (user_id, topic_id)
            )
        """)
        
        conn.commit()
        logging.info("[grammar] Tables initialized")


def seed_grammar_topics():
    """Insert initial grammar topics for each level if not exist."""
    topics = [
        # A2 Level
        ("A2", "present_simple", "Present Simple"),
        ("A2", "present_continuous", "Present Continuous"),
        ("A2", "past_simple", "Past Simple"),
        ("A2", "future_will", "Future with Will"),
        ("A2", "comparatives", "Comparatives"),
        ("A2", "superlatives", "Superlatives"),
        ("A2", "articles", "Articles (a/an/the)"),
        ("A2", "prepositions_time", "Prepositions of Time"),
        
        # B1 Level
        ("B1", "present_perfect", "Present Perfect"),
        ("B1", "present_perfect_continuous", "Present Perfect Continuous"),
        ("B1", "past_continuous", "Past Continuous"),
        ("B1", "used_to", "Used to"),
        ("B1", "first_conditional", "First Conditional"),
        ("B1", "second_conditional", "Second Conditional"),
        ("B1", "passive_voice", "Passive Voice"),
        ("B1", "modals_obligation", "Modals of Obligation"),
        
        # B2 Level
        ("B2", "past_perfect", "Past Perfect"),
        ("B2", "third_conditional", "Third Conditional"),
        ("B2", "mixed_conditionals", "Mixed Conditionals"),
        ("B2", "reported_speech", "Reported Speech"),
        ("B2", "relative_clauses", "Relative Clauses"),
        ("B2", "wish_clauses", "Wish Clauses"),
        ("B2", "inversion", "Inversion"),
        ("B2", "advanced_passives", "Advanced Passives"),
        
        # C1 Level
        ("C1", "cleft_sentences", "Cleft Sentences"),
        ("C1", "subjunctive", "Subjunctive Mood"),
        ("C1", "participle_clauses", "Participle Clauses"),
        ("C1", "advanced_modals", "Advanced Modal Verbs"),
        ("C1", "ellipsis_substitution", "Ellipsis & Substitution"),
        ("C1", "discourse_markers", "Discourse Markers"),
        ("C1", "fronting", "Fronting"),
        ("C1", "nominal_clauses", "Nominal Clauses"),
    ]
    
    with closing(db()) as conn:
        c = conn.cursor()
        for level, code, title in topics:
            c.execute("""
                INSERT INTO grammar_topics (level, code, title)
                VALUES (%s, %s, %s)
                ON CONFLICT (code) DO NOTHING
            """, (level, code, title))
        conn.commit()
        logging.info(f"[grammar] Seeded {len(topics)} topics")


# ============== DATABASE QUERIES ==============

def get_topics_by_level(level: str) -> list:
    """Get all topics for a given level."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, code, title, theory
            FROM grammar_topics
            WHERE level = %s
            ORDER BY id
        """, (level,))
        rows = c.fetchall()
        return [{"id": r[0], "code": r[1], "title": r[2], "theory": r[3]} for r in rows]


def get_topic_by_id(topic_id: int) -> dict:
    """Get topic by ID."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, level, code, title, theory
            FROM grammar_topics
            WHERE id = %s
        """, (topic_id,))
        r = c.fetchone()
        if r:
            return {"id": r[0], "level": r[1], "code": r[2], "title": r[3], "theory": r[4]}
        return None


def save_theory(topic_id: int, theory: str):
    """Save generated theory for a topic."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE grammar_topics
            SET theory = %s
            WHERE id = %s
        """, (theory, topic_id))
        conn.commit()


def clear_topic_theory(topic_id: int):
    """Clear theory for a topic (to regenerate)."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE grammar_topics SET theory = NULL WHERE id = %s", (topic_id,))
        conn.commit()
        logging.info(f"[grammar] Cleared theory for topic {topic_id}")


def clear_topic_exercises(topic_id: int):
    """Delete all exercises for a topic (to regenerate)."""
    with closing(db()) as conn:
        c = conn.cursor()
        # First delete user progress for these exercises
        c.execute("""
            DELETE FROM grammar_user_progress 
            WHERE exercise_id IN (SELECT id FROM grammar_exercises WHERE topic_id = %s)
        """, (topic_id,))
        # Then delete exercises
        c.execute("DELETE FROM grammar_exercises WHERE topic_id = %s", (topic_id,))
        conn.commit()
        logging.info(f"[grammar] Cleared exercises for topic {topic_id}")


def get_exercise_count(topic_id: int) -> int:
    """Get number of exercises for a topic."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM grammar_exercises
            WHERE topic_id = %s
        """, (topic_id,))
        return c.fetchone()[0]


def save_exercise(topic_id: int, question: str, options: list, correct_index: int, explanation: str):
    """Save a generated exercise."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO grammar_exercises (topic_id, question, options, correct_index, explanation)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (topic_id, question, options, correct_index, explanation))
        exercise_id = c.fetchone()[0]
        conn.commit()
        return exercise_id


def get_next_exercise(topic_id: int, user_id: int) -> dict:
    """Get next unanswered exercise for user."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, question, options, correct_index, explanation
            FROM grammar_exercises
            WHERE topic_id = %s
            AND id NOT IN (
                SELECT exercise_id FROM grammar_user_progress WHERE user_id = %s
            )
            ORDER BY id
            LIMIT 1
        """, (topic_id, user_id))
        r = c.fetchone()
        if r:
            return {
                "id": r[0],
                "question": r[1],
                "options": r[2],
                "correct_index": r[3],
                "explanation": r[4]
            }
        return None


def get_random_exercise(topic_id: int, user_id: int) -> dict:
    """Get random exercise (allows repeating after all done)."""
    with closing(db()) as conn:
        c = conn.cursor()
        # First try to get an unanswered exercise
        c.execute("""
            SELECT id, question, options, correct_index, explanation
            FROM grammar_exercises
            WHERE topic_id = %s
            AND id NOT IN (
                SELECT exercise_id FROM grammar_user_progress WHERE user_id = %s
            )
            ORDER BY RANDOM()
            LIMIT 1
        """, (topic_id, user_id))
        r = c.fetchone()
        if r:
            return {
                "id": r[0],
                "question": r[1],
                "options": r[2],
                "correct_index": r[3],
                "explanation": r[4]
            }
        # If all answered, get any random one
        c.execute("""
            SELECT id, question, options, correct_index, explanation
            FROM grammar_exercises
            WHERE topic_id = %s
            ORDER BY RANDOM()
            LIMIT 1
        """, (topic_id,))
        r = c.fetchone()
        if r:
            return {
                "id": r[0],
                "question": r[1],
                "options": r[2],
                "correct_index": r[3],
                "explanation": r[4]
            }
        return None


def save_user_progress(user_id: int, exercise_id: int, is_correct: bool):
    """Record user's answer."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO grammar_user_progress (user_id, exercise_id, is_correct)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, exercise_id) DO UPDATE SET
                is_correct = EXCLUDED.is_correct,
                answered_at = NOW()
        """, (user_id, exercise_id, is_correct))
        conn.commit()


def get_user_topic_stats(user_id: int, topic_id: int) -> dict:
    """Get user's stats for a topic."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE is_correct = true) as correct
            FROM grammar_user_progress gup
            JOIN grammar_exercises ge ON gup.exercise_id = ge.id
            WHERE gup.user_id = %s AND ge.topic_id = %s
        """, (user_id, topic_id))
        r = c.fetchone()
        return {"total": r[0], "correct": r[1]}


# ============== USER TOPIC TRACKING ==============

def mark_topic_started(user_id: int, topic_id: int):
    """Mark that user started a topic (for paywall counting)."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO grammar_user_topics (user_id, topic_id, theory_read, completed)
            VALUES (%s, %s, FALSE, FALSE)
            ON CONFLICT (user_id, topic_id) DO NOTHING
        """, (user_id, topic_id))
        conn.commit()


def mark_theory_read(user_id: int, topic_id: int):
    """Mark that user read the theory for a topic."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO grammar_user_topics (user_id, topic_id, theory_read, completed)
            VALUES (%s, %s, TRUE, FALSE)
            ON CONFLICT (user_id, topic_id) DO UPDATE SET theory_read = TRUE
        """, (user_id, topic_id))
        conn.commit()


def mark_topic_completed(user_id: int, topic_id: int):
    """Mark topic as completed (theory read + 7 exercises done)."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO grammar_user_topics (user_id, topic_id, theory_read, completed, completed_at)
            VALUES (%s, %s, TRUE, TRUE, NOW())
            ON CONFLICT (user_id, topic_id) DO UPDATE SET 
                completed = TRUE,
                completed_at = NOW()
        """, (user_id, topic_id))
        conn.commit()
        logging.info(f"[grammar] User {user_id} completed topic {topic_id}")


def is_topic_completed(user_id: int, topic_id: int) -> bool:
    """Check if user has completed a topic."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT completed FROM grammar_user_topics
            WHERE user_id = %s AND topic_id = %s
        """, (user_id, topic_id))
        r = c.fetchone()
        return r[0] if r else False


def get_completed_topics(user_id: int) -> set:
    """Get set of completed topic IDs for a user."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT topic_id FROM grammar_user_topics
            WHERE user_id = %s AND completed = TRUE
        """, (user_id,))
        return {r[0] for r in c.fetchall()}


def get_new_topics_started_today(user_id: int) -> int:
    """Count how many NEW topics user started today (for paywall)."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM grammar_user_topics
            WHERE user_id = %s 
            AND first_started_at::date = CURRENT_DATE
        """, (user_id,))
        return c.fetchone()[0]


def has_user_started_topic(user_id: int, topic_id: int) -> bool:
    """Check if user has already started this topic (not new)."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT 1 FROM grammar_user_topics
            WHERE user_id = %s AND topic_id = %s
        """, (user_id, topic_id))
        return c.fetchone() is not None


# ============== GPT GENERATION ==============

# Model for grammar content generation (use gpt-4o for higher quality)
GRAMMAR_MODEL = "gpt-4o"

# Exercise types for variety
EXERCISE_TYPES = [
    "fill_blank",      # Fill in the blank: She ___ to work.
    "error_correction", # Find the error: She go to work every day.
    "choose_correct",   # Which is correct? A) She go B) She goes
    "transformation",   # Rewrite: She goes to work. (negative) → ___
    "complete_sentence", # Complete: If I ___ rich, I would...
]

import random
import re


def markdown_to_html(text: str) -> str:
    """Convert markdown formatting to Telegram HTML."""
    # Bold: **text** or __text__ → <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    # Italic: *text* or _text_ → <i>text</i>
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<i>\1</i>', text)
    # Code: `text` → <code>text</code>
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    return text


async def generate_grammar_theory(topic_title: str, level: str) -> str:
    """Generate theory explanation for a grammar topic using GPT-4o."""
    
    level_descriptions = {
        "A2": "начинающий (Elementary) - используй простые объяснения и базовую лексику",
        "B1": "средний (Intermediate) - можно использовать более сложные примеры",
        "B2": "выше среднего (Upper-Intermediate) - включи нюансы и исключения",
        "C1": "продвинутый (Advanced) - добавь сложные случаи и стилистические различия"
    }
    
    level_desc = level_descriptions.get(level, "средний")
    
    prompt = f"""Ты опытный преподаватель английского языка. Напиши понятное объяснение грамматической темы "{topic_title}" для студента уровня {level} ({level_desc}).

ВАЖНЫЕ ТРЕБОВАНИЯ:
1. Пиши на русском языке, примеры на английском с переводом
2. Объясни КОГДА и ЗАЧЕМ используется эта конструкция (не только КАК)
3. Дай 3-4 разнообразных примера из реальной жизни
4. Покажи типичные ошибки русскоговорящих студентов
5. Добавь мнемонический приём или ассоциацию для запоминания
6. Длина: 400-600 слов

СТРУКТУРА (используй эмодзи):

📚 <b>{topic_title}</b>

🎯 Когда использовать:
[Объясни ситуации, когда нужна эта конструкция]

📝 Как образуется:
[Формула/структура с примерами]

✅ Примеры из жизни:
• [Пример 1] — [перевод]
• [Пример 2] — [перевод]
• [Пример 3] — [перевод]

⚠️ Типичные ошибки:
❌ [неправильно] → ✓ [правильно]
[объяснение почему]

💡 Как запомнить:
[Мнемоника или ассоциация]

🔑 Маркеры (слова-подсказки):
[Слова, которые часто используются с этой конструкцией]"""

    try:
        response = openai.ChatCompletion.create(
            model=GRAMMAR_MODEL,
            messages=[
                {"role": "system", "content": "Ты профессиональный преподаватель английского с 15-летним опытом. Твои объяснения всегда понятны, структурированы и запоминаются благодаря живым примерам."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        theory = response.choices[0].message["content"]
        # Convert markdown to HTML for Telegram
        theory = markdown_to_html(theory)
        logging.info(f"[grammar] Generated theory for {topic_title} ({level}) with {GRAMMAR_MODEL}")
        return theory
    except Exception as e:
        logging.exception(f"[grammar] Failed to generate theory: {e}")
        return f"📚 {topic_title}\n\nК сожалению, не удалось загрузить теорию. Попробуйте позже."


async def generate_grammar_exercise(topic_title: str, level: str, existing_questions: list = None) -> dict:
    """Generate a single grammar exercise using GPT-4o with variety."""
    
    # Choose random exercise type for variety
    exercise_type = random.choice(EXERCISE_TYPES)
    
    type_instructions = {
        "fill_blank": """Тип: ЗАПОЛНИ ПРОПУСК
Создай предложение с одним пропуском (___), где нужно вставить правильную форму.
Пример: "She ___ (go) to work every day." с вариантами: goes, go, going""",
        
        "error_correction": """Тип: НАЙДИ ОШИБКУ
Создай предложение с грамматической ошибкой. Варианты: исправленные версии.
Пример: "She go to work every day. Which is correct?" с вариантами: She goes..., She going..., She is go...""",
        
        "choose_correct": """Тип: ВЫБЕРИ ПРАВИЛЬНЫЙ ВАРИАНТ
Дай начало предложения и три варианта продолжения.
Пример: "By the time she arrived, ___" с вариантами: he had left, he has left, he left""",
        
        "transformation": """Тип: ТРАНСФОРМАЦИЯ
Дай предложение и попроси изменить его (в отрицательное, вопрос, другое время).
Пример: "Make negative: She has finished her work. → She ___" с вариантами: hasn't finished, didn't finish, not finished""",
        
        "complete_sentence": """Тип: ЗАКОНЧИ ПРЕДЛОЖЕНИЕ
Дай начало предложения с контекстом и три варианта окончания.
Пример: "If I were you, I ___" с вариантами: would go, will go, going"""
    }
    
    # Build context about existing questions to avoid repetition
    avoid_context = ""
    if existing_questions and len(existing_questions) > 0:
        avoid_context = f"\n\nИЗБЕГАЙ похожих вопросов (уже есть):\n" + "\n".join(f"- {q}" for q in existing_questions[-5:])
    
    level_context = {
        "A2": "Используй простую лексику и короткие предложения. Темы: повседневная жизнь, хобби, семья.",
        "B1": "Средняя сложность. Темы: работа, путешествия, планы, опыт.",
        "B2": "Сложные предложения. Темы: карьера, общество, абстрактные понятия.",
        "C1": "Продвинутая лексика, идиомы. Темы: бизнес, наука, философия, нюансы."
    }
    
    prompt = f"""Создай ОДНО качественное упражнение по грамматике.

ТЕМА: {topic_title}
УРОВЕНЬ: {level} — {level_context.get(level, '')}

{type_instructions.get(exercise_type, type_instructions["fill_blank"])}
{avoid_context}

ТРЕБОВАНИЯ:
1. Предложение должно быть естественным и полезным в реальной жизни
2. Три варианта ответа должны быть ПРАВДОПОДОБНЫМИ (не абсурдными)
3. Только ОДИН вариант правильный
4. Неправильные варианты — типичные ошибки студентов этого уровня
5. Объяснение на русском: почему правильный ответ верный И почему другие неверны

Верни ТОЛЬКО JSON (без markdown, без ```):
{{"question": "текст вопроса", "options": ["вариант1", "вариант2", "вариант3"], "correct_index": 0, "explanation": "Объяснение на русском"}}"""

    try:
        response = openai.ChatCompletion.create(
            model=GRAMMAR_MODEL,
            messages=[
                {"role": "system", "content": "Ты создаёшь упражнения для изучения английской грамматики. Твои упражнения разнообразны, практичны и проверяют реальное понимание, а не механическое запоминание. Отвечай ТОЛЬКО валидным JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.85,
            max_tokens=400
        )
        content = response.choices[0].message["content"].strip()
        
        # Clean up response if wrapped in markdown
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines with ```
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        exercise = json.loads(content)
        
        # Validate structure
        required = ["question", "options", "correct_index", "explanation"]
        if not all(k in exercise for k in required):
            raise ValueError("Missing required fields")
        if len(exercise["options"]) != 3:
            raise ValueError("Must have exactly 3 options")
        if not 0 <= exercise["correct_index"] <= 2:
            raise ValueError("correct_index must be 0, 1, or 2")
        
        # Validate that correct answer makes sense
        correct_answer = exercise["options"][exercise["correct_index"]]
        if len(correct_answer.strip()) < 1:
            raise ValueError("Empty correct answer")
            
        logging.info(f"[grammar] Generated {exercise_type} exercise for {topic_title}")
        return exercise
    except json.JSONDecodeError as e:
        logging.error(f"[grammar] JSON parse error: {e}, content: {content[:200]}")
        return None
    except Exception as e:
        logging.exception(f"[grammar] Failed to generate exercise: {e}")
        return None


async def ensure_exercises_for_topic(topic_id: int, topic_title: str, level: str, needed: int = 50):
    """Ensure at least `needed` exercises exist for a topic."""
    current_count = get_exercise_count(topic_id)
    to_generate = needed - current_count
    
    if to_generate <= 0:
        logging.info(f"[grammar] Topic {topic_id} already has {current_count} exercises")
        return current_count
    
    logging.info(f"[grammar] Generating {to_generate} exercises for topic {topic_id}")
    
    # Get existing questions to avoid repetition
    existing_questions = []
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT question FROM grammar_exercises WHERE topic_id = %s", (topic_id,))
        existing_questions = [r[0] for r in c.fetchall()]
    
    generated = 0
    failures = 0
    max_failures = 3
    
    for _ in range(to_generate):
        exercise = await generate_grammar_exercise(topic_title, level, existing_questions)
        if exercise:
            save_exercise(
                topic_id=topic_id,
                question=exercise["question"],
                options=exercise["options"],
                correct_index=exercise["correct_index"],
                explanation=exercise["explanation"]
            )
            existing_questions.append(exercise["question"])
            generated += 1
            failures = 0  # Reset failure counter on success
        else:
            failures += 1
            if failures >= max_failures:
                logging.warning(f"[grammar] Too many failures, stopping generation")
                break
        
        # Limit generation per call to avoid timeouts
        if generated >= 10:
            break
    
    logging.info(f"[grammar] Generated {generated} new exercises for topic {topic_id}")
    return current_count + generated


# ============== INITIALIZATION ==============

def init_grammar_module():
    """Initialize grammar module: create tables and seed topics."""
    init_grammar_tables()
    seed_grammar_topics()
    logging.info("[grammar] Module initialized")
