"""
Word Training Module - Training words by topics.

This module handles:
- Topic selection for new word learning
- Word generation via GPT
- Progress tracking per topic
- Integration with existing word training mechanism
"""

import logging
import json
import openai
from datetime import datetime
from contextlib import closing
from db import db


# Topics for word training
WORD_TRAINING_TOPICS = {
    "travel_japan": "Путешествие в Японию 🇯🇵",
    "interview_it": "Собеседование в IT 💻",
    "complaint": "Написать жалобу 📝",
    "phrasal_verbs": "Фразовые глаголы 🔄",
    "first_date": "Первое свидание 💕",
    "surf_lesson": "Урок по серфингу 🏄",
    "slang": "Сленг 🗣️",
    "idioms": "Идиомы и выражения 💬",
}


def init_word_training_tables():
    """Create necessary tables for word training progress."""
    with closing(db()) as conn:
        c = conn.cursor()
        
        # Table to track completed topics per user
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_word_topics (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                topic_code TEXT NOT NULL,
                status TEXT DEFAULT 'completed',
                completed_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(user_id, topic_code)
            )
        """)
        
        # Add topic_code column to user_dictionary if not exists
        c.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'user_dictionary' AND column_name = 'topic_code'
                ) THEN
                    ALTER TABLE user_dictionary ADD COLUMN topic_code TEXT;
                END IF;
            END $$;
        """)
        
        conn.commit()
        logging.info("[word_training] Tables initialized")


def get_user_dictionary_word_count(user_id: int) -> int:
    """Get count of words in user's dictionary with translations."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM user_dictionary 
            WHERE user_id = %s 
              AND translation IS NOT NULL 
              AND translation != ''
        """, (user_id,))
        row = c.fetchone()
        return row[0] if row else 0


def get_completed_word_topics(user_id: int) -> set:
    """Get set of completed topic codes for user."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT topic_code FROM user_word_topics 
            WHERE user_id = %s AND status = 'completed'
        """, (user_id,))
        return {row[0] for row in c.fetchall()}


def is_word_topic_completed(user_id: int, topic_code: str) -> bool:
    """Check if user has completed a topic."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT 1 FROM user_word_topics 
            WHERE user_id = %s AND topic_code = %s AND status = 'completed'
        """, (user_id, topic_code))
        return c.fetchone() is not None


def mark_word_topic_completed(user_id: int, topic_code: str):
    """Mark a topic as completed for user."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO user_word_topics (user_id, topic_code, status, completed_at)
            VALUES (%s, %s, 'completed', NOW())
            ON CONFLICT (user_id, topic_code) DO UPDATE SET 
                status = 'completed',
                completed_at = NOW()
        """, (user_id, topic_code))
        conn.commit()
        logging.info(f"[word_training] Topic {topic_code} marked completed for user {user_id}")


def get_topic_words_for_user(user_id: int, topic_code: str) -> list:
    """Get words from user's dictionary for a specific topic."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT word, translation FROM user_dictionary 
            WHERE user_id = %s 
              AND topic_code = %s
              AND translation IS NOT NULL 
              AND translation != ''
        """, (user_id, topic_code))
        return [{"word": row[0], "translation": row[1]} for row in c.fetchall()]


def save_words_to_dictionary(user_id: int, words: list, topic_code: str):
    """Save generated words to user's dictionary.
    
    Args:
        user_id: User ID
        words: List of dicts with 'word', 'translation', 'example' keys
        topic_code: Topic code for tracking
    """
    with closing(db()) as conn:
        c = conn.cursor()
        for w in words:
            try:
                c.execute("""
                    INSERT INTO user_dictionary (user_id, word, translation, source, topic_code)
                    VALUES (%s, %s, %s, 'topic_generated', %s)
                    ON CONFLICT (user_id, word) DO UPDATE SET
                        translation = EXCLUDED.translation,
                        topic_code = EXCLUDED.topic_code
                """, (user_id, w["word"], w["translation"], topic_code))
            except Exception as e:
                logging.warning(f"[word_training] Failed to save word {w.get('word')}: {e}")
        conn.commit()
        logging.info(f"[word_training] Saved {len(words)} words for user {user_id}, topic {topic_code}")


async def generate_topic_words(level: str, topic_code: str, limit: int = 6) -> list:
    """Generate words for a topic using GPT.
    
    Returns:
        List of dicts: [{"word": "...", "translation": "...", "example": "..."}, ...]
    """
    topic_name = WORD_TRAINING_TOPICS.get(topic_code, topic_code)
    
    # Detailed context for each topic - helps GPT generate accurate translations
    topic_context = {
        "travel_japan": {
            "en": "traveling to Japan, booking hotels, ordering food, asking directions, cultural phrases",
            "ru": "путешествие в Японию, бронирование отелей, заказ еды, ориентирование, культурные фразы"
        },
        "interview_it": {
            "en": "job interviews in tech/IT industry, discussing skills, salary negotiation, asking about company",
            "ru": "собеседование в IT-компанию, обсуждение навыков, переговоры о зарплате, вопросы о компании"
        },
        "complaint": {
            "en": "writing formal complaints, expressing dissatisfaction politely, requesting refunds or resolution",
            "ru": "написание жалоб, вежливое выражение недовольства, запрос возврата или решения проблемы"
        },
        "phrasal_verbs": {
            "en": "common phrasal verbs used in everyday English conversation and their contextual meanings",
            "ru": "распространённые фразовые глаголы в повседневном английском и их значения в контексте"
        },
        "first_date": {
            "en": "casual conversation on a romantic first date, compliments, making plans, expressing interest",
            "ru": "непринуждённая беседа на первом свидании, комплименты, планирование, выражение интереса"
        },
        "surf_lesson": {
            "en": "surfing terminology, beach and ocean vocabulary, water sports instructions and safety",
            "ru": "сёрфинг-терминология, пляжная лексика, инструкции по водным видам спорта и безопасности"
        },
        "slang": {
            "en": "modern informal English expressions and slang used by native speakers in casual settings",
            "ru": "современный неформальный английский, сленг носителей языка в повседневной речи"
        },
        "idioms": {
            "en": "common English idioms, fixed expressions and their figurative meanings in context",
            "ru": "распространённые английские идиомы, устойчивые выражения и их переносные значения"
        },
    }
    
    ctx = topic_context.get(topic_code, {"en": f"topic: {topic_name}", "ru": topic_name})
    context_en = ctx["en"]
    context_ru = ctx["ru"]
    
    prompt = f"""Generate {limit} useful English words/phrases for a {level} level student.

TOPIC CONTEXT: {context_en}
КОНТЕКСТ ТЕМЫ: {context_ru}

Requirements:
1. Words should be practical and commonly used IN THIS SPECIFIC CONTEXT
2. Mix of single words and short phrases (2-3 words max)
3. Appropriate for {level} level learner
4. Translation to Russian must be ACCURATE for THIS CONTEXT (not general dictionary meaning!)
   - Example: "reservation" in travel context = "бронь/бронирование", not "резервация"
   - Example: "run" in IT context might be "запустить (программу)", not "бежать"
5. Include a short example sentence showing the word in context

Return ONLY valid JSON array (no markdown):
[
  {{"word": "word/phrase in English", "translation": "точный перевод в контексте", "example": "Example sentence using the word in context."}},
  ...
]"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an English vocabulary teacher. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up potential markdown
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        words = json.loads(content)
        
        # Validate structure
        if not isinstance(words, list):
            raise ValueError("Response is not a list")
        
        validated = []
        for w in words[:limit]:
            if isinstance(w, dict) and "word" in w and "translation" in w:
                validated.append({
                    "word": w["word"],
                    "translation": w["translation"],
                    "example": w.get("example", "")
                })
        
        logging.info(f"[word_training] Generated {len(validated)} words for topic {topic_code}, level {level}")
        return validated
        
    except json.JSONDecodeError as e:
        logging.error(f"[word_training] JSON parse error: {e}")
        return []
    except Exception as e:
        logging.exception(f"[word_training] Failed to generate words: {e}")
        return []


def get_user_level(user_id: int) -> str:
    """Get user's level from database, default to B1."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT level FROM users WHERE id = %s", (user_id,))
        row = c.fetchone()
        if row and row[0]:
            return row[0]
        return "B1"
