# penpal_english_bot.py
import os
import json
import logging
import psycopg2
import psycopg2.extras
import random
import uuid
import threading
from datetime import datetime, date, timedelta
from contextlib import closing
from threading import Lock
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, LabeledPrice
from aiogram.utils import executor
from dotenv import load_dotenv
import openai

# --- Unified Event Logging (new structure) ---
USER_EVENT_SESSIONS = {}
USER_EVENT_SESSIONS_LOCK = Lock()

def get_event_session_id(user_id):
    """Get or create a session_id for a user based on 30min inactivity rule."""
    now = datetime.utcnow()
    with USER_EVENT_SESSIONS_LOCK:
        sess = USER_EVENT_SESSIONS.get(user_id, {})
        last_time = sess.get('last_event_time')
        session_id = sess.get('session_id')
        if not session_id or not last_time or (now - last_time).total_seconds() > 1800:
            session_id = uuid.uuid4()
        USER_EVENT_SESSIONS[user_id] = {'session_id': session_id, 'last_event_time': now}
        return session_id

def log_event(user_id, event_type, metadata=None, session_id=None):
    """
    Log an event to the events table (new structure).
    - user_id: int
    - event_type: str
    - metadata: dict (JSONB)
    - session_id: UUID (optional, auto-managed if not provided)
    """
    if metadata is None:
        metadata = {}
    if session_id is None:
        session_id = get_event_session_id(user_id)
    event_id = uuid.uuid4()
    # Always serialize metadata to JSON for psycopg2
    metadata_json = json.dumps(metadata)
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO events (id, user_id, event_type, metadata, session_id, created_at)
            VALUES (%s, %s, %s, %s::jsonb, %s, now())
            """,
            (str(event_id), user_id, event_type, metadata_json, str(session_id))
        )
        conn.commit()
import hashlib
from yookassa import Configuration, Payment as YooPayment
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
    logging.warning("zoneinfo not available; timezone features will be limited")
import asyncio
import copy

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# YooKassa Configuration
YOOKASSA_SHOP_ID = os.getenv("YOOKASSA_SHOP_ID")
YOOKASSA_SECRET_KEY = os.getenv("YOOKASSA_SECRET_KEY")

if YOOKASSA_SHOP_ID and YOOKASSA_SECRET_KEY:
    Configuration.account_id = YOOKASSA_SHOP_ID
    Configuration.secret_key = YOOKASSA_SECRET_KEY
    logging.info("YooKassa configured successfully")
else:
    logging.warning("YooKassa credentials not set")

# Telegram Payments provider token (from @BotFather > Payments)
PAYMENTS_PROVIDER_TOKEN = os.getenv("PAYMENTS_PROVIDER_TOKEN")
# Subscription price configuration
SUBSCRIPTION_PRICE = int(os.getenv("SUBSCRIPTION_PRICE", "299"))  # currency units
SUBSCRIPTION_CURRENCY = os.getenv("SUBSCRIPTION_CURRENCY", "RUB")
# GNews API key: prefer env var, fall back to user-provided key
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
if not GNEWS_API_KEY:
    logging.warning(
        "GNEWS_API_KEY is not set. The bot will fall back to RSS feeds. Set GNEWS_API_KEY in .env or your host environment for GNews support."
    )

bot = Bot(BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from contextlib import closing
from db import db
DB_URL = os.getenv("DATABASE_URL") or (
    f"postgresql://{os.getenv('POSTGRES_USER', 'your_db_user')}:{os.getenv('POSTGRES_PASSWORD', 'your_db_password')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'penpal_english')}"
)

# GNews categories - present these to users
TOPIC_CHOICES = [
    "World",
    "Nation",
    "Business",
    "Tech",
    "Entertainment",
    "Sports",
    "Science",
    "Health",
]

# Display names for UI (Russian)
TOPIC_DISPLAY = {
    "World": "Мир",
    "Nation": "Страна",
    "Business": "Бизнес",
    "Tech": "Технологии",
    "Entertainment": "Развлечения",
    "Sports": "Спорт",
    "Science": "Наука",
    "Health": "Здоровье",
}

# Map friendly names to GNews topic values
GNEWS_TOPIC_MAP = {
    "World": "world",
    "Nation": "nation",
    "Business": "business",
    "Tech": "technology",
    "Entertainment": "entertainment",
    "Sports": "sports",
    "Science": "science",
    "Health": "health",
}

# GNews allowed topic values (used with /top-headlines)
GNEWS_ALLOWED_TOPICS = {
    "world",
    "nation",
    "business",
    "technology",
    "entertainment",
    "sports",
    "science",
    "health",
}

# In-memory store for users who chose the "I don't know" level flow.
# Maps user_id -> set(selected_words)
USER_WORD_SELECTIONS = {}

# In-memory chat session store: user_id -> session dict
# session: {topic, tasks:[{id,text,keywords,done}], completed_count, turns}
USER_CHAT_SESSIONS = {}


def chat_topics_kb():
    rows = [
        [InlineKeyboardButton("Пройди собеседование 👔", callback_data="chat:topic:interview")],
        [InlineKeyboardButton("Закажи еду в ресторане 🍲", callback_data="chat:topic:restaurant")],
        [InlineKeyboardButton("Попроси повышение 💰", callback_data="chat:topic:raise")],
        [InlineKeyboardButton("Обсуди с турагентом поездку 🌴", callback_data="chat:topic:travel")],
        [InlineKeyboardButton("Свободное общение 🗣️", callback_data="chat:topic:free")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def make_tasks_for_topic(topic):
    """Return a list of task dicts for the given topic.
    Each task is {id, text, keywords} where keywords is a list of strings to match in user replies.
    """
    tasks = []
    if topic == "interview":
        tasks = [
            {"id": 1, "text": "Представься (имя, текущая работа или учёба)", "keywords": ["i am", "my name", "i'm", "i am a"]},
            {"id": 2, "text": "Объясни, почему хочешь эту работу", "keywords": ["because", "i want", "interested", "why i want"]},
            {"id": 3, "text": "Опиши одно своё сильное качество", "keywords": ["strength", "skill", "i can", "i am good at", "my strength"]},
            {"id": 4, "text": "Задай вопрос о компании", "keywords": ["what", "company", "position", "role", "could you tell"]},
        ]
    elif topic == "restaurant":
        tasks = [
            {"id": 1, "text": "Закажи основное блюдо и напиток", "keywords": ["i'll have", "i would like", "could i have", "i want"]},
            {"id": 2, "text": "Спроси про аллергены или диетические ограничения", "keywords": ["allerg", "gluten", "vegan", "vegetarian", "contains"]},
            {"id": 3, "text": "Попроси счёт", "keywords": ["check", "bill", "the bill", "can i pay", "pay"]},
        ]
    elif topic == "raise":
        tasks = [
            {"id": 1, "text": "Попроси повышение и расскажи о своих достижениях", "keywords": ["raise", "salary", "i have achieved", "increase", "promotion", "i deserve"]},
            {"id": 2, "text": "Предложи конкретную сумму или диапазон зарплаты", "keywords": ["salary", "per month", "per year", "amount", "rub", "$", "€"]},
            {"id": 3, "text": "Спроси о следующих шагах", "keywords": ["next steps", "when will i know", "follow up"]},
        ]
    elif topic == "travel":
        tasks = [
            {"id": 1, "text": "Спроси у турагента о цене и доступных датах", "keywords": ["price", "cost", "how much", "when", "dates"]},
            {"id": 2, "text": "Уточни, что входит в стоимость (отель, трансферы)", "keywords": ["hotel", "transfer", "included", "meals", "flight"]},
            {"id": 3, "text": "Попроси вариант дешевле или спроси о скидках", "keywords": ["discount", "cheaper", "alternative", "other options"]},
        ]
    else:  # free
        tasks = [
            {"id": 1, "text": "Поздоровайся и спроси, как дела у собеседника", "keywords": ["hello", "hi", "how are you", "how's it going"]},
            {"id": 2, "text": "Расскажи что-нибудь о своём дне", "keywords": ["today", "i went", "i saw", "my day", "i did"]},
            {"id": 3, "text": "Задай вопрос в ответ", "keywords": ["what about you", "and you", "do you", "tell me"]},
        ]
    # mark all as not done
    for t in tasks:
        t["done"] = False
    return tasks


# Persona instructions used for roleplay per topic. Keep them short; actual replies are generated by the model.
PERSONA_PROMPTS = {
    "interview": "You are a hiring manager conducting a short job interview. Your name is Sarah Mitchell. Speak as a polite, professional manager in English. Introduce yourself very briefly (1 sentence) and then ask a specific interview question to start. Be concise.",
    "restaurant": "You are a friendly restaurant waiter. Greet the customer in English very briefly (1 sentence), and ask what they would like to order. Keep it casual and helpful.",
    "raise": "You are the user's manager. Your name is Michael Thompson. Start the conversation professionally in English: introduce yourself very briefly (1 sentence), ask why the employee thinks they deserve a raise.",
    "travel": "You are a travel agent. Greet the customer in English very briefly (1 sentence), and ask about destination and travel dates.",
    "free": "You are a friendly conversation partner. Greet the user in English very briefly (1 sentence), and ask an open question to start a chat.",
}


def persona_emoji(topic_key):
    return {
        "interview": "👔",
        "restaurant": "🍲",
        "raise": "💰",
        "travel": "🌴",
        "free": "🗣️",
    }.get(topic_key, "👋")


async def send_assistant_intro_delayed(user_id, text, topic_key, delay=10):
    """Send the assistant's intro after a delay, prefixed with a small emoji and without the word 'Bot'."""
    try:
        await asyncio.sleep(delay)
        emoji = persona_emoji(topic_key)
        full_text = f"{emoji} {text}"
        # Save message so translation works
        save_msg(user_id, "assistant", full_text)
        
        kb = InlineKeyboardMarkup().add(InlineKeyboardButton("Перевести 🔁", callback_data="translate:chat"))
        # include emoji and send as a natural reply
        await bot.send_message(user_id, full_text, reply_markup=kb)
    except Exception:
        logging.exception("Failed to send delayed assistant intro")


async def check_task_completion(user_text: str, task_text: str) -> dict:
    """Ask the language model whether the user's reply completes the task.
    Returns a dict with keys: done (bool) and explanation (short Russian string).
    Falls back to a heuristic if the model call fails.
    """
    try:
        prompt = (
            "You are a friendly English teacher evaluating task completion.\n"
            "Mark done=true only if the user gave a meaningful, relevant answer.\n"
            "Requirements:\n"
            "- The answer directly addresses the task (not off-topic)\n"
            "- It's a genuine attempt (not a joke, nonsense, or single word like 'smoking')\n"
            "- It shows effort (at least one complete sentence)\n\n"
            f"Task: {task_text}\n\n"
            f"User reply: {user_text}\n\n"
            "Answer with strict JSON only, no extra text.\n"
            "Format: {\"done\": true|false, \"explanation\": \"one short sentence in Russian\"}"
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You return strict JSON only."}, {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = resp.choices[0].message["content"]
        print(f"[check_task] OpenAI response: {text}", flush=True)
        try:
            data = json.loads(text)
            # ensure keys
            result = {"done": bool(data.get("done")), "explanation": str(data.get("explanation", ""))}
            print(f"[check_task] Parsed result: {result}", flush=True)
            return result
        except Exception:
            # try to extract a JSON-like substring
            import re

            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    data = json.loads(m.group(0))
                    result = {"done": bool(data.get("done")), "explanation": str(data.get("explanation", ""))}
                    print(f"[check_task] Extracted JSON result: {result}", flush=True)
                    return result
                except Exception:
                    logging.exception("Failed to parse JSON from model task-check response")
    except Exception:
        logging.exception("check_task_completion OpenAI call failed")

    # Fallback heuristic: simple substring match of important words from task_text
    print(f"[check_task] Using fallback heuristic for task: {task_text}", flush=True)
    try:
        lowered = (user_text or "").lower()
        words = [w.strip('.,?!') for w in task_text.split() if len(w) > 3][:5]
        hits = sum(1 for w in words if w.lower() in lowered)
        print(f"[check_task] Heuristic: text_len={len(user_text)}, keywords={words}, hits={hits}", flush=True)
        # Stricter: require at least 15 chars and 2+ keyword matches
        if len(user_text) > 15 and hits >= 2:
            result = {"done": True, "explanation": "(эвристика) найдено 2+ ключевых слова"}
            print(f"[check_task] Heuristic PASSED: {result}", flush=True)
            return result
    except Exception:
        pass
    result = {"done": False, "explanation": "(эвристика) не выполнено"}
    logging.info(f"[check_task] Heuristic FAILED: {result}")
    return result





def fetch_article(url: str, min_sentences: int = 10):
    """Fetch the article page and try to extract a main image and at least min_sentences of text.
    Returns (text, image_url). Text is a string join of paragraphs; image_url may be None.
    """
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Try common article containers
        paragraphs = []
        for tag in soup.find_all(["p"]):
            txt = tag.get_text(separator=" ", strip=True)
            if txt and len(txt) > 20:
                paragraphs.append(txt)
            if len(" ".join(paragraphs).split(".")) >= min_sentences:
                break
        text = " ".join(paragraphs).strip()
        # find a likely image
        img = None
        # Prefer og:image
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            img = og["content"]
        else:
            first_img = soup.find("img")
            if first_img and first_img.get("src"):
                img = first_img["src"]
        return (text, img)
    except Exception:
        logging.exception(f"Failed to fetch article page: {url}")
        return (None, None)


def get_gnews_articles(topic=None, limit=10):
    """Query GNews API and return a list of articles with keys title, description, url, image."""
    try:
        params = {"token": GNEWS_API_KEY, "lang": "en", "max": limit}
        if topic:
            # prefer mapping
            q = GNEWS_TOPIC_MAP.get(topic, topic).lower()
        else:
            q = None

        # If q is a recognized GNews topic, call top-headlines with topic param
        if q in GNEWS_ALLOWED_TOPICS:
            params["topic"] = q
            logging.debug(f"GNews: using top-headlines topic={q}")
            resp = requests.get("https://gnews.io/api/v4/top-headlines", params=params, timeout=8)
        else:
            # fallback to search by keyword
            logging.debug(f"GNews: using search q={q}")
            search_params = {"token": GNEWS_API_KEY, "lang": "en", "max": limit}
            if q:
                search_params["q"] = q
            resp = requests.get("https://gnews.io/api/v4/search", params=search_params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for a in data.get("articles", []):
            articles.append(
                {
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "url": a.get("url"),
                    "image": a.get("image"),
                }
            )
        return articles
    except Exception:
        logging.exception("GNews API request failed")
        return []


SYSTEM_PROMPT = """You are “PenPal English,” a friendly pen-pal and English tutor.
Goals: keep a natural chat tone, adapt to the user’s level, and build confidence.
Rules:
"""

SYSTEM_PROMPT = """You are “PenPal English,” a friendly pen-pal and English tutor.
Goals: keep a natural chat tone, adapt to the user’s level, and build confidence.
Rules:
1) Be concise (≤120 words unless asked).
2) Ask one engaging follow-up.
3) After every user message, correct grammar and word-choice mistakes inline. Highlight corrections visually: Telegram doesn't support colored text, so simulate color using a colored emoji marker and HTML emphasis. For each correction show the original (if short) and the corrected form, using this format:

- 🔴 <i>original</i> → ✅ <b><u>corrected</u></b> — краткая причина на русском (1 строка)

Example:
User: "I has a dog"
Assistant: "🔴 I has a dog → ✅ <b><u>I have a dog</u></b> — ошибка согласования подлежащего и сказуемого"

Use at most 3 corrections per reply unless the user asks for full-sentence review.

IMPORTANT: Do NOT correct punctuation (missing periods, commas), capitalization, or contractions (it's vs it is). Only correct actual grammar errors (tenses, articles, prepositions, word order) and vocabulary mistakes (wrong word choice).

4) Respect user’s topics and tone.
5) When asked to explain, use A2–B2-friendly English, bullet points, and one mini exercise.
"""


def db():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.DictCursor)


def init_db():
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id SERIAL PRIMARY KEY,
            tg_username TEXT,
            level TEXT,
            topics TEXT,
            mode TEXT,
            created_at TEXT,
            last_news_url TEXT,
            timezone TEXT,
            last_daily_sent TEXT,
            last_interaction TEXT,
            goal TEXT,
            feeling TEXT,
            daily_minutes INTEGER,
            daily_articles INTEGER DEFAULT 0,
            last_article_reset DATE,
            subscription TEXT DEFAULT 'free',
            paywall_shown INTEGER DEFAULT 0,
            subscribe_click INTEGER DEFAULT 0,
            daily_sent INTEGER DEFAULT 0
        )
        """)
        # Events table for analytics
        c.execute("""
        CREATE TABLE IF NOT EXISTS events(
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            session_id UUID,
            event_name TEXT,
            event_value JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)
        # Payments table for YooKassa
        c.execute("""
        CREATE TABLE IF NOT EXISTS payments(
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            payment_id TEXT UNIQUE NOT NULL,
            amount DECIMAL(10,2) NOT NULL,
            currency TEXT DEFAULT 'RUB',
            status TEXT DEFAULT 'pending',
            plan TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            paid_at TIMESTAMPTZ,
            metadata JSONB
        )
        """)
        
        # Add streak fields if they don't exist
        c.execute("""
        ALTER TABLE users 
        ADD COLUMN IF NOT EXISTS current_streak INTEGER DEFAULT 0,
        ADD COLUMN IF NOT EXISTS last_activity_date DATE,
        ADD COLUMN IF NOT EXISTS streak_notified_today BOOLEAN DEFAULT FALSE
        """)
        
        conn.commit()

# Paywall helpers
# In-memory session store: user_id -> (session_id, last_event_time)
USER_SESSIONS = defaultdict(lambda: {'session_id': None, 'last_event_time': None})
USER_SESSIONS_LOCK = threading.Lock()

def get_session_id(user_id):
    """Get or create a session_id for a user based on 30min inactivity rule."""
    now = datetime.utcnow()
    with USER_SESSIONS_LOCK:
        sess = USER_SESSIONS[user_id]
        if not sess['session_id'] or not sess['last_event_time'] or (now - sess['last_event_time']) > timedelta(minutes=30):
            sess['session_id'] = uuid.uuid4()
        sess['last_event_time'] = now
        return sess['session_id']


# Streak helpers
def update_streak(user_id):
    """Update user's streak based on today's activity. Returns (current_streak, is_new_day)."""
    try:
        today = date.today()
        
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT current_streak, last_activity_date, streak_notified_today 
                FROM users WHERE id=%s
            """, (user_id,))
            row = c.fetchone()
            
            if not row:
                logging.error(f"[update_streak] No user found with id={user_id}")
                return (0, False)
                
            current_streak, last_activity_date, streak_notified_today = row
            
            # Default values
            if current_streak is None:
                current_streak = 0
            
            # Calculate new streak
            if last_activity_date is None:
                # First time user
                new_streak = 1
                is_new_day = True
            elif last_activity_date == today:
                # Same day - no change
                new_streak = current_streak
                is_new_day = False
            elif last_activity_date == today - timedelta(days=1):
                # Consecutive day - increment
                new_streak = current_streak + 1
                is_new_day = True
            else:
                # Gap - reset to 1
                new_streak = 1
                is_new_day = True
            
            # Update database
            c.execute("""
                UPDATE users 
                SET current_streak = %s, 
                    last_activity_date = %s, 
                    streak_notified_today = CASE WHEN %s THEN FALSE ELSE streak_notified_today END
                WHERE id=%s
            """, (new_streak, today, is_new_day, user_id))
            conn.commit()
            
            logging.error(f"[update_streak] user={user_id}, streak={new_streak}, is_new_day={is_new_day}")
            return (new_streak, is_new_day)
            
    except Exception as e:
        logging.error(f"[update_streak] ERROR for user={user_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return (0, False)


def should_show_streak_notification(user_id):
    """Check if we should show streak notification today."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT streak_notified_today FROM users WHERE id=%s", (user_id,))
        row = c.fetchone()
        return row and not row[0]


def mark_streak_notified(user_id):
    """Mark that we've shown streak notification today."""
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET streak_notified_today = TRUE WHERE id=%s", (user_id,))
        conn.commit()


def get_day_word(n):
    """Return correct Russian word form for 'day' based on number."""
    if n % 10 == 1 and n % 100 != 11:
        return "день"
    elif 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 >= 20):
        return "дня"
    else:
        return "дней"


# Daily sent helpers
def increment_daily_sent(user_id):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET daily_sent = COALESCE(daily_sent,0) + 1 WHERE id=%s", (user_id,))
        conn.commit()

def reset_daily_sent(user_id):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET daily_sent = 0 WHERE id=%s", (user_id,))
        conn.commit()
FREE_ARTICLE_LIMIT = 3
FREE_GRAMMAR_LIMIT = 3
FREE_TRUTH_LIE_LIMIT = 3
FREE_CHAT_MESSAGES_LIMIT = 10

def get_user_article_count(user_id):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT daily_articles, last_article_reset FROM users WHERE id=%s", (user_id,))
        row = c.fetchone()
    return row


def get_user_grammar_count_today(user_id):
    """Get count of grammar sets played by user today."""
    today = date.today()
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM user_game_grammar_history 
            WHERE user_id = %s AND DATE(created_at) = %s
        """, (user_id, today))
        row = c.fetchone()
    return row[0] if row else 0


def get_user_truth_lie_count_today(user_id):
    """Get count of truth/lie games played by user today."""
    today = date.today()
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM user_game_truth_lie_history 
            WHERE user_id = %s AND DATE(created_at) = %s
        """, (user_id, today))
        row = c.fetchone()
    return row[0] if row else 0


def get_user_chat_messages_count_today(user_id):
    """Get count of user messages in chat mode today (for paywall)."""
    today = date.today()
    with closing(db()) as conn:
        c = conn.cursor()
        # Count only user messages (role='user') sent today in chat mode
        # We consider messages as "chat" if they're not commands
        c.execute("""
            SELECT COUNT(*) FROM messages 
            WHERE user_id = %s 
              AND role = 'user' 
              AND created_at::date = %s
        """, (user_id, today))
        row = c.fetchone()
    return row[0] if row else 0

def increment_user_counter(user_id, field):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute(f"UPDATE users SET {field} = COALESCE({field},0) + 1 WHERE id=%s", (user_id,))
        conn.commit()

def increment_user_article_count(user_id):
    today = date.today()
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT last_article_reset FROM users WHERE id=%s", (user_id,))
        row = c.fetchone()
        # row[0] is either None or a date/datetime object
        if not row or not row[0] or row[0] != today:
            c.execute("UPDATE users SET daily_articles=1, last_article_reset=%s WHERE id=%s", (today, user_id))
        else:
            c.execute("UPDATE users SET daily_articles=daily_articles+1 WHERE id=%s", (user_id,))
        conn.commit()

def is_paid_user(user_id):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT subscription FROM users WHERE id=%s", (user_id,))
        row = c.fetchone()
    return row and row[0] == "paid"

def set_user_subscription(user_id, status: str = "paid"):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET subscription=%s WHERE id=%s", (status, user_id))
        conn.commit()


# --- YooKassa Payment Functions ---

def create_payment(user_id: int, amount: float, plan: str, description: str):
    """Create YooKassa payment and save to DB."""
    try:
        # Generate unique idempotence key
        timestamp = datetime.utcnow().isoformat()
        idempotence_key = hashlib.md5(f"{user_id}{amount}{plan}{timestamp}".encode()).hexdigest()
        
        # Get user email if available (for receipt)
        user = get_user(user_id)
        user_email = user.get("email") if user and user.get("email") else None
        
        logging.info(f"Creating payment for user {user_id}, amount {amount}, plan {plan}")
        
        payment_data = {
            "amount": {
                "value": f"{amount:.2f}",
                "currency": "RUB"
            },
            "confirmation": {
                "type": "redirect",
                "return_url": "https://t.me/MaxEnglishPracticeBot"
            },
            "capture": True,
            "description": description,
            "metadata": {
                "user_id": str(user_id),
                "plan": plan
            },
            "receipt": {
                "customer": {
                    "email": user_email if user_email else "customer@example.com"
                },
                "items": [
                    {
                        "description": description,
                        "quantity": "1.00",
                        "amount": {
                            "value": f"{amount:.2f}",
                            "currency": "RUB"
                        },
                        "vat_code": 1,
                        "payment_mode": "full_payment",
                        "payment_subject": "service"
                    }
                ]
            }
        }
        
        logging.info(f"Payment data prepared: {json.dumps(payment_data, indent=2)}")
        
        payment = YooPayment.create(payment_data, idempotence_key)
        
        logging.info(f"YooKassa payment created: {payment.id}, status: {payment.status}")
        
        # Save to DB
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO payments (user_id, payment_id, amount, currency, status, plan, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, payment.id, amount, "RUB", payment.status, plan, json.dumps(dict(payment.metadata))))
            conn.commit()
            
        logging.info(f"Payment saved to DB: {payment.id} for user {user_id}, amount {amount}")
        return payment
        
    except Exception as e:
        logging.exception(f"Failed to create payment: {e}")
        return None

def check_payment_status(payment_id: str):
    """Check payment status from YooKassa."""
    try:
        payment = YooPayment.find_one(payment_id)
        
        # Update DB
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE payments 
                SET status = %s, 
                    paid_at = CASE WHEN %s = 'succeeded' THEN now() ELSE paid_at END
                WHERE payment_id = %s
            """, (payment.status, payment.status, payment_id))
            conn.commit()
            
        logging.info(f"Payment {payment_id} status: {payment.status}")
        return payment
        
    except Exception as e:
        logging.exception(f"Failed to check payment: {e}")
        return None


def activate_subscription(user_id: int, plan: str):
    """Activate subscription for user after successful payment."""
    try:
        set_user_subscription(user_id, "paid")
        log_event(user_id, "subscription_activated", {"plan": plan})
        logging.info(f"Subscription activated for user {user_id}, plan: {plan}")
        return True
    except Exception as e:
        logging.exception(f"Failed to activate subscription: {e}")
        return False
    c.execute("""
    CREATE TABLE IF NOT EXISTS messages(
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        role TEXT,
        content TEXT,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS vocab(
        user_id INTEGER,
        phrase TEXT,
        example TEXT,
        added_at TEXT,
        bin INTEGER DEFAULT 1
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS news_cache(
        id SERIAL PRIMARY KEY,
        url TEXT,
        title TEXT,
        summary TEXT,
        published_at TEXT,
        questions TEXT
    )
    """)
    conn.commit()


def save_user(user_id, username):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO users (id, tg_username, created_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (user_id, username, datetime.utcnow().isoformat()),
        )
        conn.commit()


def set_user_level(user_id, level):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET level=%s WHERE id=%s", (level, user_id))
        conn.commit()


def set_user_topics(user_id, topics):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET topics=%s WHERE id=%s", (",".join(topics), user_id))
        conn.commit()


def get_user(user_id):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT id, tg_username, level, topics, mode, last_news_url, timezone, last_daily_sent, last_interaction FROM users WHERE id=%s", (user_id,))
        row = c.fetchone()
    return dict(row) if row else None


def set_user_mode(user_id, mode):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET mode=%s WHERE id=%s", (mode, user_id))
        conn.commit()


def set_user_last_news(user_id, url):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET last_news_url=%s WHERE id=%s", (url, user_id))
        conn.commit()


def set_user_timezone(user_id, tz_name):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET timezone=%s WHERE id=%s", (tz_name, user_id))
        conn.commit()

def set_user_goal(user_id, goal):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET goal=%s WHERE id=%s", (goal, user_id))
        conn.commit()

def set_user_feeling(user_id, feeling):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET feeling=%s WHERE id=%s", (feeling, user_id))
        conn.commit()

def set_user_daily_minutes(user_id, minutes):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET daily_minutes=%s WHERE id=%s", (minutes, user_id))
        conn.commit()


def get_all_users_for_daily():
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT id, tg_username, last_daily_sent, last_interaction FROM users")
        rows = c.fetchall()
    return [dict(r) for r in rows]




def save_msg(user_id, role, content):
    with closing(db()) as conn:
        now = datetime.utcnow().isoformat()
        today = date.today()
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages(user_id, role, content, created_at) VALUES(%s, %s, %s, %s)",
            (user_id, role, content, now),
        )
        # Update last_interaction AND streak in one query (only for user messages)
        if role == "user":
            try:
                c.execute("""
                    UPDATE users 
                    SET last_interaction = %s,
                        streak_count = CASE 
                            WHEN last_active_date IS NULL THEN 1
                            WHEN last_active_date = %s THEN COALESCE(streak_count, 0)
                            WHEN last_active_date = %s - INTERVAL '1 day' THEN COALESCE(streak_count, 0) + 1
                            ELSE 1
                        END,
                        max_streak = GREATEST(
                            COALESCE(max_streak, 0),
                            CASE 
                                WHEN last_active_date IS NULL THEN 1
                                WHEN last_active_date = %s THEN COALESCE(streak_count, 0)
                                WHEN last_active_date = %s - INTERVAL '1 day' THEN COALESCE(streak_count, 0) + 1
                                ELSE 1
                            END
                        ),
                        last_active_date = %s
                    WHERE id = %s
                """, (now, today, today, today, today, today, user_id))
            except Exception:
                logging.exception("Failed to update last_interaction and streak")
        else:
            # For assistant messages, just update last_interaction
            try:
                c.execute("UPDATE users SET last_interaction=%s WHERE id=%s", (now, user_id))
            except Exception:
                logging.exception("Failed to update last_interaction")
        # keep last 30
        c.execute(
            "DELETE FROM messages WHERE id NOT IN (SELECT id FROM messages WHERE user_id=%s ORDER BY id DESC LIMIT 30) AND user_id=%s",
            (user_id, user_id),
        )
        conn.commit()


def add_vocab(user_id, items):
    with closing(db()) as conn:
        c = conn.cursor()
        for it in items:
            c.execute(
                "INSERT INTO vocab(user_id, phrase, example, added_at) VALUES(%s, %s, %s, %s)",
                (
                    user_id,
                    it.get("phrase", ""),
                    it.get("example", ""),
                    datetime.utcnow().isoformat(),
                ),
            )
        conn.commit()


if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY is not set; OpenAI calls will fail until you set it in .env")


async def gpt_chat(messages):
    # Use small, cheap model; swap to 4o/omni later
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
        )
        return resp.choices[0].message["content"]
    except Exception:
        logging.exception("gpt_chat OpenAI call failed")
        # graceful fallback so handlers don't crash
    return "Sorry — my language engine is unavailable right now. Try again later or use /news to get a short article. 🤖"


async def gpt_structured_news(level, topics, article_title, article_text, url):
    prompt = f"""Create a 2–3 sentence summary (CEFR {level}) for this article, then 3 casual discussion questions that are directly and specifically related to the article's main points, themes, or consequences.
User interests: {topics}
Title: {article_title}
Article: {article_text[:3000]}
Return strict JSON with keys: summary (string), questions (array of 3 short question strings), vocab (array of 0-2 objects with 'phrase' and 'example').
Requirements: each question must reference the article content (avoid generic prompts like “What surprised you most?” unless tied to text). Keep everything concise and on-topic."""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        text = resp.choices[0].message["content"]
        logging.debug(f"GPT raw news response: {text}")
        try:
            data = json.loads(text)
        except Exception:
            logging.exception("Failed to parse GPT JSON response; using fallback data")
            # deterministic fallback: short extract from article_text and generic questions
            summary = (article_text or article_title)[:300].strip()
            if len(summary) > 280:
                summary = summary.rsplit(" ", 1)[0] + "..."
            data = {
                "summary": summary or article_title,
                "questions": [
                    "What surprised you most from this article?",
                    "How could this news affect people like you?",
                    "Do you agree with the main idea? Why or why not?",
                ],
                "vocab": [],
            }
    except Exception:
        logging.exception("OpenAI request failed; using fallback news data")
        # If OpenAI fails, produce a simple summary from the article text and three friendly questions.
        summary = (article_text or article_title)[:300].strip()
        if len(summary) > 280:
            summary = summary.rsplit(" ", 1)[0] + "..."
        data = {
            "summary": summary or article_title,
            "questions": [
                "What surprised you most from this article?",
                "How could this news affect people like you?",
                "Do you agree with the main idea? Why or why not?",
            ],
            "vocab": [],
        }
    return data


def topic_keyboard(selected=None):
    selected = set(selected or [])
    rows = []
    row = []
    for t in TOPIC_CHOICES:
        mark = "✅ " if t in selected else ""
        label = TOPIC_DISPLAY.get(t, t)
        row.append(InlineKeyboardButton(f"{mark}{label}", callback_data=f"topic:{t}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("Готово ✔️", callback_data="topic:done")])
    # Кнопка 'Меню' убрана для выбора темы
    return InlineKeyboardMarkup(inline_keyboard=rows)


def news_topics_reselect_keyboard():
    """Keyboard entry point for changing news topics from commands.

    This lets the user explicitly return to news topic selection instead of
    being forced to reselect every time they click "Обсудить статью".
    """
    rows = [
        [InlineKeyboardButton("Сменить темы новостей 📰", callback_data="news:topics")],
        [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def level_keyboard():
    levels = ["A2", "B1", "B2", "C1"]
    top_row = [InlineKeyboardButton(l, callback_data=f"level:{l}") for l in levels]
    unknown_row = [InlineKeyboardButton("Я не знаю", callback_data="level:unknown")]
    return InlineKeyboardMarkup(inline_keyboard=[top_row, unknown_row])


def render_word_selection_kb(user_id):
    """Render an inline keyboard showing the six words and checkmarks for selected ones.
    Words: River, Rarely, Whale, Ambiguous, Gossip, Knowledge
    Buttons toggle selection and there's a final Done button.
    """
    words = ["River", "Rarely", "Whale", "Ambiguous", "Gossip", "Knowledge"]
    sel = USER_WORD_SELECTIONS.get(user_id, set())
    kb_rows = []
    for i in range(0, len(words), 2):
        row = []
        for w in words[i : i + 2]:
            mark = "✅ " if w in sel else ""
            row.append(InlineKeyboardButton(f"{mark}{w}", callback_data=f"word:toggle:{w}"))
        kb_rows.append(row)
    kb_rows.append([InlineKeyboardButton("Готово ✔️", callback_data="word:done")])
    # Кнопка 'Меню' убрана для определения уровня по словам
    return InlineKeyboardMarkup(inline_keyboard=kb_rows)


def mode_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton("Обсудить новость 📰", callback_data="mode:news")],
            [InlineKeyboardButton("Разговорная практика 💬", callback_data="mode:chat")],
            [InlineKeyboardButton("Тренировать слова 🧠", callback_data="mode:train_words")],
            [InlineKeyboardButton("Играть 🎮", callback_data="mode:games")],
            [InlineKeyboardButton("👤 Профиль", callback_data="mode:profile")],
        ]
    )

def games_selection_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton("2 правды 1 ложь 🤥", callback_data="game:truth_lie:start")],
            [InlineKeyboardButton("Исправь грамматику 🎯", callback_data="game:grammar:start")],
            [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")],
        ]
    )


def news_topics_keyboard(existing_topics=None):
    """Keyboard for (re)selecting news topics.

    existing_topics: optional list of topic codes already saved for the user.
    """
    return topic_keyboard(existing_topics or [])

# Onboarding keyboards
def onboarding_goal_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Работа / карьера 💼", callback_data="onboard:goal:career")],
        [InlineKeyboardButton("Путешествия ✈️", callback_data="onboard:goal:travel")],
        [InlineKeyboardButton("Переезд 🌍", callback_data="onboard:goal:relocation")],
        [InlineKeyboardButton("Экзамен / сертификат 🎓", callback_data="onboard:goal:exam")],
        [InlineKeyboardButton("Свободное общение 🗣️", callback_data="onboard:goal:conversation")],
        [InlineKeyboardButton("Другое ✨", callback_data="onboard:goal:other")],
    # Кнопка 'Меню' убрана из онбординга
    ])

def onboarding_interest_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Обсуждение свежих новостей 📰", callback_data="onboard:interest:news")],
        [InlineKeyboardButton("Разговорная практика 🗣️", callback_data="onboard:interest:ai")],
        [InlineKeyboardButton("Тренировка грамматики ✍️", callback_data="onboard:interest:grammar")],
        [InlineKeyboardButton("Всё интересно! 🌟", callback_data="onboard:interest:all")],
    # Кнопка 'Меню' убрана из онбординга
    ])

def onboarding_minutes_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("5 мин ⏱", callback_data="onboard:minutes:5"), InlineKeyboardButton("10 мин 🔟", callback_data="onboard:minutes:10")],
        [InlineKeyboardButton("15 мин 🧠", callback_data="onboard:minutes:15"), InlineKeyboardButton("20+ мин 🚀", callback_data="onboard:minutes:20")],
        [InlineKeyboardButton("Не знаю 🤷", callback_data="onboard:minutes:unknown")],
    # Кнопка 'Меню' убрана из онбординга
    ])



async def send_news(user_id):
    try:
        user = get_user(user_id)
        if not user:
            logging.warning(f"send_news called for unknown user_id={user_id}")
            return
        level = user.get("level") or "B1"
        raw_topics = (user.get("topics") or "World").split(",")
        selected_topics = [t.strip() for t in raw_topics if t and t.strip()]
        if not selected_topics:
            selected_topics = ["World"]
        logging.info(f"Fetching news for user={user_id} level={level} topics={selected_topics}")

        # Try GNews (topic-aware) first. Randomize order so different selected topics get used.
        articles = []
        topics_shuffled = selected_topics[:]
        random.shuffle(topics_shuffled)
        for t in topics_shuffled:
            try:
                articles = get_gnews_articles(topic=t, limit=10)
            except Exception:
                logging.exception("GNews lookup failed for topic %s", t)
                articles = []
            if articles:
                break

        if articles:
            # avoid repeating the last article shown to this user
            last_url = user.get("last_news_url")
            candidates = [a for a in articles if a.get("url") != last_url]
            if not candidates:
                # all articles match last_url; fall back to full list
                candidates = articles
            item = random.choice(candidates)
            title = item.get("title") or "News"
            url = item.get("url") or ""
            desc = item.get("description") or title
            image_candidate = item.get("image")
        else:
            # No articles from GNews: inform the user (we no longer use RSS fallback)
            logging.warning(f"No GNews articles found for topics {selected_topics}")
            await bot.send_message(
                user_id,
                "Извини — сейчас не удалось найти подходящие статьи по твоим темам. Попробуй /topics или /news позже. 🤖",
            )
            return

        # Try to fetch longer article text and an image when available
        article_text, fetched_img = (None, None)
        if url:
            article_text, fetched_img = fetch_article(url, min_sentences=10)
        image_url = fetched_img or image_candidate

        # Prepare content using article_text when possible
        article_body = article_text or desc
        logging.debug(f"Selected item title={title} url={url}")

        try:
            data = await gpt_structured_news(level, selected_topics, title, article_body, url)
        except Exception:
            logging.exception("gpt_structured_news failed, using fallback data")
            data = {
                "summary": title,
                "questions": [
                    "What surprised you most?",
                    "How could this affect daily life?",
                    "Do you agree?",
                ],
                "vocab": [],
            }

        # store vocab
        try:
            add_vocab(user_id, data.get("vocab", []))
        except Exception:
            logging.exception("Failed to add vocab; continuing")

        # Do not show questions immediately; they will be shown when the user presses "Completed".
        voc = data.get("vocab", [])
        voc_txt = (
            "\n".join([f"🔹 <b>{v['phrase']}</b> — <i>{v['example']}</i>" for v in voc])
            if voc
            else ""
        )
        text = f"<b>{title}</b>\n\n{data.get('summary', '')}"
        if voc_txt:
            text += f"\n\n<b>Useful phrases:</b>\n{voc_txt}"

        # Save the news and questions to cache so we can support translate/completed flows
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO news_cache(url, title, summary, published_at, questions) VALUES(%s, %s, %s, %s, %s) RETURNING id",
                (
                    url,
                    title,
                    data.get("summary", ""),
                    datetime.utcnow().isoformat(),
                    json.dumps(data.get("questions", [])),
                ),
            )
            cache_id = c.fetchone()[0]
            conn.commit()

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        "Перевести 🔁", callback_data=f"news:translate:{cache_id}"
                    ),
                    InlineKeyboardButton("Прочитал(а) ✅", callback_data=f"news:done:{cache_id}"),
                ],
                [InlineKeyboardButton("Поменять статью 🔁", callback_data="news:more")],
                [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")],
            ]
        )

        logging.info(f"Sending news (image={bool(image_url)}) to user {user_id}")
        # If we have an image and a longer article, send as photo with caption (Telegram caption limit ~1024 chars)
        if image_url and article_text:
            caption = (text[:900] + "...") if len(text) > 900 else text
            try:
                await bot.send_photo(user_id, image_url, caption=caption, reply_markup=kb)
                return
            except Exception:
                logging.exception("Failed to send photo; falling back to text message")

        # Default: send as text
        logging.debug(f"Sending news text length={len(text)}")
        await bot.send_message(user_id, text, reply_markup=kb, disable_web_page_preview=True)
    except Exception:
        logging.exception("send_news failed")
        # Inform the user rather than failing silently
        try:
            await bot.send_message(
                user_id, "Извини — не удалось получить новости. Попробуй /news позже. 🙏"
            )
        except Exception:
            logging.exception("Failed to send error message to user")


@dp.message_handler(commands=["start"])
async def start(m: types.Message):
    # Any previous roleplay/chat topic session should be cleared on full restart
    USER_CHAT_SESSIONS.pop(m.from_user.id, None)
    save_user(m.from_user.id, m.from_user.username or "")
    save_msg(m.from_user.id, "user", "/start")
    log_event(m.from_user.id, "onboarding_started", {})
    
    # Send welcome sticker
    try:
        sticker_file_id = "CAACAgIAAxkBAAILm2kq_J_TOkHArja22n1yyA1Z5wiNAAIYgwACMLFZSQy5VwJHiV9nNgQ"
        await m.answer_sticker(sticker_file_id)
        # Wait 1.5 seconds before sending onboarding message
        await asyncio.sleep(1.5)
    except Exception:
        logging.exception("Failed to send welcome sticker")
    
    # Reset topics and onboarding fields for this user
    try:
        set_user_topics(m.from_user.id, [])
        set_user_mode(m.from_user.id, None)
        set_user_goal(m.from_user.id, None)
        set_user_feeling(m.from_user.id, None)
        set_user_daily_minutes(m.from_user.id, None)
    except Exception:
        logging.exception("Failed to reset user topics/onboarding on /start")
    try:
        await m.answer(
            "Супер, ты на шаг ближе к цели 🎯\n\nПеред тем как начнём, расскажи немного о себе:\n\n<b>Какая твоя главная цель в изучении английского?</b>",
            reply_markup=onboarding_goal_kb(),
        )
    except Exception:
        logging.exception("Failed to send onboarding goal; falling back to safe message")
        try:
            await m.answer("Давай начнём! Выбери свою цель:", reply_markup=onboarding_goal_kb())
        except Exception:
            logging.exception("Fallback onboarding goal also failed")

@dp.callback_query_handler(lambda c: c.data.startswith("onboard:goal:"))
async def onboard_goal(c: types.CallbackQuery):
    save_msg(c.from_user.id, "user", c.data)
    log_event(c.from_user.id, "onboarding_goal_selected", {"goal": c.data.split(":")[2]})
    goal = c.data.split(":")[2]
    set_user_goal(c.from_user.id, goal)
    await c.answer()
    await c.message.edit_text(
        "Отлично, я с радостью помогу тебе🙌\n\nКакой формат тебе сейчас больше всего подходит?",
        reply_markup=onboarding_interest_kb()
    )

@dp.callback_query_handler(lambda c: c.data.startswith("onboard:interest:"))
async def onboard_interest(c: types.CallbackQuery):
    save_msg(c.from_user.id, "user", c.data)
    log_event(c.from_user.id, "onboarding_topic_selected", {"interest": c.data.split(":")[2]})
    interest = c.data.split(":")[2]
    set_user_feeling(c.from_user.id, interest)
    await c.answer()
    await c.message.edit_text(
        "Сколько времени в день ты готов уделять английскому? ⏳",
        reply_markup=onboarding_minutes_kb()
    )

@dp.callback_query_handler(lambda c: c.data.startswith("onboard:minutes:"))
async def onboard_minutes(c: types.CallbackQuery):
    save_msg(c.from_user.id, "user", c.data)
    # User selected how many minutes per day they can study
    log_event(c.from_user.id, "onboarding_minutes_selected", {"minutes": c.data.split(":")[2]})
    minutes = c.data.split(":")[2]
    set_user_daily_minutes(c.from_user.id, minutes if minutes != "unknown" else None)
    # Onboarding completed
    log_event(c.from_user.id, "onboarding_completed", {})
    await c.answer()
    await c.message.edit_text(
        "Спасибо! Теперь выбери свой уровень английского:", reply_markup=level_keyboard()
    )


@dp.callback_query_handler(lambda c: c.data.startswith("level:"))
async def choose_level(c: types.CallbackQuery):
    log_event(c.from_user.id, "onboarding_level_selected", {"level": c.data.split(":")[1]})
    save_msg(c.from_user.id, "user", c.data)
    level = c.data.split(":")[1]
    user_id = c.from_user.id
    # If user chose unknown, start the quick word-selection flow
    if level == "unknown":
        # initialize selection set
        USER_WORD_SELECTIONS[user_id] = set()
        await c.answer()
        await c.message.edit_text(
            "Не беда! Сейчас мы с тобой вместе его определим 🙌\nВыбери все слова, которые знаешь:",
            reply_markup=render_word_selection_kb(user_id),
        )
        return

    set_user_level(user_id, level)
    # reset mode so the user can pick again
    set_user_mode(user_id, None)
    await c.answer()
    await c.message.edit_text(
        f"Отлично! Уровень установлен: <b>{level}</b> 🎯\n\nС чего начнем?",
        reply_markup=mode_keyboard(),
    )


@dp.callback_query_handler(lambda c: c.data.startswith("mode:") and c.data not in ["mode:profile", "mode:train_words"])
async def choose_mode(c: types.CallbackQuery):
    log_event(c.from_user.id, "mode_selected", {"mode": c.data.split(":")[1]})
    save_msg(c.from_user.id, "user", c.data)
    user_id = c.from_user.id
    mode = c.data.split(":")[1]
    if mode not in {"news", "chat", "games"}:
        await c.answer("Неизвестный режим.", show_alert=True)
        return
    # Switching between news/chat should drop any previous chat topic state
    USER_CHAT_SESSIONS.pop(user_id, None)
    set_user_mode(user_id, mode)
    await c.answer()

    if mode == "games":
        log_event(user_id, "games_menu_opened", {})
        await c.message.edit_text("Во что сыграем?", reply_markup=games_selection_keyboard())
        return

    if mode == "news":
        user = get_user(user_id)
        existing = []
        if user and user.get("topics"):
            existing = [t.strip() for t in (user.get("topics") or "").split(",") if t.strip()]

        # If topics already chosen before, don't force selection every time;
        # just bring a new article based on saved interests.
        if existing:
            # Check article limit BEFORE sending news
            paid = is_paid_user(user_id)
            
            if not paid:
                count_row = get_user_article_count(user_id)
                today = date.today()
                daily_articles = count_row[0] if count_row else 0
                last_reset = count_row[1] if count_row else None
                
                logging.info(f"[mode:news] user={user_id} paid={paid} daily_articles={daily_articles} last_reset={last_reset} today={today} limit={FREE_ARTICLE_LIMIT}")
                
                if last_reset == today and daily_articles >= FREE_ARTICLE_LIMIT:
                    increment_user_counter(user_id, "paywall_shown")
                    log_event(user_id, "paywall_shown", {"reason": "article_limit", "count": daily_articles})
                    logging.info(f"[mode:news] PAYWALL TRIGGERED for user={user_id}")
                    kb = InlineKeyboardMarkup(inline_keyboard=[
                        [InlineKeyboardButton("Приобрести доступ 💎", callback_data="profile_buy_unlimited")],
                        [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
                    ])
                    await c.message.edit_text(
                        "🔒 Ты прочитал 3 статьи сегодня!\n\n"
                        "Чтобы продолжить без ограничений, приобрети безлимитный доступ 💎",
                        reply_markup=kb
                    )
                    return
            
            # Increment counter and send news
            increment_user_article_count(user_id)
            logging.info(f"[mode:news] Article count incremented for user={user_id}")
            
            await c.message.edit_text(
                "Отлично! Я подберу статью по твоим темам. Вот новость 📰:",
            )
            await send_news(user_id)
        else:
            await c.message.edit_text(
                "Выбери темы, которые тебе нравятся:", reply_markup=topic_keyboard(existing)
            )
    else:
        # Present chat topic choices when user selects free chat
        await c.message.edit_text(
            "Отлично! Выбери тему для свободного разговора:", reply_markup=chat_topics_kb()
        )


@dp.callback_query_handler(lambda c: c.data.startswith("chat:topic:"))
async def choose_chat_topic(c: types.CallbackQuery):
    log_event(c.from_user.id, "topic_session_started", {"topic": c.data.split(":")[2]})
    save_msg(c.from_user.id, "user", c.data)
    user_id = c.from_user.id
    parts = c.data.split(":")
    topic_key = parts[2]
    # Map topic_key to readable name
    names = {
        "interview": "Пройди собеседование 👔",
        "restaurant": "Закажи еду в ресторане 🍲",
        "raise": "Попроси повышение 💰",
        "travel": "Обсуди с турагентом поездку 🌴",
        "free": "Свободное общение 🗣️",
    }
    topic_name = names.get(topic_key, topic_key)
    # start a session with 3 required tasks to complete
    tasks = make_tasks_for_topic(topic_key)
    # Take only first 3 tasks
    tasks = tasks[:3]
    # we will require 3 tasks to be completed
    USER_CHAT_SESSIONS[user_id] = {
        "type": "roleplay",
        "topic": topic_key,
        "tasks": tasks,
        "completed_count": 0,
        "turns": 0,
    }
    # show rules and first tasks
    await c.answer()
    intro = f"Тема: {topic_name}\n\nПравила: Выполни 3 задания или скажи bye 👋, чтобы завершить диалог."
    tasks_list = "\n".join([f"{t['id']}) {t['text']}" for t in tasks])
    # Ask the language model to play the persona and produce a short intro (in English)
    persona = PERSONA_PROMPTS.get(topic_key, PERSONA_PROMPTS.get("free"))
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": persona}, {"role": "user", "content": "Introduce yourself briefly and ask the first question to the user."}],
            temperature=0.7,
        )
        assistant_intro = resp.choices[0].message["content"]
    except Exception:
        logging.exception("Failed to generate persona intro; using fallback intro")
        assistant_intro = "Hello — let's start. Please answer the question." 

    # store assistant intro in session for context
    USER_CHAT_SESSIONS[user_id]["assistant_intro"] = assistant_intro

    # Send first message: topic, rules and tasks (tasks are already in Russian)
    await c.message.edit_text(intro + "\n\nЗадания:\n" + tasks_list)
    # Send assistant intro as a separate message after 5 seconds without the word 'Bot' and with emoji
    try:
        asyncio.create_task(send_assistant_intro_delayed(c.from_user.id, assistant_intro, topic_key, delay=5))
    except Exception:
        logging.exception("Failed to schedule assistant intro")


@dp.callback_query_handler(lambda c: c.data.startswith("topic:"))
async def choose_topics(c: types.CallbackQuery):
    save_msg(c.from_user.id, "user", c.data)
    # Selecting news topics means we are no longer in a roleplay topic session
    USER_CHAT_SESSIONS.pop(c.from_user.id, None)
    user = get_user(c.from_user.id)
    if not user:
        await c.answer("Не удалось найти профиль. Попробуй /start.", show_alert=True)
        return
    if user.get("mode") != "news":
        await c.answer(
            "Сначала выбери режим «Обсудить статью» после выбора уровня.", show_alert=True
        )
        return
    selected = [t.strip() for t in (user.get("topics") or "").split(",") if t.strip()]
    val = c.data.split(":")[1]
    if val == "done":
        if not selected:
            await c.answer("Выбери хотя бы одну тему 🙂", show_alert=True)
            return
        
        # Check article limit BEFORE sending news
        user_id = c.from_user.id
        paid = is_paid_user(user_id)
        
        if not paid:
            count_row = get_user_article_count(user_id)
            today = date.today()
            daily_articles = count_row[0] if count_row else 0
            last_reset = count_row[1] if count_row else None
            
            logging.info(f"[topic:done] user={user_id} paid={paid} daily_articles={daily_articles} last_reset={last_reset} today={today} limit={FREE_ARTICLE_LIMIT}")
            
            if last_reset == today and daily_articles >= FREE_ARTICLE_LIMIT:
                increment_user_counter(user_id, "paywall_shown")
                log_event(user_id, "paywall_shown", {"reason": "article_limit", "count": daily_articles})
                logging.info(f"[topic:done] PAYWALL TRIGGERED for user={user_id}")
                kb = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton("Приобрести доступ 💎", callback_data="profile_buy_unlimited")],
                    [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
                ])
                await c.message.edit_text(
                    "🔒 Ты прочитал 3 статьи сегодня!\n\n"
                    "Чтобы продолжить без ограничений, приобрети безлимитный доступ 💎",
                    reply_markup=kb
                )
                return
        
        # Increment counter and send news
        increment_user_article_count(user_id)
        logging.info(f"[topic:done] Article count incremented for user={user_id}")
        
        await c.message.edit_text(
            "Отлично! Я принесу материал для обсуждения. Вот новость 📰:\n\n"
            "Темы новостей всегда можно изменить командой /newstopics."
        )
        await send_news(c.from_user.id)
        return
    if val in selected:
        selected = [t for t in selected if t != val]
    else:
        selected.append(val)
    set_user_topics(c.from_user.id, selected)
    await c.message.edit_reply_markup(reply_markup=topic_keyboard(selected))


@dp.callback_query_handler(lambda c: c.data == "news:topics")
async def reselect_news_topics(c: types.CallbackQuery):
    """Explicit entry point to change saved news topics.

    Opens the same topic selection keyboard used during onboarding when the
    user first chose "Обсудить статью".
    """
    save_msg(c.from_user.id, "user", c.data)
    user = get_user(c.from_user.id)
    if not user:
        await c.answer("Не удалось найти профиль. Попробуй /start.", show_alert=True)
        return
    # Ensure we're in news mode so that subsequent flows behave correctly
    set_user_mode(c.from_user.id, "news")
    existing = []
    if user.get("topics"):
        existing = [t.strip() for t in (user.get("topics") or "").split(",") if t.strip()]
    await c.answer()
    await c.message.edit_text(
        "Выбери темы, которые тебе нравятся (эти темы всегда можно изменить командой /newstopics):",
        reply_markup=topic_keyboard(existing),
    )


@dp.message_handler(commands=["newstopics"])
async def cmd_newstopics(m: types.Message):
    """Text command to (re)select news topics at any time."""
    save_msg(m.from_user.id, "user", "/newstopics")
    user = get_user(m.from_user.id)
    if not user:
        await m.answer("Не удалось найти профиль. Попробуй /start.")
        return
    # Переключаем режим на news и показываем текущий выбор тем (если есть)
    set_user_mode(m.from_user.id, "news")
    existing = []
    if user.get("topics"):
        existing = [t.strip() for t in (user.get("topics") or "").split(",") if t.strip()]
    await m.answer(
        "Выбери темы, которые тебе нравятся (эти темы всегда можно изменить командой /newstopics):",
        reply_markup=topic_keyboard(existing),
    )


@dp.callback_query_handler(lambda c: c.data.startswith("word:toggle:"))
async def toggle_word(c: types.CallbackQuery):
    save_msg(c.from_user.id, "user", c.data)
    # Toggle the selected word for the user
    parts = c.data.split(":")
    word = parts[2]
    uid = c.from_user.id
    sel = USER_WORD_SELECTIONS.get(uid, set())
    if word in sel:
        sel.remove(word)
    else:
        sel.add(word)
    USER_WORD_SELECTIONS[uid] = sel
    await c.answer()
    # update the keyboard to reflect new selection
    await c.message.edit_reply_markup(reply_markup=render_word_selection_kb(uid))


@dp.callback_query_handler(lambda c: c.data == "word:done")
async def finalize_word_selection(c: types.CallbackQuery):
    save_msg(c.from_user.id, "user", c.data)
    uid = c.from_user.id
    sel = USER_WORD_SELECTIONS.get(uid, set())
    count = len(sel)
    # Determine level based on count
    if count <= 2:
        level = "A2"
    elif 3 <= count <= 4:
        level = "B1"
    elif count == 5:
        level = "B2"
    else:
        # 6 or more
        level = "C1"

    set_user_level(uid, level)
    # clean up selection
    USER_WORD_SELECTIONS.pop(uid, None)
    set_user_mode(uid, None)
    await c.answer()
    await c.message.edit_text(
        f"Готово — по твоему выбору ({count} слов) уровень определён как <b>{level}</b>.\n\nС чего начнем?",
        reply_markup=mode_keyboard(),
    )


@dp.callback_query_handler(lambda c: c.data == "news:continue")
async def news_continue_dialog(c: types.CallbackQuery):
    """User chose to continue discussing the article after completing 3 questions."""
    user_id = c.from_user.id
    await c.answer("Продолжаем! Просто пиши свои мысли 💬")
    # Session stays active, user can keep chatting
    await c.message.edit_text(
        "Продолжаем обсуждение! Напиши, что ты думаешь о статье, или задай вопрос. 💬\n\n"
        "Напиши <b>bye</b> или <b>пока</b>, чтобы вернуться в меню.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
        ])
    )


@dp.callback_query_handler(lambda c: c.data.startswith("news:more"))
async def more_news(c: types.CallbackQuery):
    user_id = c.from_user.id
    log_event(user_id, "news_requested", {})
    save_msg(user_id, "user", c.data)
    
    # Check limit BEFORE incrementing
    paid = is_paid_user(user_id)
    
    if not paid:
        count_row = get_user_article_count(user_id)
        today = date.today()
        daily_articles = count_row[0] if count_row else 0
        last_reset = count_row[1] if count_row else None
        
        logging.info(f"[more_news] user={user_id} paid={paid} daily_articles={daily_articles} last_reset={last_reset} today={today} limit={FREE_ARTICLE_LIMIT}")
        
        # If last reset was today, use current count; otherwise it's a new day (count will reset)
        if last_reset == today and daily_articles >= FREE_ARTICLE_LIMIT:
            # Increment paywall_shown counter
            increment_user_counter(user_id, "paywall_shown")
            log_event(user_id, "paywall_shown", {"reason": "article_limit", "count": daily_articles})
            logging.info(f"[more_news] PAYWALL TRIGGERED for user={user_id}")
            try:
                await c.answer("Вы достигли лимита бесплатных статей на сегодня.", show_alert=True)
            except Exception:
                pass
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton("Приобрести доступ 💎", callback_data="profile_buy_unlimited")],
                [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
            ])
            # Delete the message with the article (might have image) and send new text message
            try:
                await c.message.delete()
            except Exception:
                pass
            await bot.send_message(
                user_id,
                "🔒 Ты прочитал 3 статьи сегодня!\n\n"
                "Чтобы продолжить без ограничений, приобрети безлимитный доступ 💎",
                reply_markup=kb
            )
            return

    # Now increment (this will also reset if new day)
    increment_user_article_count(user_id)
    logging.info(f"[more_news] Article count incremented for user={user_id}")
    
    await c.answer("Загружаю другую статью… ⏳")
    await send_news(user_id)


@dp.callback_query_handler(lambda c: c.data.startswith("ans:"))
async def answer_hint(c: types.CallbackQuery):
    save_msg(c.from_user.id, "user", c.data)
    idx = int(c.data.split(":")[1])
    prompts = [
        "Напиши свой ответ на вопрос 1 👇",
        "Что ты думаешь о вопросе 2? 👇",
        "Твои мысли по вопросу 3? 👇",
    ]
    await bot.send_message(c.from_user.id, prompts[idx])


@dp.callback_query_handler(lambda c: c.data.startswith("news:translate:"))
async def news_translate(c: types.CallbackQuery):
    parts = c.data.split(":")
    # User requested translation of a news article
    log_event(c.from_user.id, "translation_requested", {"cache_id": int(parts[2])})
    save_msg(c.from_user.id, "user", c.data)
    cache_id = int(parts[2])
    with closing(db()) as conn:
        c_db = conn.cursor()
        c_db.execute("SELECT title, summary FROM news_cache WHERE id=%s", (cache_id,))
        row = c_db.fetchone()
    if not row:
        await c.answer("Не удалось найти статью.", show_alert=True)
        return
    title, summary = row
    # Try OpenAI translate (short), else simple fallback
    try:
        prompt = f"Translate this short article to Russian, keep sentences aligned:\n\nTitle: {title}\n\n{summary}"
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        translated = resp.choices[0].message["content"]
    except Exception:
        logging.exception("Translation failed via OpenAI; using naive fallback")
        # naive fallback: return the original for now
        translated = "(Translation unavailable)"
    await bot.send_message(c.from_user.id, f"<b>{title}</b>\n\n{translated}", parse_mode="HTML")


@dp.callback_query_handler(lambda c: c.data.startswith("news:done:"))
async def news_done(c: types.CallbackQuery):
    parts = c.data.split(":")
    log_event(c.from_user.id, "news_completed", {"cache_id": int(parts[2])})
    save_msg(c.from_user.id, "user", c.data)
    cache_id = int(parts[2])
    with closing(db()) as conn:
        c_db = conn.cursor()
        c_db.execute("SELECT questions FROM news_cache WHERE id=%s", (cache_id,))
        row = c_db.fetchone()
    if not row:
        await c.answer("Не удалось найти статью.", show_alert=True)
        return
    questions = json.loads(row[0] or "[]")
    if not questions:
        await bot.send_message(c.from_user.id, "Что ты думаешь по этой теме?")
        return

    # Send only the first question with instructions and an 'Another question' button
    q0 = questions[0]
    instr = (
        "Отлично - ты прочитал(а) статью! Чтобы выполнить задание - ответь на три вопроса или напиши bye 👋\n\n"
    )

    # Initialize news session
    USER_CHAT_SESSIONS[c.from_user.id] = {
        "type": "news",
        "cache_id": cache_id,
        "answers_count": 0,
        "last_q_index": 0
    }

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton("Другой вопрос 🔁", callback_data=f"news:next:{cache_id}:1")],
            [InlineKeyboardButton("Перевести 🔁", callback_data=f"translate:news:{cache_id}:0")]
        ]
    )
    await bot.send_message(c.from_user.id, instr + q0, reply_markup=kb)


@dp.callback_query_handler(lambda c: c.data.startswith("news:next:"))
async def news_next(c: types.CallbackQuery):
    # callback format: news:next:<cache_id>:<index>
    parts = c.data.split(":")
    log_event(c.from_user.id, "news_question_answered", {"cache_id": int(parts[2]), "index": int(parts[3])})
    save_msg(c.from_user.id, "user", c.data)
    cache_id = int(parts[2])
    idx = int(parts[3])

    # Update session index if it exists and is news
    session = USER_CHAT_SESSIONS.get(c.from_user.id)
    if session and session.get("type") == "news":
        session["last_q_index"] = idx

    with closing(db()) as conn:
        c_db = conn.cursor()
        c_db.execute("SELECT questions FROM news_cache WHERE id=%s", (cache_id,))
        row = c_db.fetchone()
    if not row:
        await c.answer("Не удалось найти статью.", show_alert=True)
        return
    questions = json.loads(row[0] or "[]")
    if idx < 0 or idx >= len(questions):
        await c.answer("No more questions.", show_alert=True)
        return
    q = questions[idx]
    next_idx = idx + 1
    kb_buttons = []
    if next_idx < len(questions):
        kb_buttons.append(
            InlineKeyboardButton(
                "Другой вопрос 🔁", callback_data=f"news:next:{cache_id}:{next_idx}"
            )
        )
    kb = InlineKeyboardMarkup(inline_keyboard=[kb_buttons])
    kb.add(InlineKeyboardButton("Перевести 🔁", callback_data=f"translate:news:{cache_id}:{idx}"))
    await bot.send_message(c.from_user.id, q, reply_markup=kb)

@dp.callback_query_handler(lambda c: c.data == "menu:main")
async def menu_main_callback(c: types.CallbackQuery):
    await c.answer()
    user_id = c.from_user.id
    
    # Update streak and check if we should show notification
    streak, is_new_day = update_streak(user_id)
    show_notification = is_new_day and should_show_streak_notification(user_id)
    
    if show_notification:
        # Show streak notification
        mark_streak_notified(user_id)
        logging.info(f"[menu_main_callback] Showing streak notification: user={user_id}, streak={streak}")
        
        streak_emoji = "🔥" * min(streak, 5)  # Show up to 5 fire emojis
        await c.message.answer(
            f"🎉 <b>Отличная работа!</b>\n\n"
            f"{streak_emoji} Победная серия: <b>{streak} {get_day_word(streak)}</b>\n\n"
            f"Тренируйся ежедневно и общайся как носитель! 💪",
        )
        
        # Wait before showing menu
        await asyncio.sleep(2)
    
    try:
        await c.message.edit_text(
            "Меню активности — выбери, что хочешь сделать:",
            reply_markup=mode_keyboard()
        )
    except Exception:
        # Если сообщение нельзя отредактировать, отправляем новое
        await c.message.answer(
            "Меню активности — выбери, что хочешь сделать:",
            reply_markup=mode_keyboard()
        )
    # Also clear active chat topic session when user returns to menu from callbacks
    USER_CHAT_SESSIONS.pop(user_id, None)


@dp.message_handler(commands=["topics"])
async def cmd_topics(m: types.Message):
    save_msg(m.from_user.id, "user", "/topics")
    session_id = get_session_id(m.from_user.id)
    log_event(m.from_user.id, "command_used", {"command": "/topics"})
    user = get_user(m.from_user.id)
    current = (user.get("topics") or "").split(",") if user and user.get("topics") else []
    await m.answer("Update your interests 🌟:", reply_markup=topic_keyboard(current))


@dp.message_handler(commands=["stats"])
async def cmd_stats(m: types.Message):
    save_msg(m.from_user.id, "user", "/stats")
    session_id = get_session_id(m.from_user.id)
    log_event(m.from_user.id, "command_used", {"command": "/stats"})
    # Return basic usage statistics: total users, users with level, activity windows, messages, news-engaged users.
    admin_env = os.getenv("ADMIN_ID")
    if admin_env:
        try:
            admins = {int(x.strip()) for x in admin_env.split(",") if x.strip()}
        except Exception:
            admins = set()
        if m.from_user.id not in admins:
            await m.answer("Доступ к статистике ограничен.")
            return

    with closing(db()) as conn:
        c = conn.cursor()
        total_users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        users_with_level = c.execute(
            "SELECT COUNT(*) FROM users WHERE level IS NOT NULL AND level != ''"
        ).fetchone()[0]
        total_messages = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        active_7 = c.execute(
            "SELECT COUNT(DISTINCT user_id) FROM messages WHERE datetime(created_at) >= datetime('now','-7 days')"
        ).fetchone()[0]
        active_30 = c.execute(
            "SELECT COUNT(DISTINCT user_id) FROM messages WHERE datetime(created_at) >= datetime('now','-30 days')"
        ).fetchone()[0]
        news_users = c.execute(
            "SELECT COUNT(*) FROM users WHERE (topics IS NOT NULL AND topics != '') OR mode='news'"
        ).fetchone()[0]

    resp = (
        f"📊 Статистика\n"
        f"Всего зарегистрированных пользователей: {total_users}\n"
        f"Пользователей с уровнем: {users_with_level}\n"
        f"Активных за 7 дней: {active_7}\n"
        f"Активных за 30 дней: {active_30}\n"
        f"Всего сообщений в БД: {total_messages}\n"
        f"Пользователей, заинтересовавшихся новостями (approx): {news_users}\n\n"
        "Примечание: это приближённые метрики, основанные на таблицах users/messages."
    )
    await m.answer(resp)


@dp.message_handler(commands=["stats"])
async def cmd_stats(m: types.Message):
    save_msg(m.from_user.id, "user", "/stats")
    admin_env = os.getenv("ADMIN_ID")
    if admin_env:
        try:
            admins = {int(x.strip()) for x in admin_env.split(",") if x.strip()}
        except Exception:
            admins = set()
        if m.from_user.id not in admins:
            await m.answer("Доступ к статистике ограничен.")
            return
    with closing(db()) as conn:
        c = conn.cursor()
        total_users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        users_with_level = c.execute(
            "SELECT COUNT(*) FROM users WHERE level IS NOT NULL AND level != ''"
        ).fetchone()[0]
        total_messages = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        active_7 = c.execute(
            "SELECT COUNT(DISTINCT user_id) FROM messages WHERE datetime(created_at) >= datetime('now','-7 days')"
        ).fetchone()[0]
        active_30 = c.execute(
            "SELECT COUNT(DISTINCT user_id) FROM messages WHERE datetime(created_at) >= datetime('now','-30 days')"
        ).fetchone()[0]
        news_users = c.execute(
            "SELECT COUNT(*) FROM users WHERE (topics IS NOT NULL AND topics != '') OR mode='news'"
        ).fetchone()[0]
    resp = (
        f"📊 Статистика\n"
        f"Всего зарегистрированных пользователей: {total_users}\n"
        f"Пользователей с уровнем: {users_with_level}\n"
        f"Активных за 7 дней: {active_7}\n"
        f"Активных за 30 дней: {active_30}\n"
        f"Всего сообщений в БД: {total_messages}\n"
        f"Пользователей, заинтересовавшихся новостями (approx): {news_users}\n\n"
        "Примечание: это приближённые метрики, основанные на таблицах users/messages."
    )
    await m.answer(resp)


@dp.message_handler(commands=["level"])
async def cmd_level(m: types.Message):
    save_msg(m.from_user.id, "user", "/level")
    session_id = get_session_id(m.from_user.id)
    log_event(m.from_user.id, "command_used", {"command": "/level"})
    await m.answer("Pick your level 🎯:", reply_markup=level_keyboard())


@dp.message_handler(commands=["news"])
async def cmd_news(m: types.Message):
    # Explicit switch to news: clear any active chat topic session
    USER_CHAT_SESSIONS.pop(m.from_user.id, None)
    log_event(m.from_user.id, "news_requested", {})
    user_id = m.from_user.id
    save_msg(user_id, "user", "/news")
    session_id = get_session_id(user_id)
    log_event(user_id, "command_used", {"command": "/news"})
    # If user has never chosen news topics, send them to topic selection first
    user = get_user(user_id)
    existing = []
    if user and user.get("topics"):
        existing = [t.strip() for t in (user.get("topics") or "").split(",") if t.strip()]

    if not existing:
        await m.answer(
            "Сначала выбери темы новостей, которые тебе интересны (их всегда можно изменить командой /newstopics):",
            reply_markup=topic_keyboard(existing),
        )
        return

    # Check limit BEFORE incrementing
    paid = is_paid_user(user_id)
    
    if not paid:
        count_row = get_user_article_count(user_id)
        today = date.today()
        daily_articles = count_row[0] if count_row else 0
        last_reset = count_row[1] if count_row else None
        
        logging.info(f"[/news] user={user_id} paid={paid} daily_articles={daily_articles} last_reset={last_reset} today={today} limit={FREE_ARTICLE_LIMIT}")
        
        # If last reset was today, use current count; otherwise it's a new day (count will reset)
        if last_reset == today and daily_articles >= FREE_ARTICLE_LIMIT:
            increment_user_counter(user_id, "paywall_shown")
            log_event(user_id, "paywall_shown", {"reason": "article_limit", "count": daily_articles})
            logging.info(f"[/news] PAYWALL TRIGGERED for user={user_id}")
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton("Приобрести доступ 💎", callback_data="profile_buy_unlimited")],
                [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
            ])
            await m.answer(
                "🔒 Ты прочитал 3 статьи сегодня!\n\n"
                "Чтобы продолжить без ограничений, приобрети безлимитный доступ 💎",
                reply_markup=kb
            )
            return

    # Now increment (this will also reset if new day)
    increment_user_article_count(user_id)
    logging.info(f"[/news] Article count incremented for user={user_id}")
    
    await send_news(user_id)


@dp.message_handler(commands=["review"])
async def cmd_review(m: types.Message):
    with closing(db()) as conn:
        c_db = conn.cursor()
        c_db.execute(
            "SELECT phrase, example, bin FROM vocab WHERE user_id=%s ORDER BY bin ASC, added_at DESC LIMIT 6",
            (m.from_user.id,)
        )
        items = c_db.fetchall()
    if not items:
        await m.answer(
            "No vocab yet — chat a bit or try /news and I’ll save useful phrases for you. ✨"
        )
        return
    msg = "<b>Quick review:</b>\n"
    for i, (p, e, b) in enumerate(items, 1):
        msg += f"{i}) <b>{p}</b> — <i>{e}</i>\n"
    await m.answer(msg)


@dp.message_handler(commands=["help"])
async def cmd_help(m: types.Message):
    save_msg(m.from_user.id, "user", "/help")
    session_id = get_session_id(m.from_user.id)
    log_event(m.from_user.id, "command_used", {"command": "/help"})
    await m.answer(
        "Try /news for a fresh topic 📰, /topics to change interests, /newstopics to change news topics, /level to adjust difficulty, /review for phrases. Or just chat with me in English! 😊"
    )

@dp.message_handler(commands=["menu"])
async def cmd_menu(m: types.Message):
    save_msg(m.from_user.id, "user", "/menu")
    user_id = m.from_user.id
    session_id = get_session_id(user_id)
    log_event(user_id, "command_used", {"command": "/menu"})
    # From the main menu there should be no active chat topic session
    USER_CHAT_SESSIONS.pop(user_id, None)
    
    # Update streak and check if we should show notification
    streak, is_new_day = update_streak(user_id)
    show_notification = is_new_day and should_show_streak_notification(user_id)
    
    if show_notification:
        # Show streak notification
        mark_streak_notified(user_id)
        
        streak_emoji = "🔥" * min(streak, 5)  # Show up to 5 fire emojis
        await m.answer(
            f"🎉 <b>Отличная работа!</b>\n\n"
            f"{streak_emoji} Победная серия: <b>{streak} {get_day_word(streak)}</b>\n\n"
            f"Тренируйся ежедневно и общайся как носитель! 💪"
        )
        
        # Wait before showing menu
        await asyncio.sleep(2)
    
    await m.answer(
        "Меню активности — выбери, что хочешь сделать:",
        reply_markup=mode_keyboard()
    )


@dp.message_handler(commands=["subscribe", "premium"])
async def cmd_subscribe(m: types.Message):
    save_msg(m.from_user.id, "user", "/subscribe")
    log_event(m.from_user.id, "subscription_screen_opened", {"source": "command"})
    
    text = (
        "<b>Безлимитный доступ 💎</b>\n\n"
        "✅ Безлимитные разборы грамматики и лексики с Максом\n"
        "✅ Каждый день свежие новости на любые темы\n"
        "✅ Неограниченные сеты игры «Исправь грамматику»\n"
        "✅ Голосовой режим (скоро)\n\n"
        "Выбери подходящий тариф:"
    )
    
    await m.answer(text, reply_markup=subscription_keyboard())


@dp.callback_query_handler(lambda c: c.data == "pay:subscribe")
async def pay_subscribe_cb(c: types.CallbackQuery):
    save_msg(c.from_user.id, "user", c.data)
    increment_user_counter(c.from_user.id, "subscribe_click")
    if not PAYMENTS_PROVIDER_TOKEN:
        await c.answer("Платежи недоступны", show_alert=True)
        return
         
    # Reuse /subscribe flow
    try:
        amount_minor = SUBSCRIPTION_PRICE * 100
    except Exception:
        amount_minor = 29900
    prices = [LabeledPrice(label="Подписка на месяц", amount=amount_minor)]
    title = "Подписка PenPal English"
    description = "Неограниченный доступ к статьям и тренировкам на месяц."
    payload = "subscription-month-1"
    start_parameter = "subscribe"
    try:
        await bot.send_invoice(
            c.from_user.id,
            title=title,
            description=description,
            provider_token=PAYMENTS_PROVIDER_TOKEN,
            currency=SUBSCRIPTION_CURRENCY,
            prices=prices,
            start_parameter=start_parameter,
            payload=payload,
            need_name=False,
            need_email=False,
            is_flexible=False,
        )
        await c.answer()
    except Exception:
        logging.exception("Failed to send invoice from callback")
        await c.answer("Не удалось сформировать счёт", show_alert=True)


@dp.pre_checkout_query_handler(lambda q: True)
async def process_pre_checkout_q(pre_checkout_query: types.PreCheckoutQuery):
    try:
        await bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)
    except Exception:
        logging.exception("Failed to answer pre-checkout")


@dp.message_handler(content_types=types.ContentTypes.SUCCESSFUL_PAYMENT)
async def successful_payment(m: types.Message):
    try:
        sp = m.successful_payment
        logging.info(f"Payment success: user={m.from_user.id} total={sp.total_amount} {sp.currency} payload={sp.invoice_payload}")
    except Exception:
        logging.exception("Unable to log successful payment")
    # Mark user as paid
    try:
        set_user_subscription(m.from_user.id, "paid")
    except Exception:
        logging.exception("Failed to set user subscription to paid")
    await m.answer("Спасибо за оплату! Подписка активирована — теперь доступ к статьям без ограничений. ✨")


@dp.message_handler(commands=["settz"])
async def cmd_settz(m: types.Message):
    save_msg(m.from_user.id, "user", m.text)
    # Set user timezone. Usage: /settz Europe/Moscow
    parts = (m.text or "").split()
    if len(parts) < 2:
        await m.answer("Usage: /settz Europe/Moscow (use TZ database name)")
        return
    tz = parts[1].strip()
    try:
        if ZoneInfo is None:
            await m.answer("Timezone support is not available on this Python environment.")
            return
        _ = ZoneInfo(tz)
    except Exception:
        await m.answer("Unknown timezone. Use a TZ database name like Europe/Moscow or America/New_York.")
        return
    set_user_timezone(m.from_user.id, tz)
    await m.answer(f"Timezone set to {tz}. I will message you at 12:00 local time.")


@dp.callback_query_handler(lambda c: c.data.startswith("translate:"))
async def translate_message(c: types.CallbackQuery):
    user_id = c.from_user.id
    parts = c.data.split(":")
    mode = parts[1]
    text_to_translate = None

    if mode == "chat":
        # Fetch last assistant message
        with closing(db()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT content FROM messages WHERE user_id=%s AND role='assistant' ORDER BY id DESC LIMIT 1", (user_id,))
            row = cur.fetchone()
            if row:
                text_to_translate = row[0]
    elif mode == "news":
        cache_id = int(parts[2])
        idx = int(parts[3])
        with closing(db()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT questions FROM news_cache WHERE id=%s", (cache_id,))
            row = cur.fetchone()
            if row:
                questions = json.loads(row[0] or "[]")
                if 0 <= idx < len(questions):
                    text_to_translate = questions[idx]
    elif mode == "tasks":
        session = USER_CHAT_SESSIONS.get(user_id)
        if session and "tasks" in session:
            tasks = session["tasks"]
            text_to_translate = "\n".join([f"{t['id']}) {t['text']}" for t in tasks[:3]])

    if not text_to_translate:
        await c.answer("Нечего переводить.", show_alert=True)
        return

    await c.answer("Перевожу... ⏳")
    
    # Perform translation
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translator. Translate the following English text to Russian. Return only the translation."},
                {"role": "user", "content": text_to_translate}
            ],
            temperature=0.3,
        )
        translated = resp.choices[0].message["content"]
    except Exception:
        logging.exception("Translation failed")
        translated = "Перевод временно недоступен."

    await bot.send_message(user_id, f"<b>Перевод:</b>\n{translated}")


# --- Truth/Lie Game Logic ---

TRUTH_LIE_TOPICS = {
    "health": "Здоровье 🧠",
    "geography": "География 🌍",
    "animals": "Животные 🐾",
    "technologies": "Технологии 🤖"
}

TRUTH_LIE_FALLBACKS = {
    "health": {
        "facts": [
            "Apples float in water because they are 25% air.",
            "Your heart beats about 100,000 times a day.",
            "Drinking water before meals makes you gain weight."
        ],
        "lie_index": 2,
        "explanation": "Вода перед едой часто помогает снизить аппетит, а не набрать вес."
    },
    "geography": {
        "facts": [
            "Russia is the largest country in the world by area.",
            "The Amazon River is the longest river in the world.",
            "Antarctica is the driest continent on Earth."
        ],
        "lie_index": 1,
        "explanation": "Самая длинная река в мире — Нил (хотя споры продолжаются, официально это Нил)."
    },
    "animals": {
        "facts": [
            "Octopuses have three hearts.",
            "Cows can sleep standing up.",
            "Goldfish have a memory span of only 3 seconds."
        ],
        "lie_index": 2,
        "explanation": "У золотых рыбок память может сохраняться месяцами, миф о 3 секундах неверен."
    },
    "technologies": {
        "facts": [
            "The first computer mouse was made of wood.",
            "Python was named after the snake species.",
            "The QWERTY keyboard was designed to slow down typing."
        ],
        "lie_index": 1,
        "explanation": "Язык Python назван в честь комедийной группы «Монти Пайтон», а не змеи."
    }
}

def populate_game_sets():
    """
    Ensure there are enough game sets in the DB for each topic.
    If not, generate them via GPT.
    """
    TARGET_SETS_PER_TOPIC = 10
    
    logging.info("Checking game sets population...")
    
    for topic_key, topic_label in TRUTH_LIE_TOPICS.items():
        try:
            # Check count
            with closing(db()) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM game_truth_lie_sets WHERE topic = %s", (topic_key,))
                count = c.fetchone()[0]
            
            if count >= TARGET_SETS_PER_TOPIC:
                logging.info(f"Topic {topic_key} has enough sets ({count}).")
                continue

            needed = TARGET_SETS_PER_TOPIC - count
            logging.info(f"Generating {needed} sets for {topic_key}...")
            
            topic_ru = topic_label
            prompt = (
                f"Сгенерируй 5 разных наборов для игры «2 правды и 1 ложь» по теме {topic_ru}.\n"
                "Каждый набор должен содержать: 3 факта (2 правды, 1 ложь), индекс лжи (0, 1 или 2) и объяснение.\n"
                "Верни строго валидный JSON список: \n"
                "[\n"
                "  {\"facts\": [\"fact1\", \"fact2\", \"fact3\"], \"lie_index\": 1, \"explanation\": \"...\"},\n"
                "  ...\n"
                "]\n"
                "Факты на английском (B1), объяснение на русском."
            )

            # We loop until we have enough
            while needed > 0:
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Ты генератор контента. Отвечай только JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                    )
                    content = resp.choices[0].message["content"]
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[0].strip()
                    
                    sets = json.loads(content)
                    if not isinstance(sets, list):
                        sets = [sets] # Handle single object case
                    
                    with closing(db()) as conn:
                        c = conn.cursor()
                        for s in sets:
                            if needed <= 0: break
                            # Validate
                            if "facts" in s and len(s["facts"]) == 3 and "lie_index" in s:
                                # Insert
                                c.execute("""
                                    INSERT INTO game_truth_lie_sets (topic, facts, lie_index, explanation, created_at)
                                    VALUES (%s, %s, %s, %s, now())
                                """, (topic_key, json.dumps(s["facts"]), s["lie_index"], s["explanation"]))
                                needed -= 1
                        conn.commit()
                        
                except Exception as e:
                    logging.error(f"Failed to generate batch for {topic_key}: {e}")
                    break
        except Exception as e:
            logging.error(f"Error in populate_game_sets for {topic_key}: {e}")

def get_truth_lie_set(user_id, topic_key):
    """
    Get a game set for the user from DB only.
    """
    with closing(db()) as conn:
        c = conn.cursor()
        # Find sets for this topic that user hasn't seen
        c.execute("""
            SELECT id, facts, lie_index, explanation 
            FROM game_truth_lie_sets 
            WHERE topic = %s 
              AND id NOT IN (
                  SELECT set_id FROM user_game_truth_lie_history WHERE user_id = %s
              )
            ORDER BY RANDOM()
            LIMIT 1
        """, (topic_key, user_id))
        row = c.fetchone()
        
        if row:
            return {
                "id": row[0],
                "facts": row[1] if isinstance(row[1], list) else json.loads(row[1]),
                "lie_index": row[2],
                "explanation": row[3],
                "source": "db"
            }
        
        return None

def save_truth_lie_history(user_id, set_id, answer_index, is_correct):
    if set_id == -1: return # Don't save history for unsaved sets
    try:
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO user_game_truth_lie_history (user_id, set_id, answer_index, is_correct, created_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (user_id, set_id) DO UPDATE 
                SET answer_index = EXCLUDED.answer_index,
                    is_correct = EXCLUDED.is_correct,
                    created_at = now()
            """, (user_id, set_id, answer_index, is_correct))
            conn.commit()
    except Exception:
        logging.exception("Failed to save game history")

def get_grammar_set(user_id, level):
    """
    Get a grammar game set for the user from DB only.
    """
    with closing(db()) as conn:
        c = conn.cursor()
        # Find sets for this level that user hasn't seen
        c.execute("""
            SELECT id, sentences, wrong_index, explanation 
            FROM grammar_sets 
            WHERE level = %s 
              AND id NOT IN (
                  SELECT set_id FROM user_game_grammar_history WHERE user_id = %s
              )
            ORDER BY RANDOM()
            LIMIT 1
        """, (level, user_id))
        row = c.fetchone()
        
        if row:
            return {
                "id": row[0],
                "sentences": row[1] if isinstance(row[1], list) else json.loads(row[1]),
                "wrong_index": row[2],
                "explanation": row[3]
            }
        
        return None

def get_grammar_set_by_id(set_id):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT id, sentences, wrong_index, explanation FROM grammar_sets WHERE id = %s", (set_id,))
        row = c.fetchone()
        if row:
            return {
                "id": row[0],
                "sentences": row[1] if isinstance(row[1], list) else json.loads(row[1]),
                "wrong_index": row[2],
                "explanation": row[3]
            }
    return None

def save_grammar_history(user_id, set_id, answer_index, is_correct):
    try:
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO user_game_grammar_history (user_id, set_id, answer_index, is_correct, created_at)
                VALUES (%s, %s, %s, %s, now())
            """, (user_id, set_id, answer_index, is_correct))
            conn.commit()
    except Exception:
        logging.exception("Failed to save grammar game history")

def truth_lie_topics_kb():
    rows = []
    for key, label in TRUTH_LIE_TOPICS.items():
        rows.append([InlineKeyboardButton(label, callback_data=f"game:truth_lie:topic:{key}")])
    rows.append([InlineKeyboardButton("Меню 🏠", callback_data="menu:main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def truth_lie_answers_kb(set_id):
    # Buttons 1, 2, 3
    row = [
        InlineKeyboardButton("1", callback_data=f"game:truth_lie:answer:{set_id}:0"),
        InlineKeyboardButton("2", callback_data=f"game:truth_lie:answer:{set_id}:1"),
        InlineKeyboardButton("3", callback_data=f"game:truth_lie:answer:{set_id}:2"),
    ]
    # Add translate button
    translate_btn = [InlineKeyboardButton("Перевести 🇷🇺", callback_data="game:truth_lie:translate")]
    return InlineKeyboardMarkup(inline_keyboard=[row, translate_btn, [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]])

def truth_lie_post_game_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Сыграть ещё раз 🎮", callback_data="game:truth_lie:start")],
        [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
    ])

def grammar_levels_kb():
    levels = ["A2", "B1", "B2", "C1"]
    row = [InlineKeyboardButton(l, callback_data=f"game:grammar:level:{l}") for l in levels]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]])

def grammar_answers_kb(set_id):
    row = [
        InlineKeyboardButton("1", callback_data=f"game:grammar:answer:{set_id}:0"),
        InlineKeyboardButton("2", callback_data=f"game:grammar:answer:{set_id}:1"),
        InlineKeyboardButton("3", callback_data=f"game:grammar:answer:{set_id}:2"),
    ]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]])

def grammar_post_game_kb(level, set_id):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Посмотреть правило 🤓", callback_data=f"game:grammar:rule:{set_id}")],
        [InlineKeyboardButton("Следующий сет ➡️", callback_data=f"game:grammar:level:{level}")],
        [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
    ])

# --- Handlers ---

@dp.message_handler(commands=["game_truth_lie"])
async def cmd_game_truth_lie(m: types.Message):
    log_event(m.from_user.id, "game_started", {"game_type": "truth_lie"})
    USER_CHAT_SESSIONS.pop(m.from_user.id, None) # Clear other sessions
    await m.answer("Выбери тему для игры «2 правды и 1 ложь»:", reply_markup=truth_lie_topics_kb())

@dp.callback_query_handler(lambda c: c.data == "game:truth_lie:start")
async def cb_game_restart(c: types.CallbackQuery):
    await c.answer()
    await c.message.edit_text("Выбери тему:", reply_markup=truth_lie_topics_kb())

@dp.callback_query_handler(lambda c: c.data.startswith("game:truth_lie:topic:"))
async def cb_truth_lie_topic(c: types.CallbackQuery):
    topic_key = c.data.split(":")[-1]
    user_id = c.from_user.id
    log_event(user_id, "truth_lie_topic_selected", {"topic": topic_key})
    
    # Check free user limit
    if not is_paid_user(user_id):
        truth_lie_count = get_user_truth_lie_count_today(user_id)
        if truth_lie_count >= FREE_TRUTH_LIE_LIMIT:
            log_event(user_id, "truth_lie_limit_reached", {"count": truth_lie_count})
            await c.answer()
            await c.message.edit_text(
                "🔒 Ты сыграл 3 раунда «2 правды 1 ложь» сегодня!\n\n"
                "Чтобы продолжить без ограничений, приобрети безлимитный доступ 💎",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton("Приобрести доступ 💎", callback_data="profile_buy_unlimited")],
                    [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
                ])
            )
            return
    
    await c.answer("Ищу факты... 🕵️")
    
    game_set = get_truth_lie_set(user_id, topic_key)
    if not game_set:
        await c.message.edit_text("Не удалось загрузить игру. Попробуй позже.", reply_markup=truth_lie_post_game_kb())
        return

    # Save session state
    USER_CHAT_SESSIONS[user_id] = {
        "type": "truth_lie",
        "set_id": game_set["id"],
        "lie_index": game_set["lie_index"],
        "explanation": game_set["explanation"],
        "topic": topic_key,
        "facts": game_set["facts"]
    }
    
    log_event(user_id, "truth_lie_set_shown", {
        "topic": topic_key, 
        "set_id": game_set["id"], 
        "source": game_set.get("source")
    })

    facts_text = ""
    for i, f in enumerate(game_set["facts"]):
        facts_text += f"{i+1}) {f}\n"

    msg = (
        f"Тема: {TRUTH_LIE_TOPICS.get(topic_key, topic_key)}\n\n"
        "🕵️ Я пришлю 3 факта. Два из них правдивы, один — ложным, но правдоподобным. Угадай, какой факт ложный: 1, 2 или 3.\n\n"
        f"{facts_text}"
    )
    
    await c.message.edit_text(msg, reply_markup=truth_lie_answers_kb(game_set["id"]))

@dp.callback_query_handler(lambda c: c.data == "game:truth_lie:translate")
async def cb_truth_lie_translate(c: types.CallbackQuery):
    user_id = c.from_user.id
    session = USER_CHAT_SESSIONS.get(user_id)
    
    if not session or session.get("type") != "truth_lie" or not session.get("facts"):
        await c.answer("Перевод недоступен.", show_alert=True)
        return

    await c.answer("Перевожу... ⏳")
    
    facts = session["facts"]
    text_to_translate = "\n".join([f"{i+1}) {f}" for i, f in enumerate(facts)])
    
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты переводчик. Переведи эти факты на русский язык. Сохрани нумерацию."},
                {"role": "user", "content": text_to_translate}
            ],
            temperature=0.3,
        )
        translation = resp.choices[0].message["content"]
        await c.message.answer(f"🇷🇺 Перевод:\n\n{translation}")
    except Exception:
        logging.exception("Translation failed")
        await c.message.answer("Не удалось перевести. Попробуй позже.")

@dp.callback_query_handler(lambda c: c.data.startswith("game:truth_lie:answer:"))
async def cb_truth_lie_answer(c: types.CallbackQuery):
    parts = c.data.split(":")
    set_id = int(parts[3])
    answer_idx = int(parts[4])
    user_id = c.from_user.id
    
    session = USER_CHAT_SESSIONS.get(user_id)
    
    # Validate session
    if not session or session.get("type") != "truth_lie" or session.get("set_id") != set_id:
        await c.answer("Эта игра устарела.", show_alert=True)
        await c.message.edit_text("Игра устарела. Начни новую.", reply_markup=truth_lie_post_game_kb())
        return

    correct_lie_idx = session["lie_index"]
    is_correct = (answer_idx == correct_lie_idx)
    
    log_event(user_id, "truth_lie_answered", {
        "set_id": set_id, 
        "answer_index": answer_idx, 
        "is_correct": is_correct
    })
    
    # Save history
    save_truth_lie_history(user_id, set_id, answer_idx, is_correct)
    
    # Prepare result message
    if is_correct:
        res_header = "✅ Верно! Отличная работа!"
        res_body = f"Ложным был факт №{correct_lie_idx + 1}. {session['explanation']}"
    else:
        res_header = "Почти, но нет 🙂"
        res_body = f"На самом деле ложный факт — №{correct_lie_idx + 1}. {session['explanation']}"
        
    await c.message.edit_text(
        f"{res_header}\n\n{res_body}",
        reply_markup=truth_lie_post_game_kb()
    )
    
    log_event(user_id, "truth_lie_completed", {
        "set_id": set_id, 
        "topic": session["topic"], 
        "is_correct": is_correct
    })
    
    # Clear session
    USER_CHAT_SESSIONS.pop(user_id, None)

# --- Grammar Game Handlers ---

@dp.callback_query_handler(lambda c: c.data == "game:grammar:start")
async def cb_game_grammar_start(c: types.CallbackQuery):
    log_event(c.from_user.id, "grammar_game_opened", {})
    USER_CHAT_SESSIONS.pop(c.from_user.id, None)
    await c.answer()
    await c.message.edit_text("Выбери уровень сложности:", reply_markup=grammar_levels_kb())

@dp.callback_query_handler(lambda c: c.data.startswith("game:grammar:level:"))
async def cb_grammar_level(c: types.CallbackQuery):
    level = c.data.split(":")[-1]
    user_id = c.from_user.id
    log_event(user_id, "grammar_level_selected", {"level": level})
    
    # Check free user limit
    if not is_paid_user(user_id):
        grammar_count = get_user_grammar_count_today(user_id)
        if grammar_count >= FREE_GRAMMAR_LIMIT:
            log_event(user_id, "grammar_limit_reached", {"count": grammar_count})
            await c.answer()
            await c.message.edit_text(
                "🔒 Ты прошёл 3 сета грамматики сегодня!\n\n"
                "Чтобы продолжить без ограничений, приобрети безлимитный доступ 💎",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton("Приобрести доступ 💎", callback_data="profile_buy_unlimited")],
                    [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
                ])
            )
            return
    
    await c.answer("Ищу задания... 🕵️")
    
    game_set = get_grammar_set(user_id, level)
    if not game_set:
        log_event(user_id, "grammar_no_sets_left", {"level": level})
        await c.message.edit_text(f"Ты прошёл все задания уровня {level}! 🚀", reply_markup=mode_keyboard())
        return

    # Save session state
    USER_CHAT_SESSIONS[user_id] = {
        "type": "grammar",
        "set_id": game_set["id"],
        "wrong_index": game_set["wrong_index"],
        "explanation": game_set["explanation"],
        "level": level,
        "sentences": game_set["sentences"]
    }
    
    log_event(user_id, "grammar_set_shown", {
        "level": level, 
        "set_id": game_set["id"]
    })

    sentences_text = ""
    for i, s in enumerate(game_set["sentences"]):
        sentences_text += f"{i+1}) {s}\n"

    msg = (
        f"Уровень: {level}\n\n"
        "🎯 Найди предложение с грамматической ошибкой (оно здесь одно):\n\n"
        f"{sentences_text}"
    )
    
    await c.message.edit_text(msg, reply_markup=grammar_answers_kb(game_set["id"]))

@dp.callback_query_handler(lambda c: c.data.startswith("game:grammar:answer:"))
async def cb_grammar_answer(c: types.CallbackQuery):
    parts = c.data.split(":")
    set_id = int(parts[3])
    answer_idx = int(parts[4])
    user_id = c.from_user.id
    
    session = USER_CHAT_SESSIONS.get(user_id)
    
    # Validate session
    if not session or session.get("type") != "grammar" or session.get("set_id") != set_id:
        await c.answer("Эта игра устарела.", show_alert=True)
        await c.message.edit_text("Игра устарела. Начни новую.", reply_markup=grammar_levels_kb())
        return

    correct_wrong_idx = session["wrong_index"]
    
    # Fix for 1-based indexing in DB (if present)
    # If correct_wrong_idx is 1, 2, 3 -> treat as 1-based.
    # If correct_wrong_idx is 0 -> treat as 0-based.
    real_idx = correct_wrong_idx - 1 if correct_wrong_idx > 0 else correct_wrong_idx
    
    is_correct = (answer_idx == real_idx)
    
    log_event(user_id, "grammar_answer_submitted", {
        "set_id": set_id, 
        "answer_index": answer_idx, 
        "is_correct": is_correct
    })
    
    # Save history
    save_grammar_history(user_id, set_id, answer_idx, is_correct)
    
    # Prepare result message
    sentences_text = ""
    for i, s in enumerate(session["sentences"]):
        sentences_text += f"{i+1}) {s}\n"

    if is_correct:
        res_header = "✅ Верно! Ты нашёл ошибку."
        res_body = f"{session['explanation']}"
    else:
        # Display 1-based index
        display_idx = correct_wrong_idx if correct_wrong_idx > 0 else correct_wrong_idx + 1
        res_header = "Не совсем так ❌"
        res_body = f"Ошибка была в предложении №{display_idx}. {session['explanation']}"
        
    await c.message.edit_text(
        f"{sentences_text}\n{res_header}\n\n{res_body}",
        reply_markup=grammar_post_game_kb(session["level"], set_id)
    )
    
    # Clear session
    USER_CHAT_SESSIONS.pop(user_id, None)

@dp.callback_query_handler(lambda c: c.data.startswith("game:grammar:rule:"))
async def cb_grammar_rule(c: types.CallbackQuery):
    set_id = int(c.data.split(":")[-1])
    user_id = c.from_user.id
    log_event(user_id, "grammar_rule_requested", {"set_id": set_id})
    
    await c.answer("Спрашиваю у AI... 🤖")
    
    game_set = get_grammar_set_by_id(set_id)
    if not game_set:
        await c.message.answer("Не удалось найти задание.")
        return

    prompt = (
        f"Explain the grammar rule for this error briefly (in Russian).\n"
        f"Sentences: {game_set['sentences']}\n"
        f"Explanation: {game_set['explanation']}\n"
        "Keep it simple and educational."
    )
    
    try:
        explanation = await gpt_chat([
            {"role": "system", "content": "You are a helpful English tutor. Explain grammar rules clearly in Russian."},
            {"role": "user", "content": prompt}
        ])
        await c.message.answer(f"🤓 <b>Справка по правилу:</b>\n\n{explanation}")
    except Exception:
        await c.message.answer("Не удалось получить справку. Попробуй позже.")

# --- Profile Handlers ---

@dp.message_handler(commands=["profile"])
async def cmd_profile(m: types.Message):
    update_streak(m.from_user.id)
    await show_profile(m.from_user.id, m)

@dp.callback_query_handler(lambda c: c.data == "mode:profile")
async def cb_mode_profile(c: types.CallbackQuery):
    update_streak(c.from_user.id)
    await c.answer()
    await show_profile(c.from_user.id, c.message)

def subscription_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("500₽ в месяц", callback_data="pay:monthly")],
        [InlineKeyboardButton("1499₽ за год", callback_data="pay:yearly")],
        [InlineKeyboardButton("Назад ↩️", callback_data="mode:profile")]
    ])


@dp.callback_query_handler(lambda c: c.data == "profile_buy_unlimited")
async def cb_profile_buy(c: types.CallbackQuery):
    update_streak(c.from_user.id)
    log_event(c.from_user.id, "subscription_screen_opened", {})
    await c.answer()
    
    text = (
        "<b>Безлимитный доступ 💎</b>\n\n"
        "✅ Безлимитные разборы грамматики и лексики с Максом\n"
        "✅ Каждый день свежие новости на любые темы\n"
        "✅ Неограниченные сеты игры «Исправь грамматику»\n"
        "✅ Голосовой режим (скоро)\n\n"
        "Выбери подходящий тариф:"
    )
    
    await c.message.edit_text(text, reply_markup=subscription_keyboard())


@dp.callback_query_handler(lambda c: c.data in ["pay:monthly", "pay:yearly"])
async def cb_pay_plan(c: types.CallbackQuery):
    plan = c.data.split(":")[1]
    user_id = c.from_user.id
    
    log_event(user_id, "subscription_plan_selected", {"plan": plan})
    
    # Define amounts and descriptions
    amounts = {
        "monthly": 500.00,
        "yearly": 1499.00
    }
    
    descriptions = {
        "monthly": "Подписка PenPal English — 1 месяц",
        "yearly": "Подписка PenPal English — 1 год (экономия 4501₽)"
    }
    
    amount = amounts.get(plan, 500.00)
    description = descriptions.get(plan, "Подписка PenPal English")
    
    await c.answer("Создаю счёт... ⏳")
    
    # Create payment
    payment = create_payment(user_id, amount, plan, description)
    
    if not payment:
        await c.message.edit_text(
            "❌ Не удалось создать счёт. Попробуй позже или свяжись с поддержкой.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton("Назад ↩️", callback_data="profile_buy_unlimited")],
                [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
            ])
        )
        return
        
    # Get payment URL
    payment_url = payment.confirmation.confirmation_url
    
    # Create keyboard with payment button
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Оплатить 💳", url=payment_url)],
        [InlineKeyboardButton("Проверить оплату ✅", callback_data=f"check_payment:{payment.id}")],
        [InlineKeyboardButton("Назад ↩️", callback_data="profile_buy_unlimited")]
    ])
    
    await c.message.edit_text(
        f"💳 <b>Счёт создан!</b>\n\n"
        f"💰 Сумма: <b>{amount:.0f} ₽</b>\n"
        f"📦 Тариф: <b>{description}</b>\n\n"
        f"1️⃣ Нажми «Оплатить», чтобы перейти к оплате\n"
        f"2️⃣ После оплаты нажми «Проверить оплату»\n\n"
        f"Оплата защищена ЮKassa 🔒",
        reply_markup=kb
    )


@dp.callback_query_handler(lambda c: c.data.startswith("check_payment:"))
async def cb_check_payment(c: types.CallbackQuery):
    """Check payment status manually."""
    payment_id = c.data.split(":", 1)[1]
    user_id = c.from_user.id
    
    await c.answer("Проверяю оплату... ⏳")
    
    payment = check_payment_status(payment_id)
    
    if not payment:
        await c.answer("❌ Не удалось проверить статус платежа. Попробуй позже.", show_alert=True)
        return
        
    if payment.status == "succeeded":
        # Get plan from DB
        with closing(db()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT plan FROM payments WHERE payment_id = %s", (payment_id,))
            row = cur.fetchone()
            plan = row[0] if row else "unknown"
            
        # Activate subscription
        activate_subscription(user_id, plan)
        
        await c.message.edit_text(
            "✅ <b>Оплата прошла успешно!</b>\n\n"
            "🎉 Подписка активирована!\n\n"
            "Теперь у тебя безлимитный доступ:\n"
            "✅ Неограниченные статьи\n"
            "✅ Неограниченные игры\n"
            "✅ Неограниченная разговорная практика\n\n"
            "Спасибо за поддержку проекта! 💙",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
            ])
        )
        
    elif payment.status == "pending":
        await c.answer("⏳ Платёж ещё обрабатывается. Попробуй через минуту.", show_alert=True)
        
    elif payment.status == "canceled":
        await c.message.edit_text(
            "❌ <b>Платёж отменён</b>\n\n"
            "Попробуй ещё раз или свяжись с поддержкой, если возникли проблемы.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton("Попробовать снова 🔄", callback_data="profile_buy_unlimited")],
                [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
            ])
        )
    else:
        await c.answer(f"Статус платежа: {payment.status}", show_alert=True)


@dp.callback_query_handler(lambda c: c.data == "profile_news_settings")
async def cb_profile_news(c: types.CallbackQuery):
    update_streak(c.from_user.id)
    await c.answer()
    # Reuse logic from cmd_newstopics
    set_user_mode(c.from_user.id, "news")
    user = get_user(c.from_user.id)
    existing = []
    if user.get("topics"):
        existing = [t.strip() for t in (user.get("topics") or "").split(",") if t.strip()]
    await c.message.edit_text(
        "Выбери темы, которые тебе нравятся (эти темы всегда можно изменить командой /newstopics):",
        reply_markup=topic_keyboard(existing),
    )


def init_game_tables():
    try:
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS game_truth_lie_sets (
                    id SERIAL PRIMARY KEY,
                    topic TEXT,
                    facts JSONB,
                    lie_index INTEGER,
                    explanation TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_game_truth_lie_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    set_id INTEGER,
                    answer_index INTEGER,
                    is_correct BOOLEAN,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            conn.commit()
            
            # Migration: Ensure created_at exists
            try:
                c.execute("ALTER TABLE user_game_truth_lie_history ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()")
                conn.commit()
            except Exception:
                conn.rollback()

            # --- Grammar Game Tables ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS grammar_sets (
                    id SERIAL PRIMARY KEY,
                    level TEXT,
                    sentences JSONB,
                    wrong_index INTEGER,
                    explanation TEXT,
                    source TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_game_grammar_history (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    set_id BIGINT,
                    answer_index INTEGER,
                    is_correct BOOLEAN,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            conn.commit()
            
            # Migration: change set_id to BIGINT if needed
            try:
                c.execute("ALTER TABLE user_game_grammar_history ALTER COLUMN set_id TYPE BIGINT")
                c.execute("ALTER TABLE user_game_grammar_history ALTER COLUMN user_id TYPE BIGINT")
                conn.commit()
            except Exception:
                conn.rollback()
    except Exception as e:
        logging.error(f"Failed to init game tables: {e}")

# Helper functions for Profile, Streak, and Dictionary features

def update_streak(user_id):
    """
    Updates user streak based on last_active_date.
    Called from callback handlers that don't use save_msg.
    Skips DB write if already updated today.
    """
    today = date.today()
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT streak_count, last_active_date, max_streak FROM users WHERE id=%s", (user_id,))
        row = c.fetchone()
        
        if not row:
            return # User not found or not initialized

        streak_count = row[0] or 0
        last_active = row[1] # date object or None
        max_streak = row[2] or 0
        
        # Skip if already updated today
        if last_active == today:
            return
        
        new_streak = streak_count
        new_max = max_streak
        
        if last_active is None:
            new_streak = 1
            new_max = max(new_max, 1)
        elif last_active == today - timedelta(days=1):
            new_streak += 1
            new_max = max(new_max, new_streak)
        else:
            new_streak = 1 # Streak broken
            
        # Update DB
        c.execute("""
            UPDATE users 
            SET streak_count=%s, last_active_date=%s, max_streak=%s 
            WHERE id=%s
        """, (new_streak, today, new_max, user_id))
        conn.commit()

async def maybe_add_to_dictionary(m: types.Message):
    """
    Checks if message ends with 'словарь'.
    If so, adds the preceding text to user_dictionary.
    Returns True if dictionary action was taken (even if failed), False otherwise.
    """
    text = (m.text or "").strip()
    if not text:
        return False
        
    # Check for triggers: "словарь" at end OR "/word" at start
    lower_text = text.lower()
    content = None
    
    if lower_text.endswith("словарь"):
        content = text[:-7].strip()
    elif lower_text.startswith("/word"):
        content = text[5:].strip()
        
    if content is None:
        return False
        
    user_id = m.from_user.id
    
    if not content:
        await m.answer("Напиши слово, например: apple словарь или /word apple 🙂")
        return True
        
    # Check existence first
    exists = False
    try:
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute("SELECT 1 FROM user_dictionary WHERE user_id=%s AND word=%s", (user_id, content))
            if c.fetchone():
                exists = True
    except Exception as e:
        logging.error(f"Dictionary check error: {e}")
        return False

    if exists:
        await m.answer(f"ℹ️ «{content}» уже есть в словаре")
        log_event(user_id, "word_dictionary", {"word": content, "success": False, "reason": "duplicate"})
        return True

    # If not exists, get translation first
    wait_msg = await m.answer(f"⏳ Ищу значение для «{content}»...")
    
    definition = ""
    translation = ""
    
    try:
        response_text = await gpt_chat([
            {"role": "system", "content": "You are a helpful dictionary assistant. Return a JSON object with two keys: 'definition' (a brief definition in English) and 'translation' (the Russian translation of the word/phrase)."},
            {"role": "user", "content": content}
        ])
        
        # Parse JSON
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        definition = data.get("definition", "")
        translation = data.get("translation", "")
        
    except Exception as e:
        logging.error(f"GPT dictionary error: {e}")
        await wait_msg.edit_text("❌ Не удалось найти перевод. Попробуй позже.")
        return True

    # Now insert into DB with translation (only Russian translation)
    try:
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO user_dictionary (user_id, word, translation) VALUES (%s, %s, %s)", (user_id, content, translation))
            conn.commit()
            
        log_event(user_id, "word_dictionary", {"word": content, "success": True})
        await wait_msg.delete()
        
        display_text = f"🇬🇧 Definition: {definition}\n🇷🇺 Перевод: {translation}"
        await m.answer(f"✅ <b>{content}</b> добавлено в словарь.\n\n{display_text}")
        
    except Exception as e:
        logging.error(f"Dictionary insert error: {e}")
        log_event(user_id, "error", {"where": "dictionary_insert", "msg": str(e)[:200]})
        await wait_msg.edit_text("❌ Ошибка при сохранении в базу данных.")
        


    return True # Action taken

def get_profile_data(user_id):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT streak_count FROM users WHERE id=%s", (user_id,))
        row = c.fetchone()
        streak = row[0] if row else 0
        
        c.execute("SELECT COUNT(*) FROM user_dictionary WHERE user_id=%s", (user_id,))
        dict_count = c.fetchone()[0]
        
    return streak, dict_count

def profile_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Приобрести бесконечный доступ 💎", callback_data="profile_buy_unlimited")],
        [InlineKeyboardButton("Настроить темы новостей 🗞", callback_data="profile_news_settings")],
        [InlineKeyboardButton("Написать в поддержку 🆘", url="https://t.me/artyhere")],
        [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
    ])

async def show_profile(user_id, messageable):
    """
    Shows profile. messageable can be types.Message or types.CallbackQuery.message
    """
    streak, dict_count = get_profile_data(user_id)
    
    text = (
        "<b>Твой профиль</b> 👤\n\n"
        f"Победная серия: <b>{streak} дн. подряд</b> 🔥\n"
        f"Слов в словаре: <b>{dict_count}</b> 📚"
    )
    
    try:
        # Try editing first (if it's from a callback)
        await messageable.edit_text(text, reply_markup=profile_keyboard())
    except Exception:
        # If edit fails (e.g. called from command), send new
        await messageable.answer(text, reply_markup=profile_keyboard())

    log_event(user_id, "profile_opened", {})

# --- Word Training Logic ---

@dp.callback_query_handler(lambda c: c.data == "mode:train_words")
async def start_word_training(c: types.CallbackQuery):
    user_id = c.from_user.id
    
    # Fetch words
    with closing(db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT word, translation FROM user_dictionary WHERE user_id=%s AND translation IS NOT NULL AND translation != ''", (user_id,))
        rows = cur.fetchall()
        
    if len(rows) < 3:
        log_event(user_id, "word_training_insufficient_words", {"word_count": len(rows)})
        await c.answer()
        text = (
            "Для тренировки нужно минимум 3 слова в словаре 📚\n\n"
            "Добавь слова:\n"
            "1. Напиши <code>apple словарь</code>\n"
            "2. Или используй команду <code>/word apple</code>"
        )
        kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]])
        await c.message.edit_text(text, reply_markup=kb, parse_mode="HTML")
        return
    
    # Log event only when training actually starts
    log_event(user_id, "word_training_start", {"word_count": len(rows)})
    # We need 6 questions.
    # Q1-3: En -> Ru (Select translation)
    # Q4-6: Ru -> En (Select word)
    
    import random
    random.shuffle(rows)
    
    # If we have fewer than 6 words, we reuse them.
    # We need a pool of words for questions.
    # Let's pick 6 question items (word, translation).
    
    question_items = []
    while len(question_items) < 6:
        question_items.extend(rows)
    question_items = question_items[:6]
    random.shuffle(question_items)
    
    questions = []
    
    for i, (word, trans) in enumerate(question_items):
        # First 3: En -> Ru
        if i < 3:
            q_type = "en_ru"
            question_text = word
            correct_answer = trans
            # Distractors: other translations
            distractors = [r[1] for r in rows if r[1] != trans]
        else:
            # Next 3: Ru -> En
            q_type = "ru_en"
            question_text = trans
            correct_answer = word
            # Distractors: other words
            distractors = [r[0] for r in rows if r[0] != word]
            
        # Pick 2 random distractors (or fewer if not enough)
        if len(distractors) > 2:
            opts = random.sample(distractors, 2)
        else:
            opts = distractors
            
        options = opts + [correct_answer]
        random.shuffle(options)
        
        questions.append({
            "type": q_type,
            "question": question_text,
            "correct": correct_answer,
            "options": options
        })
        
    USER_CHAT_SESSIONS[user_id] = {
        "type": "word_training",
        "questions": questions,
        "current_index": 0,
        "score": 0
    }
    
    await c.answer()
    await send_training_question(c.message, user_id)


async def send_training_question(message: types.Message, user_id: int):
    session = USER_CHAT_SESSIONS.get(user_id)
    if not session or session.get("type") != "word_training":
        await message.edit_text("Ошибка сессии. Попробуй заново.", reply_markup=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]]))
        return
        
    idx = session["current_index"]
    questions = session["questions"]
    
    if idx >= len(questions):
        # Finish
        score = session["score"]
        total = len(questions)
        
        praise = "Отличная работа! 🎉"
        if score == total:
            praise = "Идеально! 🏆 Ты молодец!"
        elif score > total / 2:
            praise = "Хороший результат! 👍"
            
        text = (
            f"{praise}\n\n"
            f"Результат: <b>{score} из {total}</b>"
        )
        
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton("Тренировать еще 🧠", callback_data="mode:train_words")],
            [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
        ])
        
        await message.edit_text(text, reply_markup=kb)
        log_event(user_id, "word_training_completed", {"score": score, "total": total})
        return
        
    q = questions[idx]
    q_text = q["question"]
    
    # Title based on type
    if q["type"] == "en_ru":
        title = f"Как переводится: <b>{q_text}</b>?"
    else:
        title = f"Как будет на английском: <b>{q_text}</b>?"
        
    kb = InlineKeyboardMarkup(row_width=1)
    for opt in q["options"]:
        # We pass the index of the option in the list to verify answer? 
        # Or just pass "correct" / "incorrect"?
        # Passing full text might be too long for callback_data (64 bytes limit).
        # Let's pass 1 if correct, 0 if incorrect.
        is_correct = "1" if opt == q["correct"] else "0"
        # We need to handle potential duplicates in options if dictionary is small?
        # Logic above ensures options are unique if source rows are unique.
        
        # Truncate option text for button label if too long
        label = opt[:30] + "..." if len(opt) > 30 else opt
        kb.add(InlineKeyboardButton(label, callback_data=f"train:ans:{is_correct}"))
        
    kb.add(InlineKeyboardButton("Меню 🏠", callback_data="menu:main"))
    
    step_info = f"Вопрос {idx + 1} из {len(questions)}"
    full_text = f"{step_info}\n\n{title}"
    
    await message.edit_text(full_text, reply_markup=kb)


@dp.callback_query_handler(lambda c: c.data.startswith("train:ans:"))
async def handle_training_answer(c: types.CallbackQuery):
    user_id = c.from_user.id
    session = USER_CHAT_SESSIONS.get(user_id)
    if not session or session.get("type") != "word_training":
        await c.answer("Сессия истекла.", show_alert=True)
        await c.message.edit_text("Сессия истекла.", reply_markup=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]]))
        return
        
    is_correct = c.data.split(":")[2] == "1"
    
    if is_correct:
        session["score"] += 1
        await c.answer("Верно! ✅")
    else:
        # Show correct answer? 
        # For now just move on, maybe show alert
        q = session["questions"][session["current_index"]]
        correct = q["correct"]
        await c.answer(f"Неверно ❌\nПравильно: {correct}", show_alert=True)
        
    session["current_index"] += 1
    await send_training_question(c.message, user_id)


# --- Text message handler (must be last!) ---
@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def handle_text_message(m: types.Message):
    """
    Handle all text messages that are not commands.
    Processes:
    1. Dictionary additions (word + "словарь" or /word)
    2. Roleplay conversations
    3. General chat with GPT
    """
    user_id = m.from_user.id
    text = (m.text or "").strip()
    
    if not text:
        return
    
    # Skip if it's a command (starts with /)
    if text.startswith("/"):
        # Commands are handled by specific handlers, but /word is special
        if text.lower().startswith("/word"):
            handled = await maybe_add_to_dictionary(m)
            if handled:
                return
        return
    
    # Check for dictionary addition ("словарь" at end)
    handled = await maybe_add_to_dictionary(m)
    if handled:
        return
    
    # Save user message
    save_msg(user_id, "user", text)
    
    # Check if user is in a session
    session = USER_CHAT_SESSIONS.get(user_id)
    logging.info(f"[handle_text_message] user={user_id} text='{text[:50]}' session={session.get('type') if session else None}")
    
    # Handle "bye" command universally for any active session
    if text.lower() in ["bye", "goodbye", "пока", "выход"]:
        if session:
            session_type = session.get("type")
            if session_type == "news":
                log_event(user_id, "reading_closed", {"cache_id": session.get("cache_id")})
            elif session_type == "roleplay":
                log_event(user_id, "chat_closed", {"topic": session.get("topic")})
            USER_CHAT_SESSIONS.pop(user_id, None)
            
            # Update streak and check if we should show notification
            streak, is_new_day = update_streak(user_id)
            show_notification = is_new_day and should_show_streak_notification(user_id)
            
            if show_notification:
                # Show streak notification
                mark_streak_notified(user_id)
                
                streak_emoji = "🔥" * min(streak, 5)  # Show up to 5 fire emojis
                await m.answer(
                    f"🎉 <b>Отличная работа!</b>\n\n"
                    f"{streak_emoji} Победная серия: <b>{streak} {get_day_word(streak)}</b>\n\n"
                    f"Тренируйся ежедневно и общайся как носитель! 💪"
                )
                
                # Wait before showing menu
                await asyncio.sleep(2)
            
            await m.answer(
                "Хорошая работа! Возвращаю тебя в меню 🏠",
                reply_markup=mode_keyboard()
            )
            return
    
    if session and session.get("type") == "news":
        # Handle news discussion
        await handle_news_discussion(m, session)
        return
    
    if session and session.get("type") == "roleplay":
        # Handle roleplay conversation
        await handle_roleplay_message(m, session)
        return
    
    # Default: general chat with GPT
    await handle_general_chat(m)


async def handle_roleplay_message(m: types.Message, session: dict):
    """Handle message in roleplay mode."""
    user_id = m.from_user.id
    text = m.text.strip()
    topic_key = session.get("topic", "free")
    
    # Check chat message limit BEFORE processing
    paid = is_paid_user(user_id)
    
    if not paid:
        chat_messages_today = get_user_chat_messages_count_today(user_id)
        logging.info(f"[handle_roleplay] user={user_id} paid={paid} chat_messages_today={chat_messages_today} limit={FREE_CHAT_MESSAGES_LIMIT}")
        
        if chat_messages_today >= FREE_CHAT_MESSAGES_LIMIT:
            increment_user_counter(user_id, "paywall_shown")
            log_event(user_id, "paywall_shown", {"reason": "chat_limit", "count": chat_messages_today})
            logging.info(f"[handle_roleplay] PAYWALL TRIGGERED for user={user_id}")
            
            # End the session
            USER_CHAT_SESSIONS.pop(user_id, None)
            
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton("Приобрести доступ 💎", callback_data="profile_buy_unlimited")],
                [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
            ])
            await m.answer(
                "🔒 Ты отправил 10 сообщений в разговорной практике сегодня!\n\n"
                "Чтобы продолжить без ограничений, приобрети безлимитный доступ 💎",
                reply_markup=kb
            )
            return
    
    # Check if user wants to end the session
    if text.lower() in ["bye", "goodbye", "пока", "выход"]:
        completed = session.get("completed_count", 0)
        USER_CHAT_SESSIONS.pop(user_id, None)
        
        # Update streak and check if we should show notification
        streak, is_new_day = update_streak(user_id)
        show_notification = is_new_day and should_show_streak_notification(user_id)
        
        if show_notification:
            # Show streak notification
            mark_streak_notified(user_id)
            
            streak_emoji = "🔥" * min(streak, 5)  # Show up to 5 fire emojis
            await m.answer(
                f"🎉 <b>Отличная работа!</b>\n\n"
                f"{streak_emoji} Победная серия: <b>{streak} {get_day_word(streak)}</b>\n\n"
                f"Тренируйся ежедневно и общайся как носитель! 💪"
            )
            
            # Wait before showing menu
            await asyncio.sleep(2)
        
        await m.answer(
            f"Диалог завершён! 👋\n\nВыполнено заданий: {completed}\n\nВозвращайся, когда захочешь попрактиковаться ещё!",
            reply_markup=mode_keyboard()
        )
        return
    
    # Check task completion
    tasks = session.get("tasks", [])
    for task in tasks:
        if not task.get("done"):
            result = await check_task_completion(text, task["text"])
            print(f"[roleplay] Task check: user={user_id}, task='{task['text']}', result={result}", flush=True)
            if result.get("done"):
                task["done"] = True
                session["completed_count"] = session.get("completed_count", 0) + 1
                print(f"[roleplay] Task completed! user={user_id}, completed_count={session['completed_count']}, task='{task['text']}'", flush=True)
                # Log task completion event
                log_event(user_id, "task_completed", {
                    "topic": topic_key,
                    "task": task["text"],
                    "completed_count": session["completed_count"]
                })
                break
    
    # Increment turn counter
    session["turns"] = session.get("turns", 0) + 1
    
    # Build context for GPT
    persona = PERSONA_PROMPTS.get(topic_key, PERSONA_PROMPTS.get("free"))
    
    # Get recent messages from DB for context
    with closing(db()) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE user_id=%s ORDER BY id DESC LIMIT 10",
            (user_id,)
        )
        rows = cur.fetchall()
    
    messages = [{"role": "system", "content": persona + "\n\nKeep your responses concise (2-3 sentences). Correct any grammar mistakes the user makes, using this format: 🔴 original → ✅ corrected — краткое объяснение на русском языке (1 предложение)."}]
    
    # Add conversation history (reversed to chronological order)
    for row in reversed(rows):
        role = "assistant" if row[0] == "assistant" else "user"
        messages.append({"role": role, "content": row[1]})
    
    # Generate response
    try:
        response = await gpt_chat(messages)
    except Exception:
        logging.exception("GPT chat failed in roleplay")
        response = "Sorry, I'm having trouble responding right now. Please try again."
    
    # Save assistant response
    save_msg(user_id, "assistant", response)
    
    # Check if 3 tasks completed
    completed_count = session.get("completed_count", 0)
    
    # Build response with task status
    pending_tasks = [t for t in tasks if not t.get("done")][:3]
    
    emoji = persona_emoji(topic_key)
    full_response = f"{emoji} {response}"
    
    # Always send the GPT response first
    if pending_tasks and completed_count < 3:
        tasks_text = "\n".join([f"• {t['text']}" for t in pending_tasks])
        full_response += f"\n\n<i>Осталось:</i>\n{tasks_text}"
    
    kb = InlineKeyboardMarkup().add(InlineKeyboardButton("Перевести 🔁", callback_data="translate:chat"))
    await m.answer(full_response, reply_markup=kb)
    
    # Then send completion message if 3 tasks done (only once)
    if completed_count >= 3 and not session.get("completion_shown"):
        session["completion_shown"] = True
        log_event(user_id, "topic_completed", {"topic": topic_key, "turns": session.get("turns", 0)})
        
        completion_kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
        ])
        await m.answer(
            "🎉 <b>Отличная работа! Ты выполнил(а) 3 задания.</b>\n\n"
            "Ты можешь продолжить диалог или вернуться в меню.",
            reply_markup=completion_kb
        )


async def handle_news_discussion(m: types.Message, session: dict):
    """Handle message in news discussion mode (after reading an article)."""
    user_id = m.from_user.id
    text = m.text.strip()
    cache_id = session.get("cache_id")
    
    # Increment answer count
    session["answers_count"] = session.get("answers_count", 0) + 1
    answers_count = session["answers_count"]
    
    # Get article info for context
    with closing(db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT title, summary, questions FROM news_cache WHERE id=%s", (cache_id,))
        row = cur.fetchone()
    
    if not row:
        await m.answer("Не удалось найти статью. Попробуй /news для новой статьи.")
        USER_CHAT_SESSIONS.pop(user_id, None)
        return
    
    title, summary, questions_json = row
    questions = json.loads(questions_json or "[]")
    current_q_index = session.get("last_q_index", 0)
    current_question = questions[current_q_index] if current_q_index < len(questions) else ""
    
    # Get user level
    user = get_user(user_id)
    level = user.get("level", "B1") if user else "B1"
    
    # Build context for GPT
    system_prompt = f"""You are an English tutor discussing a news article with a student.
Article title: {title}
Article summary: {summary}
Current question being discussed: {current_question}

Your task:
1. First, correct any grammar or vocabulary mistakes in the student's response (use format: 🔴 original → ✅ corrected — краткое объяснение на русском языке)
2. Then respond naturally to their answer, asking a follow-up question to keep the conversation going.
3. Keep responses concise (2-3 sentences after corrections).
4. Adapt your language to {level} level.

IMPORTANT: Do NOT correct punctuation, capitalization, or contractions. Only correct actual grammar and vocabulary errors."""

    # Get recent messages for context
    with closing(db()) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE user_id=%s ORDER BY id DESC LIMIT 10",
            (user_id,)
        )
        rows = cur.fetchall()
    
    messages = [{"role": "system", "content": system_prompt}]
    for row in reversed(rows):
        role = "assistant" if row[0] == "assistant" else "user"
        messages.append({"role": role, "content": row[1]})
    
    # Generate response
    try:
        response = await gpt_chat(messages)
    except Exception:
        logging.exception("GPT chat failed in news discussion")
        response = "Interesting point! Could you tell me more about what you think?"
    
    # Save assistant response
    save_msg(user_id, "assistant", response)
    
    # Always send the GPT response first
    kb = InlineKeyboardMarkup().add(InlineKeyboardButton("Перевести 🔁", callback_data="translate:chat"))
    await m.answer(response, reply_markup=kb)
    
    # Then send completion message if 3 answers done (only once)
    if answers_count >= 3 and not session.get("completion_shown"):
        session["completion_shown"] = True
        log_event(user_id, "reading_completed", {"cache_id": cache_id})
        
        completion_kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
        ])
        await m.answer(
            "🎉 <b>Отличная работа! Ты ответил(а) на 3 вопроса.</b>\n\n"
            "Ты можешь продолжить диалог или вернуться в меню.",
            reply_markup=completion_kb
        )


async def handle_general_chat(m: types.Message):
    """Handle general chat (not in roleplay mode)."""
    user_id = m.from_user.id
    
    # Check chat message limit BEFORE processing
    paid = is_paid_user(user_id)
    
    if not paid:
        chat_messages_today = get_user_chat_messages_count_today(user_id)
        logging.info(f"[handle_general_chat] user={user_id} paid={paid} chat_messages_today={chat_messages_today} limit={FREE_CHAT_MESSAGES_LIMIT}")
        
        if chat_messages_today >= FREE_CHAT_MESSAGES_LIMIT:
            increment_user_counter(user_id, "paywall_shown")
            log_event(user_id, "paywall_shown", {"reason": "chat_limit", "count": chat_messages_today})
            logging.info(f"[handle_general_chat] PAYWALL TRIGGERED for user={user_id}")
            
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton("Приобрести доступ 💎", callback_data="profile_buy_unlimited")],
                [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")]
            ])
            await m.answer(
                "🔒 Ты отправил 10 сообщений в разговорной практике сегодня!\n\n"
                "Чтобы продолжить без ограничений, приобрети безлимитный доступ 💎",
                reply_markup=kb
            )
            return
    
    # Get user info for context
    user = get_user(user_id)
    level = user.get("level", "B1") if user else "B1"
    
    # Get recent messages from DB
    with closing(db()) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE user_id=%s ORDER BY id DESC LIMIT 20",
            (user_id,)
        )
        rows = cur.fetchall()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (reversed to chronological order)
    for row in reversed(rows):
        role = "assistant" if row[0] == "assistant" else "user"
        messages.append({"role": role, "content": row[1]})
    
    # Generate response
    try:
        response = await gpt_chat(messages)
    except Exception:
        logging.exception("GPT chat failed")
        response = "Sorry, I'm having trouble right now. Try /news for a fresh article or /menu for options. 🤖"
    
    # Save assistant response
    save_msg(user_id, "assistant", response)
    
    kb = InlineKeyboardMarkup().add(InlineKeyboardButton("Перевести 🔁", callback_data="translate:chat"))
    await m.answer(response, reply_markup=kb)


# --- YooKassa Webhook Handler ---

from aiohttp import web

async def yookassa_webhook(request):
    """Handle YooKassa webhook notifications for automatic payment confirmation."""
    try:
        # Get JSON data from request
        data = await request.json()
        
        event_type = data.get("event")
        payment_data = data.get("object")
        
        logging.info(f"Webhook received: event={event_type}, payment_id={payment_data.get('id') if payment_data else 'unknown'}")
        
        if event_type == "payment.succeeded" and payment_data:
            payment_id = payment_data.get("id")
            metadata = payment_data.get("metadata", {})
            user_id = metadata.get("user_id")
            plan = metadata.get("plan")
            
            if user_id and plan:
                user_id = int(user_id)
                
                # Update payment status in DB
                with closing(db()) as conn:
                    c = conn.cursor()
                    c.execute("""
                        UPDATE payments 
                        SET status = 'succeeded', paid_at = now()
                        WHERE payment_id = %s
                    """, (payment_id,))
                    conn.commit()
                
                # Activate subscription
                activate_subscription(user_id, plan)
                
                # Notify user via bot
                try:
                    await bot.send_message(
                        user_id,
                        "✅ <b>Оплата прошла успешно!</b>\n\n"
                        "🎉 Подписка активирована!\n\n"
                        "Теперь у тебя безлимитный доступ:\n"
                        "✅ Неограниченные статьи\n"
                        "✅ Неограниченные игры\n"
                        "✅ Неограниченная разговорная практика\n\n"
                        "Спасибо за поддержку проекта! 💙"
                    )
                except Exception as e:
                    logging.exception(f"Failed to send notification to user {user_id}: {e}")
                    
                logging.info(f"Subscription activated via webhook for user {user_id}, plan: {plan}")
            else:
                logging.warning(f"Missing user_id or plan in payment metadata: {metadata}")
                    
        return web.Response(status=200, text="OK")
        
    except Exception as e:
        logging.exception(f"Webhook error: {e}")
        return web.Response(status=400, text="Bad Request")


async def on_startup(dp):
    """Initialize webhook on startup."""
    webhook_path = "/yookassa/webhook"
    webhook_url = os.getenv("WEBHOOK_URL")  # e.g., https://yourdomain.com/yookassa/webhook
    
    if webhook_url:
        logging.info(f"Setting up webhook at {webhook_url}")
        # Note: You'll need to configure webhook URL in YooKassa dashboard
    else:
        logging.warning("WEBHOOK_URL not set, webhook will not be configured")


async def on_shutdown(dp):
    """Cleanup on shutdown."""
    logging.info("Shutting down...")


if __name__ == '__main__':
    # Initialize database tables
    init_db()
    # Ensure game tables exist
    init_game_tables()
    
    # Check if we should run with webhook
    use_webhook = os.getenv("USE_WEBHOOK", "false").lower() == "true"
    
    if use_webhook:
        # Production mode: run polling for Telegram + webhook server for YooKassa
        from aiohttp import web
        import asyncio
        
        logging.info("Starting bot in HYBRID mode (polling for Telegram + webhook for YooKassa)")
        
        # Create aiohttp app for YooKassa webhook
        app = web.Application()
        app.router.add_post('/yookassa/webhook', yookassa_webhook)
        
        # Start Telegram polling in background
        async def start_bot_polling():
            await dp.skip_updates()
            await dp.start_polling()
        
        async def start_all():
            # Start polling in background task
            asyncio.create_task(start_bot_polling())
            
            # Start webhook server
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8443)))
            await site.start()
            
            logging.info(f"Webhook server started on port {os.getenv('PORT', 8443)}")
            
            # Keep running
            while True:
                await asyncio.sleep(3600)
        
        # Run everything
        asyncio.run(start_all())
    else:
        # Local development: polling only
        logging.info("Starting bot in POLLING mode (local development)")
        executor.start_polling(dp, skip_updates=True)

