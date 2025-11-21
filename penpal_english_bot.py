import uuid
# --- Unified Event Logging (new structure) ---
from threading import Lock
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
from datetime import date
# penpal_english_bot.py
import os
import json
import logging
import psycopg2
import psycopg2.extras
import random
from datetime import datetime, date
from contextlib import closing
import requests
from bs4 import BeautifulSoup
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, LabeledPrice
from aiogram.utils import executor
from dotenv import load_dotenv
import openai
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
    logging.warning("zoneinfo not available; timezone features will be limited")
import asyncio

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
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
            {"id": 1, "text": "Introduce yourself briefly (name, current job or study)", "keywords": ["i am", "my name", "i'm", "i am a"]},
            {"id": 2, "text": "Explain why you want this job", "keywords": ["because", "i want", "interested", "why i want"]},
            {"id": 3, "text": "Describe one strength", "keywords": ["strength", "skill", "i can", "i am good at", "my strength"]},
            {"id": 4, "text": "Ask a question about the company", "keywords": ["what", "company", "position", "role", "could you tell"]},
        ]
    elif topic == "restaurant":
        tasks = [
            {"id": 1, "text": "Order a main dish and a drink", "keywords": ["i'll have", "i would like", "could i have", "i want"]},
            {"id": 2, "text": "Ask about allergens or dietary restrictions", "keywords": ["allerg", "gluten", "vegan", "vegetarian", "contains"]},
            {"id": 3, "text": "Ask for the bill", "keywords": ["check", "bill", "the bill", "can i pay", "pay"]},
        ]
    elif topic == "raise":
        tasks = [
            {"id": 1, "text": "Ask your manager for a raise and state your achievements", "keywords": ["raise", "salary", "i have achieved", "increase", "promotion", "i deserve"]},
            {"id": 2, "text": "Propose a salary number or range", "keywords": ["salary", "per month", "per year", "amount", "rub", "$", "€"]},
            {"id": 3, "text": "Ask about next steps", "keywords": ["next steps", "when will i know", "follow up"]},
        ]
    elif topic == "travel":
        tasks = [
            {"id": 1, "text": "Ask the travel agent about price and available dates", "keywords": ["price", "cost", "how much", "when", "dates"]},
            {"id": 2, "text": "Ask what services are included (hotel, transfers)", "keywords": ["hotel", "transfer", "included", "meals", "flight"]},
            {"id": 3, "text": "Request a cheaper option or ask about discounts", "keywords": ["discount", "cheaper", "alternative", "other options"]},
        ]
    else:  # free
        tasks = [
            {"id": 1, "text": "Say hello and ask how the other person is doing", "keywords": ["hello", "hi", "how are you", "how's it going"]},
            {"id": 2, "text": "Share something about your day", "keywords": ["today", "i went", "i saw", "my day", "i did"]},
            {"id": 3, "text": "Ask a question back", "keywords": ["what about you", "and you", "do you", "tell me"]},
        ]
    # mark all as not done
    for t in tasks:
        t["done"] = False
    return tasks


# Persona instructions used for roleplay per topic. Keep them short; actual replies are generated by the model.
PERSONA_PROMPTS = {
    "interview": "You are a hiring manager conducting a short job interview. Speak as a polite, professional manager in English. Introduce yourself briefly (one short paragraph) and then ask a specific interview question to start. Be concise.",
    "restaurant": "You are a friendly restaurant waiter. Greet the customer in English, introduce the restaurant briefly, and ask what they would like to order. Keep it casual and helpful.",
    "raise": "You are the user's manager. Start the conversation professionally in English: introduce yourself as the manager, ask why the employee thinks they deserve a raise, and ask for specifics.",
    "travel": "You are a travel agent. Greet the customer in English, introduce services briefly, and ask about destination and travel dates.",
    "free": "You are a friendly conversation partner (philosopher/mentor). Greet the user in English, introduce yourself briefly, and ask an open question to start a thoughtful chat.",
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
        # include emoji and send as a natural reply
        await bot.send_message(user_id, f"{emoji} {text}")
    except Exception:
        logging.exception("Failed to send delayed assistant intro")


async def check_task_completion(user_text: str, task_text: str) -> dict:
    """Ask the language model whether the user's reply completes the task.
    Returns a dict with keys: done (bool) and explanation (short Russian string).
    Falls back to a heuristic if the model call fails.
    """
    try:
        prompt = (
            "You are an objective evaluator.\n"
            "Determine whether the user's reply fulfills the short task below.\n\n"
            f"Task: {task_text}\n\n"
            f"User reply: {user_text}\n\n"
            "Answer with strict JSON only, no extra text.\n"
            "Format: {\"done\": true|false, \"explanation\": \"one short sentence in Russian\"}"
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You return strict JSON only."}, {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp.choices[0].message["content"]
        logging.debug(f"Task check raw response: {text}")
        try:
            data = json.loads(text)
            # ensure keys
            return {"done": bool(data.get("done")), "explanation": str(data.get("explanation", ""))}
        except Exception:
            # try to extract a JSON-like substring
            import re

            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    data = json.loads(m.group(0))
                    return {"done": bool(data.get("done")), "explanation": str(data.get("explanation", ""))}
                except Exception:
                    logging.exception("Failed to parse JSON from model task-check response")
    except Exception:
        logging.exception("check_task_completion OpenAI call failed")

    # Fallback heuristic: simple substring match of 2-3 important words from task_text
    try:
        lowered = (user_text or "").lower()
        words = [w.strip('.,?!') for w in task_text.split() if len(w) > 3][:3]
        hits = sum(1 for w in words if w.lower() in lowered)
        if hits >= 1:
            return {"done": True, "explanation": "(эвристика) найдено ключевое слово"}
    except Exception:
        pass
    return {"done": False, "explanation": "(эвристика) не выполнено"}





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
import uuid
import threading
from collections import defaultdict
from datetime import timedelta
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

def get_user_article_count(user_id):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("SELECT daily_articles, last_article_reset FROM users WHERE id=%s", (user_id,))
        row = c.fetchone()
    return row

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
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages(user_id, role, content, created_at) VALUES(%s, %s, %s, %s)",
            (user_id, role, content, now),
        )
        # update last interaction on users table
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
            [InlineKeyboardButton("Обсудить статью 📰", callback_data="mode:news")],
            [InlineKeyboardButton("Свободный разговор 💬", callback_data="mode:chat")],
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
        [InlineKeyboardButton("Обсуждение свежих новостей �️", callback_data="onboard:interest:news")],
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
                    InlineKeyboardButton("Завершил(а) ✅", callback_data=f"news:done:{cache_id}"),
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
            "Супер, ты на шаг ближе к цели 🎯\n\nПеред тем как начнем, расскажи немного о себе:\n\n<b>Какая твоя главная цель в изучении английского?</b>",
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
        "Спасибо!\n\n<b>Что для тебя наиболее интересно?</b>\n\nВыбери, что хочется попробовать:",
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
        "Отлично, я с радостью помогу тебе🙌\nКакой формат тебе сейчас больше всего подходит?",
        reply_markup=onboarding_interest_kb()
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
        f"Отлично! Уровень установлен: <b>{level}</b> 🎯\n\nЧто ты хочешь сделать дальше?",
        reply_markup=mode_keyboard(),
    )


@dp.callback_query_handler(lambda c: c.data.startswith("mode:"))
async def choose_mode(c: types.CallbackQuery):
    log_event(c.from_user.id, "mode_selected", {"mode": c.data.split(":")[1]})
    save_msg(c.from_user.id, "user", c.data)
    user_id = c.from_user.id
    mode = c.data.split(":")[1]
    if mode not in {"news", "chat"}:
        await c.answer("Неизвестный режим.", show_alert=True)
        return
    # Switching between news/chat should drop any previous chat topic state
    USER_CHAT_SESSIONS.pop(user_id, None)
    set_user_mode(user_id, mode)
    await c.answer()
    if mode == "news":
        user = get_user(user_id)
        existing = []
        if user and user.get("topics"):
            existing = [t.strip() for t in (user.get("topics") or "").split(",") if t.strip()]

        # If topics already chosen before, don't force selection every time;
        # just bring a new article based on saved interests.
        if existing:
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
    # start a session with 2 required tasks to complete
    tasks = make_tasks_for_topic(topic_key)
    # we will require 2 tasks to be completed (or all if fewer)
    USER_CHAT_SESSIONS[user_id] = {
        "topic": topic_key,
        "tasks": tasks,
        "completed_count": 0,
        "turns": 0,
    }
    # show rules and first tasks
    await c.answer()
    intro = f"Тема: {topic_name}\n\nПравила: Выполни 2 задания или скажи bye 👋, чтобы завершить диалог."
    tasks_list = "\n".join([f"{t['id']}) {t['text']}" for t in tasks[:3]])
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

    # Send first message: topic, rules and tasks
    await c.message.edit_text(intro + "\n\nЗадания:\n" + tasks_list)
    # Send assistant intro as a separate message after 10 seconds without the word 'Bot' and with emoji
    try:
        asyncio.create_task(send_assistant_intro_delayed(c.from_user.id, assistant_intro, topic_key, delay=10))
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
        f"Готово — по твоему выбору ({count} слов) уровень определён как <b>{level}</b>.\n\nЧто хочешь сделать дальше?",
        reply_markup=mode_keyboard(),
    )


@dp.callback_query_handler(lambda c: c.data.startswith("news:more"))
async def more_news(c: types.CallbackQuery):
    user_id = c.from_user.id
    log_event(user_id, "news_requested", {})
    save_msg(user_id, "user", c.data)
    increment_user_article_count(user_id)
    count_row = get_user_article_count(user_id)
    today = date.today()
    daily_articles = count_row[0] if count_row else 0
    last_reset = count_row[1] if count_row else None
    if last_reset != today:
        daily_articles = 0

    paid = is_paid_user(user_id)
    try:
        logging.info(f"Paywall check [more]: user={user_id} daily_articles={daily_articles} last_reset={last_reset} paid={paid} limit={FREE_ARTICLE_LIMIT}")
    except Exception:
        pass

    if not paid and daily_articles > FREE_ARTICLE_LIMIT:
        # Increment paywall_shown counter
        increment_user_counter(user_id, "paywall_shown")
        try:
            await c.answer("Вы достигли лимита бесплатных статей на сегодня.", show_alert=True)
        except Exception:
            pass
        kb = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton("Оформить подписку ⭐️", callback_data="pay:subscribe")]]
        )
        await bot.send_message(user_id, "Оформите подписку для неограниченного доступа к статьям.", reply_markup=kb)
        return

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
        "Отлично — ты прочитал(а) статью! Я задам по одному вопросу за раз.\n"
        "Ответь, когда будешь готов(а), или нажми «Другой вопрос», чтобы увидеть другой вопрос.\n\n"
    )
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton("Другой вопрос 🔁", callback_data=f"news:next:{cache_id}:1")],
            [InlineKeyboardButton("Поменять статью 🔁", callback_data="news:more")],
            [InlineKeyboardButton("Меню 🏠", callback_data="menu:main")],
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
    kb_buttons.append(InlineKeyboardButton("Поменять статью 🔁", callback_data="news:more"))
    kb_buttons.append(InlineKeyboardButton("Меню 🏠", callback_data="menu:main"))
    kb = InlineKeyboardMarkup(inline_keyboard=[kb_buttons])
    await bot.send_message(c.from_user.id, q, reply_markup=kb)

@dp.callback_query_handler(lambda c: c.data == "menu:main")
async def menu_main_callback(c: types.CallbackQuery):
    await c.answer()
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
    USER_CHAT_SESSIONS.pop(c.from_user.id, None)


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

    increment_user_article_count(user_id)
    count_row = get_user_article_count(user_id)
    today = date.today()
    daily_articles = count_row[0] if count_row else 0
    last_reset = count_row[1] if count_row else None
    if last_reset != today:
        daily_articles = 0

    paid = is_paid_user(user_id)
    try:
        logging.info(f"Paywall check [/news]: user={user_id} daily_articles={daily_articles} last_reset={last_reset} paid={paid} limit={FREE_ARTICLE_LIMIT}")
    except Exception:
        pass

    if not paid and daily_articles > FREE_ARTICLE_LIMIT:
        increment_user_counter(user_id, "paywall_shown")
        kb = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton("Оформить подписку ⭐️", callback_data="pay:subscribe")]]
        )
        await m.answer("Вы достигли лимита бесплатных статей на сегодня.", reply_markup=kb)
        return

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
    session_id = get_session_id(m.from_user.id)
    log_event(m.from_user.id, "command_used", {"command": "/menu"})
    # From the main menu there should be no active chat topic session
    USER_CHAT_SESSIONS.pop(m.from_user.id, None)
    await m.answer(
        "Меню активности — выбери, что хочешь сделать:",
        reply_markup=mode_keyboard()
    )


@dp.message_handler(commands=["subscribe", "premium"])
async def cmd_subscribe(m: types.Message):
    save_msg(m.from_user.id, "user", "/subscribe")
    session_id = get_session_id(m.from_user.id)
    log_event(m.from_user.id, session_id, "command_used", {"command": "/subscribe"})
    if not PAYMENTS_PROVIDER_TOKEN:
        await m.answer("Платежи временно недоступны. Свяжитесь с поддержкой или попробуйте позже.")
        return
    # Build prices in minor units (kopeks/cents)
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
            m.chat.id,
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
    except Exception:
        logging.exception("Failed to send invoice")
        await m.answer("Не удалось сформировать счёт. Попробуйте позже.")


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


@dp.message_handler()
async def chat(m: types.Message):
    log_event(m.from_user.id, "user_message", {"text": m.text})
    user = get_user(m.from_user.id)
    if not user or not user.get("level"):
        save_user(m.from_user.id, m.from_user.username or "")
        await m.answer("Let’s set your level first:", reply_markup=level_keyboard())
        return
    # If there is an active chat session with tasks, handle it here
    session = USER_CHAT_SESSIONS.get(m.from_user.id)
    if session:
        session["turns"] += 1
        text = (m.text or "").lower()
        # immediate exit
        if "bye" in text or "bye 👋" in text:
            # If user completed at least one task, consider topic completed as well
            if session.get("completed_count", 0) > 0:
                log_event(
                    m.from_user.id,
                    "topic_completed",
                    {"topic": session.get("topic"), "completed_tasks": session.get("completed_count")},
                )
            USER_CHAT_SESSIONS.pop(m.from_user.id, None)
            await m.answer("Диалог завершён. Возвращаю тебя в меню активности.", reply_markup=mode_keyboard())
            return

        # check each task for completion using the language model
        tasks = session.get("tasks", [])
        newly_completed = []
        for t in tasks:
            if not t.get("done"):
                try:
                    res = await check_task_completion(m.text or "", t.get("text", ""))
                    if res.get("done"):
                        t["done"] = True
                        session["completed_count"] += 1
                        newly_completed.append({"task": t, "explanation": res.get("explanation")})
                except Exception:
                    logging.exception("Error while checking task completion")

        # send feedback for completed tasks and log events
        if newly_completed:
            for item in newly_completed:
                t = item.get("task")
                expl = item.get("explanation") or ""
                log_event(
                    m.from_user.id,
                    "task_completed",
                    {"topic": session.get("topic"), "task_id": t.get("id"), "task_text": t.get("text")},
                )
                await m.answer(f"Отлично — задание выполнено: {t['text']}\n{expl}")

        # check for completion criteria
        if session["completed_count"] >= 2:
            # topic completed (required tasks done)
            log_event(
                m.from_user.id,
                "topic_completed",
                {"topic": session.get("topic"), "completed_tasks": session.get("completed_count")},
            )
            USER_CHAT_SESSIONS.pop(m.from_user.id, None)
            await m.answer(
                "Поздравляю — ты выполнил(а) необходимые задания! Возвращаю тебя в меню активности.",
                reply_markup=mode_keyboard(),
            )
            return

        if session["turns"] >= 20:
            # If user completed at least one task before timeout, mark topic as completed
            if session.get("completed_count", 0) > 0:
                log_event(
                    m.from_user.id,
                    "topic_completed",
                    {"topic": session.get("topic"), "completed_tasks": session.get("completed_count")},
                )
            USER_CHAT_SESSIONS.pop(m.from_user.id, None)
            await m.answer(
                "Диалог окончен (лимит реплик достигнут). Возвращаю тебя в меню активности.",
                reply_markup=mode_keyboard(),
            )
            return

        # Otherwise, ask the language model to (a) provide brief corrections, (b) continue the roleplay persona
        persona_intro = session.get("assistant_intro", PERSONA_PROMPTS.get(session.get("topic"), "You are a friendly conversational partner."))
        try:
            # Build messages: system=persona instruction, user=the user's reply + short task status
            sys_msg = persona_intro
            user_msg = (
                f"User reply: {m.text}\n\nTasks completed so far: {session['completed_count']} of 2.\n"
                "Please do two things in one concise response:\n"
                "1) If the user's English contains mistakes, show up to 3 inline corrections using this format (use HTML tags as shown):\n"
                "- 🔴 <i>original</i> → ✅ <b><u>corrected</u></b> — one short reason in English\n"
                "Example: 🔴 <i>I has a dog</i> → ✅ <b><u>I have a dog</u></b> — subject-verb agreement\n"
                "2) Continue the roleplay as the persona (speak in English). First show corrections (if any), then a short assistant reply that continues the scene (one question or prompt). Keep the entire reply concise and in English."
            )
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                temperature=0.5,
            )
            assistant_next = resp.choices[0].message["content"]
        except Exception:
            logging.exception("Roleplay LM call failed; using fallback reply")
            assistant_next = "Спасибо — давай продолжим. Можешь ответить ещё?"

        await m.answer(assistant_next)
        return
    # Build short context (last ~6 turns)
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT role, content FROM messages WHERE user_id=%s ORDER BY id DESC LIMIT 6",
            (m.from_user.id,),
        )
        rows = c.fetchall()
    history = [{"role": r, "content": ct} for (r, ct) in rows[::-1]]
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": m.text}]
    )
    save_msg(m.from_user.id, "user", m.text)
    session_id = get_session_id(m.from_user.id)
    log_event(m.from_user.id, "user_message", {"text": m.text}, session_id=session_id)
    reply = await gpt_chat(messages)
    # No longer mining 'Useful:' phrases. Corrections are handled by the assistant per SYSTEM_PROMPT.
    save_msg(m.from_user.id, "assistant", reply)
    log_event(m.from_user.id, "assistant_message", {"text": reply}, session_id=session_id)
    await m.answer(reply)


if __name__ == "__main__":
    init_db()
    # start background daily sender task
    loop = asyncio.get_event_loop()
    # Removed daily_sender_loop: now handled by Heroku Scheduler
    executor.start_polling(dp, skip_updates=True)
