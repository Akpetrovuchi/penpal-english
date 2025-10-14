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
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
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
            last_interaction TEXT
        )
        """)
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
    return InlineKeyboardMarkup(inline_keyboard=rows)


def level_keyboard():
    levels = ["A2", "B1", "B2", "C1"]
    # top row: compact level buttons; second row: full-width "I don't know" button
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
    # two buttons per row
    for i in range(0, len(words), 2):
        row = []
        for w in words[i : i + 2]:
            mark = "✅ " if w in sel else ""
            row.append(InlineKeyboardButton(f"{mark}{w}", callback_data=f"word:toggle:{w}"))
        kb_rows.append(row)
    kb_rows.append([InlineKeyboardButton("Готово ✔️", callback_data="word:done")])
    return InlineKeyboardMarkup(inline_keyboard=kb_rows)


def mode_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton("Обсудить статью 📰", callback_data="mode:news")],
            [InlineKeyboardButton("Свободный разговор 💬", callback_data="mode:chat")],
        ]
    )


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
    save_user(m.from_user.id, m.from_user.username or "")
    # Reset topics for this user when they press /start
    try:
        set_user_topics(m.from_user.id, [])
        set_user_mode(m.from_user.id, None)
    except Exception:
        logging.exception("Failed to reset user topics on /start")
    try:
        await m.answer(
            "Привет! Я <b>PenPal English</b> 👋 — твой дружелюбный собеседник по английскому.\n\nКакой у тебя уровень?",
            reply_markup=level_keyboard(),
        )
    except Exception:
        logging.exception("Failed to send /start reply; falling back to safe message")
        # send a safe non-empty fallback so Telegram doesn't reject it
        try:
            await m.answer("Привет! Давай начнём. Выбери уровень:", reply_markup=level_keyboard())
        except Exception:
            logging.exception("Fallback /start reply also failed")


@dp.callback_query_handler(lambda c: c.data.startswith("level:"))
async def choose_level(c: types.CallbackQuery):
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
    user_id = c.from_user.id
    mode = c.data.split(":")[1]
    if mode not in {"news", "chat"}:
        await c.answer("Неизвестный режим.", show_alert=True)
        return
    set_user_mode(user_id, mode)
    await c.answer()
    if mode == "news":
        user = get_user(user_id)
        existing = []
        if user and user.get("topics"):
            existing = [t for t in user["topics"].split(",") if t]
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
        await c.message.edit_text("Отлично! Я принесу материал для обсуждения. Вот новость 📰:")
        await send_news(c.from_user.id)
        return
    if val in selected:
        selected = [t for t in selected if t != val]
    else:
        selected.append(val)
    set_user_topics(c.from_user.id, selected)
    await c.message.edit_reply_markup(reply_markup=topic_keyboard(selected))


@dp.callback_query_handler(lambda c: c.data.startswith("word:toggle:"))
async def toggle_word(c: types.CallbackQuery):
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
    await c.answer("Загружаю другую статью… ⏳")
    await send_news(c.from_user.id)


@dp.callback_query_handler(lambda c: c.data.startswith("ans:"))
async def answer_hint(c: types.CallbackQuery):
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
        ]
    )
    await bot.send_message(c.from_user.id, instr + q0, reply_markup=kb)


@dp.callback_query_handler(lambda c: c.data.startswith("news:next:"))
async def news_next(c: types.CallbackQuery):
    # callback format: news:next:<cache_id>:<index>
    parts = c.data.split(":")
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
    # Prepare next index (wrap or stop at end)
    next_idx = idx + 1
    kb_buttons = []
    if next_idx < len(questions):
        kb_buttons.append(
            InlineKeyboardButton(
                "Другой вопрос 🔁", callback_data=f"news:next:{cache_id}:{next_idx}"
            )
        )
    kb_buttons.append(InlineKeyboardButton("Поменять статью 🔁", callback_data="news:more"))
    kb = InlineKeyboardMarkup(inline_keyboard=[kb_buttons])
    await bot.send_message(c.from_user.id, q, reply_markup=kb)


@dp.message_handler(commands=["topics"])
async def cmd_topics(m: types.Message):
    user = get_user(m.from_user.id)
    current = (user.get("topics") or "").split(",") if user and user.get("topics") else []
    await m.answer("Update your interests 🌟:", reply_markup=topic_keyboard(current))


@dp.message_handler(commands=["stats"])
async def cmd_stats(m: types.Message):
    """Return basic usage statistics from the SQLite database.

    Queries performed:
    - total registered users
    - users with a saved level
    - active unique users in last 7 and 30 days (based on messages table)
    - total messages stored
    - users who used or selected news-related options (approximate)

    Note: accuracy depends on what's saved to `users` and `messages` tables. If you
    want more precise metrics (e.g. per-command tracking or Prometheus), we can add
    explicit instrumentation.
    """
    # Optional admin restriction via ADMIN_ID environment variable (comma-separated ids)
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
    await m.answer("Pick your level 🎯:", reply_markup=level_keyboard())


@dp.message_handler(commands=["news"])
async def cmd_news(m: types.Message):
    await send_news(m.from_user.id)


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
    await m.answer(
        "Try /news for a fresh topic 📰, /topics to change interests, /level to adjust difficulty, /review for phrases. Or just chat with me in English! 😊"
    )


@dp.message_handler(commands=["settz"])
async def cmd_settz(m: types.Message):
    """Set user timezone. Usage: /settz Europe/Moscow"""
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

        # send feedback for completed tasks
        if newly_completed:
            for item in newly_completed:
                t = item.get("task")
                expl = item.get("explanation") or ""
                await m.answer(f"Отлично — задание выполнено: {t['text']}\n{expl}")

        # check for completion criteria
        if session["completed_count"] >= 2:
            USER_CHAT_SESSIONS.pop(m.from_user.id, None)
            await m.answer(
                "Поздравляю — ты выполнил(а) необходимые задания! Возвращаю тебя в меню активности.",
                reply_markup=mode_keyboard(),
            )
            return

        if session["turns"] >= 20:
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
        rows = c.execute(
            "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT 6",
            (m.from_user.id,),
        ).fetchall()
    history = [{"role": r, "content": ct} for (r, ct) in rows[::-1]]
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": m.text}]
    )
    save_msg(m.from_user.id, "user", m.text)
    reply = await gpt_chat(messages)
    # No longer mining 'Useful:' phrases. Corrections are handled by the assistant per SYSTEM_PROMPT.
    save_msg(m.from_user.id, "assistant", reply)
    await m.answer(reply)


if __name__ == "__main__":
    init_db()
    # start background daily sender task
    loop = asyncio.get_event_loop()
    # Removed daily_sender_loop: now handled by Heroku Scheduler
    executor.start_polling(dp, skip_updates=True)
