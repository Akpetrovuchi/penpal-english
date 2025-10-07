# penpal_english_bot.py
import os, json, asyncio, logging, sqlite3, time, random
from datetime import datetime, timedelta
from contextlib import closing
import requests
from bs4 import BeautifulSoup
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils import executor
from dotenv import load_dotenv
import openai

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
# GNews API key: prefer env var, fall back to user-provided key
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
if not GNEWS_API_KEY:
    logging.warning("GNEWS_API_KEY is not set. The bot will fall back to RSS feeds. Set GNEWS_API_KEY in .env or your host environment for GNews support.")

bot = Bot(BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

DB = "penpal.sqlite"

# GNews categories - present these to users
TOPIC_CHOICES = ["World", "Nation", "Business", "Tech", "Entertainment", "Sports", "Science", "Health"]

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
GNEWS_ALLOWED_TOPICS = {"world", "nation", "business", "technology", "entertainment", "sports", "science", "health"}




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
        for tag in soup.find_all(['p']):
            txt = tag.get_text(separator=' ', strip=True)
            if txt and len(txt) > 20:
                paragraphs.append(txt)
            if len(' '.join(paragraphs).split('.')) >= min_sentences:
                break
        text = ' '.join(paragraphs).strip()
        # find a likely image
        img = None
        # Prefer og:image
        og = soup.find('meta', property='og:image')
        if og and og.get('content'):
            img = og['content']
        else:
            first_img = soup.find('img')
            if first_img and first_img.get('src'):
                img = first_img['src']
        return (text, img)
    except Exception:
        logging.exception(f"Failed to fetch article page: {url}")
        return (None, None)


def get_gnews_articles(topic=None, limit=10):
    """Query GNews API and return a list of articles with keys title, description, url, image."""
    try:
        params = {'token': GNEWS_API_KEY, 'lang': 'en', 'max': limit}
        if topic:
            # prefer mapping
            q = GNEWS_TOPIC_MAP.get(topic, topic).lower()
        else:
            q = None

        # If q is a recognized GNews topic, call top-headlines with topic param
        if q in GNEWS_ALLOWED_TOPICS:
            params['topic'] = q
            logging.debug(f"GNews: using top-headlines topic={q}")
            resp = requests.get('https://gnews.io/api/v4/top-headlines', params=params, timeout=8)
        else:
            # fallback to search by keyword
            logging.debug(f"GNews: using search q={q}")
            search_params = {'token': GNEWS_API_KEY, 'lang': 'en', 'max': limit}
            if q:
                search_params['q'] = q
            resp = requests.get('https://gnews.io/api/v4/search', params=search_params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for a in data.get('articles', []):
            articles.append({
                'title': a.get('title'),
                'description': a.get('description'),
                'url': a.get('url'),
                'image': a.get('image'),
            })
        return articles
    except Exception:
        logging.exception('GNews API request failed')
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
    return sqlite3.connect(DB)

def init_db():
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY,
            tg_username TEXT,
            level TEXT,
            topics TEXT,
            mode TEXT,
            created_at TEXT,
            last_news_url TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, role TEXT, content TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS vocab(
            user_id INTEGER, phrase TEXT, example TEXT, added_at TEXT, bin INTEGER DEFAULT 1)""")
        c.execute("""CREATE TABLE IF NOT EXISTS news_cache(
            id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, title TEXT, summary TEXT, published_at TEXT)""")
        # Add questions column if it doesn't exist (safe to run repeatedly)
        try:
            c.execute("ALTER TABLE news_cache ADD COLUMN questions TEXT")
        except Exception:
            # column probably exists already
            pass
        # Add per-user mode preference if it doesn't exist
        try:
            c.execute("ALTER TABLE users ADD COLUMN mode TEXT")
        except Exception:
            pass
        # Add per-user last_news_url to users table if missing
        try:
            c.execute("ALTER TABLE users ADD COLUMN last_news_url TEXT")
        except Exception:
            # column probably exists already
            pass
        conn.commit()

def save_user(user_id, username):
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users(id, tg_username, created_at) VALUES(?,?,?)",
                  (user_id, username, datetime.utcnow().isoformat()))
        conn.commit()

def set_user_level(user_id, level):
    with closing(db()) as conn:
        conn.execute("UPDATE users SET level=? WHERE id=?", (level, user_id)); conn.commit()

def set_user_topics(user_id, topics):
    with closing(db()) as conn:
        conn.execute("UPDATE users SET topics=? WHERE id=?", (",".join(topics), user_id)); conn.commit()

def get_user(user_id):
    with closing(db()) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        row = c.execute("SELECT id, tg_username, level, topics, mode, last_news_url FROM users WHERE id= ?", (user_id,)).fetchone()
    return dict(row) if row else None

def set_user_mode(user_id, mode):
    with closing(db()) as conn:
        conn.execute("UPDATE users SET mode=? WHERE id=?", (mode, user_id)); conn.commit()

def set_user_last_news(user_id, url):
    with closing(db()) as conn:
        conn.execute("UPDATE users SET last_news_url=? WHERE id=?", (url, user_id)); conn.commit()

def save_msg(user_id, role, content):
    with closing(db()) as conn:
        conn.execute("INSERT INTO messages(user_id, role, content, created_at) VALUES(?,?,?,?)",
                     (user_id, role, content, datetime.utcnow().isoformat()))
        # keep last 30
        conn.execute("""DELETE FROM messages WHERE id NOT IN (
            SELECT id FROM messages WHERE user_id=? ORDER BY id DESC LIMIT 30) AND user_id=?""", (user_id, user_id))
        conn.commit()

def add_vocab(user_id, items):
    with closing(db()) as conn:
        for it in items:
            conn.execute("INSERT INTO vocab(user_id, phrase, example, added_at) VALUES(?,?,?,?)",
                         (user_id, it.get("phrase",""), it.get("example",""), datetime.utcnow().isoformat()))
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
            messages=[{"role":"system","content":"You return strict JSON only."},
                      {"role":"user","content":prompt}],
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
                summary = summary.rsplit(' ', 1)[0] + '...'
            data = {"summary": summary or article_title, "questions": [
                "What surprised you most from this article?",
                "How could this news affect people like you?",
                "Do you agree with the main idea? Why or why not?"
            ], "vocab": []}
    except Exception:
        logging.exception("OpenAI request failed; using fallback news data")
        # If OpenAI fails, produce a simple summary from the article text and three friendly questions.
        summary = (article_text or article_title)[:300].strip()
        if len(summary) > 280:
            summary = summary.rsplit(' ', 1)[0] + '...'
        data = {"summary": summary or article_title, "questions": [
            "What surprised you most from this article?",
            "How could this news affect people like you?",
            "Do you agree with the main idea? Why or why not?"
        ], "vocab": []}
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
            rows.append(row); row=[]
    if row: rows.append(row)
    rows.append([InlineKeyboardButton("Готово ✔️", callback_data="topic:done")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def level_keyboard():
    levels = ["A2","B1","B2","C1"]
    btns = [[InlineKeyboardButton(l, callback_data=f"level:{l}") for l in levels]]
    return InlineKeyboardMarkup(inline_keyboard=btns)

def mode_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Обсудить статью 📰", callback_data="mode:news")],
        [InlineKeyboardButton("Свободный разговор 💬", callback_data="mode:chat")]
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
                chosen_topic = t
                break

        if articles:
            # avoid repeating the last article shown to this user
            last_url = user.get("last_news_url")
            candidates = [a for a in articles if a.get('url') != last_url]
            if not candidates:
                # all articles match last_url; fall back to full list
                candidates = articles
            item = random.choice(candidates)
            title = item.get('title') or 'News'
            url = item.get('url') or ''
            desc = item.get('description') or title
            image_candidate = item.get('image')
        else:
            # No articles from GNews: inform the user (we no longer use RSS fallback)
            logging.warning(f"No GNews articles found for topics {selected_topics}")
            await bot.send_message(user_id, "Извини — сейчас не удалось найти подходящие статьи по твоим темам. Попробуй /topics или /news позже. 🤖")
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
            data = {"summary": title, "questions": ["What surprised you most?", "How could this affect daily life?", "Do you agree?"], "vocab": []}

        # store vocab
        try:
            add_vocab(user_id, data.get("vocab", []))
        except Exception:
            logging.exception("Failed to add vocab; continuing")

        # Do not show questions immediately; they will be shown when the user presses "Completed".
        voc = data.get("vocab", [])
        voc_txt = "\n".join([f"🔹 <b>{v['phrase']}</b> — <i>{v['example']}</i>" for v in voc]) if voc else ""
        text = f"<b>{title}</b>\n\n{data.get('summary', '')}"
        if voc_txt:
            text += f"\n\n<b>Useful phrases:</b>\n{voc_txt}"

        # Save the news and questions to cache so we can support translate/completed flows
        with closing(db()) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO news_cache(url, title, summary, published_at, questions) VALUES(?,?,?,?,?)",
                      (url, title, data.get('summary',''), datetime.utcnow().isoformat(), json.dumps(data.get('questions',[]))))
            cache_id = c.lastrowid
            conn.commit()

        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton("Перевести 🔁", callback_data=f"news:translate:{cache_id}"),
             InlineKeyboardButton("Завершил(а) ✅", callback_data=f"news:done:{cache_id}")],
            [InlineKeyboardButton("Поменять статью 🔁", callback_data="news:more")]
        ])

        logging.info(f"Sending news (image={bool(image_url)}) to user {user_id}")
        # If we have an image and a longer article, send as photo with caption (Telegram caption limit ~1024 chars)
        if image_url and article_text:
            caption = (text[:900] + '...') if len(text) > 900 else text
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
            await bot.send_message(user_id, "Извини — не удалось получить новости. Попробуй /news позже. 🙏")
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
    await m.answer("Привет! Я <b>PenPal English</b> 👋 — твой дружелюбный собеседник по английскому.\n\nКакой у тебя уровень?", reply_markup=level_keyboard())

@dp.callback_query_handler(lambda c: c.data.startswith("level:"))
async def choose_level(c: types.CallbackQuery):
    level = c.data.split(":")[1]
    user_id = c.from_user.id
    set_user_level(user_id, level)
    # reset mode so the user can pick again
    set_user_mode(user_id, None)
    await c.answer()
    await c.message.edit_text(
        f"Отлично! Уровень установлен: <b>{level}</b> 🎯\n\nЧто ты хочешь сделать дальше?",
        reply_markup=mode_keyboard()
    )

@dp.callback_query_handler(lambda c: c.data.startswith("mode:"))
async def choose_mode(c: types.CallbackQuery):
    user_id = c.from_user.id
    mode = c.data.split(":")[1]
    if mode not in {"news", "chat"}:
        await c.answer("Неизвестный режим.", show_alert=True); return
    set_user_mode(user_id, mode)
    await c.answer()
    if mode == "news":
        user = get_user(user_id)
        existing = []
        if user and user.get("topics"):
            existing = [t for t in user["topics"].split(",") if t]
        await c.message.edit_text("Выбери темы, которые тебе нравятся:", reply_markup=topic_keyboard(existing))
    else:
        await c.message.edit_text(
            "Отлично! Пиши мне, и я поддержу свободный разговор на английском 😊\n\n"
            "Если захочешь обсуждать статьи, выбери уровень ещё раз через /level или используй /news."
        )

@dp.callback_query_handler(lambda c: c.data.startswith("topic:"))
async def choose_topics(c: types.CallbackQuery):
    user = get_user(c.from_user.id)
    if not user:
        await c.answer("Не удалось найти профиль. Попробуй /start.", show_alert=True); return
    if user.get("mode") != "news":
        await c.answer("Сначала выбери режим «Обсудить статью» после выбора уровня.", show_alert=True); return
    selected = [t.strip() for t in (user.get("topics") or "").split(",") if t.strip()]
    val = c.data.split(":")[1]
    if val == "done":
        if not selected:
            await c.answer("Выбери хотя бы одну тему 🙂", show_alert=True); return
        await c.message.edit_text("Отлично! Я принесу материал для обсуждения. Вот новость 📰:")
        await send_news(c.from_user.id)
        return
    if val in selected:
        selected = [t for t in selected if t != val]
    else:
        selected.append(val)
    set_user_topics(c.from_user.id, selected)
    await c.message.edit_reply_markup(reply_markup=topic_keyboard(selected))

@dp.callback_query_handler(lambda c: c.data.startswith("news:more"))
async def more_news(c: types.CallbackQuery):
    await c.answer("Загружаю другую статью… ⏳")
    await send_news(c.from_user.id)

@dp.callback_query_handler(lambda c: c.data.startswith("ans:"))
async def answer_hint(c: types.CallbackQuery):
    idx = int(c.data.split(":")[1])
    prompts = ["Напиши свой ответ на вопрос 1 👇",
               "Что ты думаешь о вопросе 2? 👇",
               "Твои мысли по вопросу 3? 👇"]
    await bot.send_message(c.from_user.id, prompts[idx])


@dp.callback_query_handler(lambda c: c.data.startswith("news:translate:"))
async def news_translate(c: types.CallbackQuery):
    parts = c.data.split(":")
    cache_id = int(parts[2])
    with closing(db()) as conn:
        row = conn.cursor().execute("SELECT title, summary FROM news_cache WHERE id=?", (cache_id,)).fetchone()
    if not row:
        await c.answer("Не удалось найти статью.", show_alert=True); return
    title, summary = row
    # Try OpenAI translate (short), else simple fallback
    try:
        prompt = f"Translate this short article to Russian, keep sentences aligned:\n\nTitle: {title}\n\n{summary}"
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0)
        translated = resp.choices[0].message["content"]
    except Exception:
        logging.exception("Translation failed via OpenAI; using naive fallback")
        # naive fallback: return the original for now
        translated = "(Translation unavailable)"
    await bot.send_message(c.from_user.id, f"<b>{title}</b>\n\n{translated}", parse_mode='HTML')


@dp.callback_query_handler(lambda c: c.data.startswith("news:done:"))
async def news_done(c: types.CallbackQuery):
    parts = c.data.split(":")
    cache_id = int(parts[2])
    with closing(db()) as conn:
        row = conn.cursor().execute("SELECT questions FROM news_cache WHERE id=?", (cache_id,)).fetchone()
    if not row:
        await c.answer("Не удалось найти статью.", show_alert=True); return
    questions = json.loads(row[0] or '[]')
    if not questions:
        await bot.send_message(c.from_user.id, "Что ты думаешь по этой теме?")
        return

    # Send only the first question with instructions and an 'Another question' button
    q0 = questions[0]
    instr = ("Отлично — ты прочитал(а) статью! Я задам по одному вопросу за раз.\n"
             "Ответь, когда будешь готов(а), или нажми «Другой вопрос», чтобы увидеть другой вопрос.\n\n")
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Другой вопрос 🔁", callback_data=f"news:next:{cache_id}:1")],
        [InlineKeyboardButton("Поменять статью 🔁", callback_data="news:more")]
    ])
    await bot.send_message(c.from_user.id, instr + q0, reply_markup=kb)


@dp.callback_query_handler(lambda c: c.data.startswith("news:next:"))
async def news_next(c: types.CallbackQuery):
    # callback format: news:next:<cache_id>:<index>
    parts = c.data.split(":")
    cache_id = int(parts[2])
    idx = int(parts[3])
    with closing(db()) as conn:
        row = conn.cursor().execute("SELECT questions FROM news_cache WHERE id=?", (cache_id,)).fetchone()
    if not row:
        await c.answer("Не удалось найти статью.", show_alert=True); return
    questions = json.loads(row[0] or '[]')
    if idx < 0 or idx >= len(questions):
        await c.answer("No more questions.", show_alert=True); return
    q = questions[idx]
    # Prepare next index (wrap or stop at end)
    next_idx = idx + 1
    kb_buttons = []
    if next_idx < len(questions):
        kb_buttons.append(InlineKeyboardButton("Другой вопрос 🔁", callback_data=f"news:next:{cache_id}:{next_idx}"))
    kb_buttons.append(InlineKeyboardButton("Поменять статью 🔁", callback_data="news:more"))
    kb = InlineKeyboardMarkup(inline_keyboard=[kb_buttons])
    await bot.send_message(c.from_user.id, q, reply_markup=kb)

@dp.message_handler(commands=["topics"])
async def cmd_topics(m: types.Message):
    user = get_user(m.from_user.id)
    current = (user.get("topics") or "").split(",") if user and user.get("topics") else []
    await m.answer("Update your interests 🌟:", reply_markup=topic_keyboard(current))

@dp.message_handler(commands=["level"])
async def cmd_level(m: types.Message):
    await m.answer("Pick your level 🎯:", reply_markup=level_keyboard())

@dp.message_handler(commands=["news"])
async def cmd_news(m: types.Message):
    await send_news(m.from_user.id)

@dp.message_handler(commands=["review"])
async def cmd_review(m: types.Message):
    with closing(db()) as conn:
        c = conn.cursor()
        items = c.execute("SELECT phrase, example, bin FROM vocab WHERE user_id=? ORDER BY bin ASC, added_at DESC LIMIT 6",
                          (m.from_user.id,)).fetchall()
    if not items:
        await m.answer("No vocab yet — chat a bit or try /news and I’ll save useful phrases for you. ✨")
        return
    msg = "<b>Quick review:</b>\n"
    for i,(p,e,b) in enumerate(items,1):
        msg += f"{i}) <b>{p}</b> — <i>{e}</i>\n"
    await m.answer(msg)

@dp.message_handler(commands=["help"])
async def cmd_help(m: types.Message):
    await m.answer("Try /news for a fresh topic 📰, /topics to change interests, /level to adjust difficulty, /review for phrases. Or just chat with me in English! 😊")

@dp.message_handler()
async def chat(m: types.Message):
    user = get_user(m.from_user.id)
    if not user or not user.get("level"):
        save_user(m.from_user.id, m.from_user.username or "")
        await m.answer("Let’s set your level first:", reply_markup=level_keyboard()); return
    # Build short context (last ~6 turns)
    with closing(db()) as conn:
        c = conn.cursor()
        rows = c.execute("SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT 6", (m.from_user.id,)).fetchall()
    history = [{"role": r, "content": ct} for (r, ct) in rows[::-1]]
    messages = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":m.text}]
    save_msg(m.from_user.id, "user", m.text)
    reply = await gpt_chat(messages)
    # No longer mining 'Useful:' phrases. Corrections are handled by the assistant per SYSTEM_PROMPT.
    save_msg(m.from_user.id, "assistant", reply)
    await m.answer(reply)

if __name__ == "__main__":
    init_db()
    executor.start_polling(dp, skip_updates=True)
