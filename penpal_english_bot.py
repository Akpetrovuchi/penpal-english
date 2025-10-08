# penpal_english_bot.py
import os, json, asyncio, logging, sqlite3, time, random
from datetime import datetime, timedelta
from contextlib import closing
import feedparser
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

bot = Bot(BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

DB = "penpal.sqlite"

# ---- News feeds (topic-aware) ----
TOPIC_FEEDS = {
    "World": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.theguardian.com/world/rss",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
    "Tech": [
        "https://www.theverge.com/rss/index.xml",
        "https://www.wired.com/feed/rss",
        "https://www.engadget.com/rss.xml",
    ],
    "Business": [
        "https://www.ft.com/rss/world",
        "https://www.economist.com/business/rss.xml",
        "https://www.reuters.com/markets/rss",
    ],
    "Science": [
        "https://www.sciencedaily.com/rss/top/science.xml",
        "https://www.npr.org/rss/rss.php?id=1007",
        "https://www.nature.com/subjects/science.rss",
    ],
    "AI": [
        "https://www.technologyreview.com/feed/",
        "https://feeds.feedburner.com/venturebeat/SZYF",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    ],
    "Health": [
        "https://www.who.int/feeds/entity/mediacentre/news/en/rss.xml",
        "https://www.medicalnewstoday.com/rss",
        "https://www.npr.org/rss/rss.php?id=1128",
    ],
    "Movies": [
        "https://www.theguardian.com/film/rss",
        "https://www.hollywoodreporter.com/feeds/rss/",
        "https://www.empireonline.com/feeds/all.xml",
    ],
    "Travel": [
        "https://www.theguardian.com/travel/rss",
        "https://www.cntraveler.com/feed/rss",
        "https://www.lonelyplanet.com/news/rss",
    ],
    "Fitness": [
        "https://www.menshealth.com/fitness/rss",
        "https://www.womenshealthmag.com/fitness/rss",
        "https://www.self.com/feed/all",
    ],
    "Food": [
        "https://www.seriouseats.com/rss",
        "https://www.bonappetit.com/feed/rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/Food.xml",
    ],
    "Finance": [
        "https://www.reuters.com/finance/rss",
        "https://www.marketwatch.com/feeds/topstories",
        "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline",
    ],
    "Culture": [
        "https://www.theguardian.com/culture/rss",
        "https://www.newyorker.com/feed/culture",
        "https://www.vogue.com/feed/rss",
    ],
    "Design": [
        "https://www.dezeen.com/feed/",
        "https://www.creativebloq.com/feeds/all",
        "https://www.fastcompany.com/rss",
    ],
}

# Add missing topic choices and default feeds
TOPIC_CHOICES = list(TOPIC_FEEDS.keys())
DEFAULT_FEEDS = [url for urls in TOPIC_FEEDS.values() for url in urls]

BUDDY_TOPICS = [
    "ordering food at a cafe",
    "booking a surfing lesson",
    "discussing an interesting article",
    "planning a weekend city break",
    "making small talk at a networking event",
    "arranging a doctor appointment",
    "organizing a surprise birthday party",
    "preparing for a job interview",
    "chatting about a favorite movie night",
    "asking a friend to help with homework",
]


# Fetch helper with a real browser user-agent (prevents some RSS blocks)
import urllib.request
def fetch_feed(url: str):
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read()
        return feedparser.parse(data)
    except Exception:
        # last-chance fallback; feedparser will try itself
        return feedparser.parse(url)


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



SYSTEM_PROMPT = """You are “PenPal English,” a friendly pen-pal and English tutor.
Goals: keep a natural chat tone, adapt to the user’s level, and build confidence.
Rules:
1) Be concise (≤120 words unless asked).
2) Ask one engaging follow-up.
3) Correct only high-impact mistakes inline (✅ corrected → brief 1-line reason).
4) Highlight up to 2 useful phrases per turn with simple examples.
5) Respect user’s topics and tone.
6) When asked to explain, use A2–B2-friendly English, bullet points, and one mini exercise.
"""

def db():
    return sqlite3.connect(DB)

def init_db():
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY, tg_username TEXT, level TEXT, topics TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, role TEXT, content TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS vocab(
            user_id INTEGER, phrase TEXT, example TEXT, added_at TEXT, bin INTEGER DEFAULT 1)""")
        c.execute("""CREATE TABLE IF NOT EXISTS news_cache(
            id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, title TEXT, summary TEXT, published_at TEXT)""")
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
        c = conn.cursor()
        return c.execute("SELECT id, tg_username, level, topics FROM users WHERE id=?", (user_id,)).fetchone()

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
        return "Sorry — my language engine is unavailable right now. Try again later or use /news to get a short article." 

async def gpt_structured_news(level, topics, article_title, article_text, url):
    prompt = f"""Create a 2–3 sentence summary (CEFR {level}) for this article, then 3 casual discussion questions.
User interests: {topics}
Title: {article_title}
Article: {article_text[:2000]}
Return JSON with keys summary, questions (list of 3), vocab (list of 1-2 items with 'phrase' and 'example'). Keep it short."""
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
        row.append(InlineKeyboardButton(f"{mark}{t}", callback_data=f"topic:{t}"))
        if len(row) == 3:
            rows.append(row); row=[]
    if row: rows.append(row)
    rows.append([InlineKeyboardButton("Done ✔️", callback_data="topic:done")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def level_keyboard():
    levels = ["A2","B1","B2","C1"]
    btns = [[InlineKeyboardButton(l, callback_data=f"level:{l}") for l in levels]]
    return InlineKeyboardMarkup(inline_keyboard=btns)


def buddy_options_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton("🤖 Chat with AI Buddy", callback_data="buddy:start")],
            [InlineKeyboardButton("🎯 Choose topics", callback_data="buddy:topics")],
        ]
    )

async def send_news(user_id):
    try:
        user = get_user(user_id)
        level = user[2] or "B1"
        topics = (user[3] or "World").split(",")
        logging.info(f"Fetching news for user={user_id} level={level} topics={topics}")
        # Use fetch_feed for reliability
        feed_url = random.choice(DEFAULT_FEEDS)
        logging.debug(f"Selected feed: {feed_url}")
        feed = fetch_feed(feed_url)
        entries_count = len(getattr(feed, 'entries', []) or [])
        logging.debug(f"Feed entries count: {entries_count}")
        if not feed.entries:
            await bot.send_message(user_id, "Couldn’t fetch news right now. Try /news again.")
            return
        item = random.choice(feed.entries[:10])
        title = item.get("title", "World news")
        url = item.get("link", "")
        desc = item.get("summary", "") or item.get("description", "") or title
        # Try to fetch longer article text and an image when available
        article_text, image_url = (None, None)
        if url:
            article_text, image_url = fetch_article(url, min_sentences=10)
        # prefer article_text if it's long enough
        article_body = article_text or desc
        logging.debug(f"Selected item title={title} url={url}")
        try:
            data = await gpt_structured_news(level, topics, title, article_body, url)
        except Exception:
            logging.exception("gpt_structured_news failed, using fallback data")
            data = {"summary": title, "questions": ["What surprised you most?", "How could this affect daily life?", "Do you agree?"], "vocab": []}
        # store vocab
        try:
            add_vocab(user_id, data.get("vocab", []))
        except Exception:
            logging.exception("Failed to add vocab; continuing")
        q = "\n".join([f"• {x}" for x in data.get("questions", [])])
        voc = data.get("vocab", [])
        voc_txt = "\n".join([f"🔹 <b>{v['phrase']}</b> — <i>{v['example']}</i>" for v in voc]) if voc else ""
        text = f"<b>{title}</b>\n\n{data.get('summary', '')}\n\n<b>Let’s talk:</b>\n{q}"
        if voc_txt:
            text += f"\n\n<b>Useful phrases:</b>\n{voc_txt}"
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton("Answer Q1", callback_data="ans:0"),
             InlineKeyboardButton("Answer Q2", callback_data="ans:1"),
             InlineKeyboardButton("Answer Q3", callback_data="ans:2")],
            [InlineKeyboardButton("More news 🔄", callback_data="news:more")]
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
            await bot.send_message(user_id, "Sorry — I couldn’t fetch or prepare news just now. Try /news again later.")
        except Exception:
            logging.exception("Failed to send error message to user")

@dp.message_handler(commands=["start"])
async def start(m: types.Message):
    save_user(m.from_user.id, m.from_user.username or "")
    await m.answer("Hi! I’m <b>PenPal English</b>—your friendly AI pen-pal.\n\nWhat’s your level?", reply_markup=level_keyboard())

@dp.callback_query_handler(lambda c: c.data.startswith("level:"))
async def choose_level(c: types.CallbackQuery):
    level = c.data.split(":")[1]
    set_user_level(c.from_user.id, level)
    await c.answer()
    await c.message.edit_text(
        f"Great! Level set to <b>{level}</b>.\n\nWant to jump into a quick chat or pick your interests first?",
        reply_markup=buddy_options_keyboard(),
    )


@dp.callback_query_handler(lambda c: c.data == "buddy:topics")
async def buddy_choose_topics(c: types.CallbackQuery):
    await c.answer()
    user = get_user(c.from_user.id)
    current = (user[3] or "").split(",") if user and user[3] else []
    await c.message.edit_text("Pick topics you like:", reply_markup=topic_keyboard(current))


@dp.callback_query_handler(lambda c: c.data == "buddy:start")
async def buddy_start_chat(c: types.CallbackQuery):
    await c.answer("Let’s chat!", show_alert=False)
    topic = random.choice(BUDDY_TOPICS)
    await c.message.edit_text(
        "Awesome! Here’s a random chat idea. Reply when you’re ready.",
        reply_markup=None,
    )
    scenario_note = f"[AI Buddy scenario] {topic}"
    save_msg(c.from_user.id, "user", scenario_note)
    intro_prompt = (
        "Start a friendly role-play with the learner. Scenario: "
        f"{topic}. In 2 short sentences set the scene and invite them to respond."
    )
    reply = await gpt_chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": intro_prompt},
    ])
    text = f"🎲 <b>Random topic:</b> {topic}\n\n{reply}"
    save_msg(c.from_user.id, "assistant", text)
    await bot.send_message(c.from_user.id, text)

@dp.callback_query_handler(lambda c: c.data.startswith("topic:"))
async def choose_topics(c: types.CallbackQuery):
    user = get_user(c.from_user.id)
    selected = set((user[3] or "").split(",")) if user[3] else set()
    val = c.data.split(":")[1]
    if val == "done":
        if not selected:
            await c.answer("Choose at least one 🙂", show_alert=True); return
        await c.message.edit_text("Awesome! I’ll bring you something to talk about.\nHere’s a news bite:")
        await send_news(c.from_user.id)
        return
    if val in selected:
        selected.remove(val)
    else:
        selected.add(val)
    set_user_topics(c.from_user.id, list(selected))
    await c.message.edit_reply_markup(reply_markup=topic_keyboard(list(selected)))

@dp.callback_query_handler(lambda c: c.data.startswith("news:more"))
async def more_news(c: types.CallbackQuery):
    await c.answer("Fetching…")
    await send_news(c.from_user.id)

@dp.callback_query_handler(lambda c: c.data.startswith("ans:"))
async def answer_hint(c: types.CallbackQuery):
    idx = int(c.data.split(":")[1])
    prompts = ["Type your answer to Question 1 👇",
               "What do you think about Question 2? 👇",
               "Your thoughts on Question 3? 👇"]
    await bot.send_message(c.from_user.id, prompts[idx])

@dp.message_handler(commands=["topics"])
async def cmd_topics(m: types.Message):
    user = get_user(m.from_user.id)
    current = (user[3] or "").split(",") if user and user[3] else []
    await m.answer("Update your interests:", reply_markup=topic_keyboard(current))

@dp.message_handler(commands=["level"])
async def cmd_level(m: types.Message):
    await m.answer("Pick your level:", reply_markup=level_keyboard())

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
        await m.answer("No vocab yet—chat a bit or try /news and I’ll save phrases for you.")
        return
    msg = "<b>Quick review:</b>\n"
    for i,(p,e,b) in enumerate(items,1):
        msg += f"{i}) <b>{p}</b> — <i>{e}</i>\n"
    await m.answer(msg)

@dp.message_handler(commands=["help"])
async def cmd_help(m: types.Message):
    await m.answer("Try /news for a fresh topic, /topics to change interests, /level to adjust difficulty, /review for phrases. Or just chat with me in English!")

@dp.message_handler()
async def chat(m: types.Message):
    user = get_user(m.from_user.id)
    if not user or not user[2]:
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
    # naive vocab mining: look for “Useful:” block
    useful = []
    if "Useful:" in reply:
        after = reply.split("Useful:",1)[1]
        lines = after.strip().splitlines()[:2]
        for ln in lines:
            parts = ln.replace("•","").replace("🔹","").strip().split("—")
            phrase = parts[0].strip()
            example = (parts[1].strip() if len(parts)>1 else "")
            if phrase:
                useful.append({"phrase":phrase, "example":example})
        if useful: add_vocab(m.from_user.id, useful)
    save_msg(m.from_user.id, "assistant", reply)
    await m.answer(reply)

if __name__ == "__main__":
    init_db()
    executor.start_polling(dp, skip_updates=True)

