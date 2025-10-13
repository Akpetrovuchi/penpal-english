import os
from contextlib import closing
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load env variables from .env (supports both root and env/.env)
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), 'env/.env'), override=True)

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "penpal_english")
    user = os.getenv("POSTGRES_USER", "your_db_user")
    password = os.getenv("POSTGRES_PASSWORD", "your_db_password")
    DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"

def db():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.DictCursor)

def get_db_url():
    return DB_URL
