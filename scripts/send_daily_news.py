import os
import logging
from datetime import datetime
from contextlib import closing
from db import db
from penpal_english_bot import get_all_users_for_daily, get_user, send_news
import asyncio

logging.basicConfig(level=logging.INFO)

async def main():
    users = get_all_users_for_daily()
    for u in users:
        user_id = u["id"]
        user = get_user(user_id)
        if not user:
            continue
        # Only send if user's timezone is UTC or not set
        tz = user.get("timezone")
        if tz and tz != "UTC":
            continue
        try:
            # Send Russian greeting before news
            from penpal_english_bot import bot, mode_keyboard
            await bot.send_message(user_id, "👋 Привет! У меня для тебя свежая новость — обсудим?", reply_markup=mode_keyboard())
            await send_news(user_id)
            logging.info(f"Sent daily news to user {user_id}")
        except Exception as e:
            logging.error(f"Failed to send news to user {user_id}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
