import os
import logging
from datetime import datetime
from contextlib import closing
from db import db
from penpal_english_bot import get_all_users_for_daily, get_user, send_news
import asyncio

logging.basicConfig(level=logging.INFO)

async def main():
    # Determine what to send based on day of year (odd/even)
    day_of_year = datetime.now().timetuple().tm_yday
    is_news_day = (day_of_year % 2 == 0)  # Even days = news, odd days = "how was your day"
    
    logging.info(f"Day {day_of_year}: {'NEWS' if is_news_day else 'HOW WAS YOUR DAY'} mode")
    
    users = get_all_users_for_daily()
    from penpal_english_bot import bot, mode_keyboard
    
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
            if is_news_day:
                # News day: send greeting + news
                await bot.send_message(user_id, "👋 Привет! У меня для тебя свежая новость — обсудим?", reply_markup=mode_keyboard())
                await send_news(user_id)
                logging.info(f"Sent daily news to user {user_id}")
            else:
                # "How was your day" day
                await bot.send_message(
                    user_id, 
                    "👋 Привет! Расскажи, как прошёл твой день?",
                    reply_markup=mode_keyboard()
                )
                logging.info(f"Sent 'how was your day' to user {user_id}")
        except Exception as e:
            logging.error(f"Failed to send daily message to user {user_id}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
