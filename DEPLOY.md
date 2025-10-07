Quick deploy checklist

1) Create a GitHub repo and push code

git init
git add .
git commit -m "initial"
gh repo create my-penpal-bot --public --source=. --remote=origin
git push -u origin main

2) Create a Render background worker

- New -> Background worker
- Connect GitHub repo
- Set Build Command: pip install -r requirements.txt
- Set Start Command: /opt/render/project/src/.venv/bin/python penpal_english_bot.py
- Add environment variables: BOT_TOKEN, OPENAI_API_KEY

3) Verify logs on Render and test the bot in Telegram.
