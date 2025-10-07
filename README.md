PenPal English Bot

This repository contains a Telegram bot that provides short news bites and English practice.

Quick start (local)

1. Create and activate venv

python3 -m venv .venv
source .venv/bin/activate

2. Install deps

pip install -r requirements.txt

3. Create a .env with your Telegram bot token and optional OpenAI key

BOT_TOKEN=123:ABC
OPENAI_API_KEY=sk-...

4. Run

python penpal_english_bot.py

Deployment options

- GitHub: push your repository to GitHub. Example:

	git init
	git add .
	git commit -m "initial"
	gh repo create YOUR_REPO_NAME --public --source=. --remote=origin
	git push -u origin main

- Render (recommended for background workers):
	1. Create a new service -> Background Worker.
	2. Connect your GitHub repo.
	3. Set the Build Command: pip install -r requirements.txt
	4. Set the Start Command: /opt/render/project/src/.venv/bin/python penpal_english_bot.py
	5. Add environment variables (BOT_TOKEN, OPENAI_API_KEY) in the Render dashboard.

- Railway / Heroku:
	- Use the start command: python penpal_english_bot.py
	- Add environment variables in the project settings.

Notes

- Keep your `.env` (local) and service environment variables secret.
- For OpenAI features use a valid `OPENAI_API_KEY`.
- Some hosting providers require a Procfile (provided) or a start command.
