Quick deploy checklist
======================

1. Prep & push to GitHub
------------------------

```bash
git init
git add .
git commit -m "Initial PenPal English bot"
gh repo create YOUR_REPO --public --source=. --remote=origin
git push -u origin main
```

2. Configure GitHub Actions secrets
-----------------------------------

- Open *Settings → Secrets and variables → Actions*.
- Add `BOT_TOKEN`, `OPENAI_API_KEY`, and optionally `GNEWS_API_KEY`.
- The `CI` workflow (`.github/workflows/python-app.yml`) lints the project on every push/pull request.

3. Render deployment (recommended background worker)
----------------------------------------------------

1. Create a **Background Worker** service.
2. Connect the GitHub repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `/opt/render/project/src/.venv/bin/python penpal_english_bot.py`
5. Environment variables: `BOT_TOKEN`, `OPENAI_API_KEY`, `GNEWS_API_KEY` (optional).
6. Deploy and check the Render logs to confirm the bot comes online.

4. Railway / Heroku
-------------------

- Start command: `python penpal_english_bot.py`
- Add the same environment variables through the provider dashboard.
- Ensure a worker dyno/process is used (no HTTP server required).

5. Post-deploy smoke test
-------------------------

- Ping the bot in Telegram with `/start`.
- Trigger `/news` to verify OpenAI + GNews integration.
- Confirm vocabulary review (`/review`) works and that logs do not show database errors.
