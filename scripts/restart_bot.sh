#!/usr/bin/env bash
# Restart the penpal bot: load .env, activate venv, kill existing process, start in background
set -e
cd "$(dirname "$0")/.."
# load env if present
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi
# activate venv if present
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
# ensure run dir
mkdir -p run
# kill existing process (matching penpal_english_bot.py)
pkill -f penpal_english_bot.py || true
# start bot in background and write pid
nohup python penpal_english_bot.py > bot.log 2>&1 &
PID=$!
echo $PID > run/bot.pid
# brief status
echo "Bot restarted with PID $PID"
exit 0
