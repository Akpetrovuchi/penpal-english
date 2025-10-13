#!/usr/bin/env python3
"""Check presence of expected environment variables and print safe validation commands.

This script loads `.env` (project root) and `env/.env` (local folder) and reports which
keys are set. It masks values when printing. It does NOT call external APIs automatically;
it prints curl commands you can run manually to validate tokens.
"""
from dotenv import load_dotenv
import os
from pathlib import Path


def mask_value(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= 8:
        return s[0] + "*" * (len(s) - 2) + s[-1]
    return s[:4] + "..." + s[-4:]


def main():
    # Try project root .env first, then env/.env
    root = Path(".env")
    local = Path("env/.env")
    if root.exists():
        load_dotenv(dotenv_path=root)
    if local.exists():
        load_dotenv(dotenv_path=local, override=False)

    keys = ["BOT_TOKEN", "OPENAI_API_KEY", "GNEWS_API_KEY", "ADMIN_ID"]
    print("Environment checks (values masked):\n")
    for k in keys:
        v = os.getenv(k)
        status = "SET" if v else "MISSING"
        if v:
            print(f"{k}: {status} ({mask_value(v)})")
        else:
            print(f"{k}: {status}")

    print("\nTo validate keys against the services, export your env file and run the curl commands below (do this locally):\n")
    print("# Load env/.env into the current shell")
    print("export $(grep -v '^#' env/.env | xargs)")
    print("\n# Telegram: get bot info")
    print("curl -s \"https://api.telegram.org/bot$BOT_TOKEN/getMe\" | jq")
    print("\n# OpenAI: list models (works if key is valid)")
    print("curl -s -H \"Authorization: Bearer $OPENAI_API_KEY\" https://api.openai.com/v1/models | jq")
    print("\n# GNews: fetch one top headline (if key valid)")
    print("curl -s \"https://gnews.io/api/v4/top-headlines?token=$GNEWS_API_KEY&lang=en&max=1\" | jq")


if __name__ == "__main__":
    main()
