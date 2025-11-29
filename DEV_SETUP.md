# Dev Environment Setup

## Quick Start

1. **Copy environment template**
   ```bash
   cp .env.example .env
   # Fill in your dev bot token and other credentials
   ```

2. **Install PostgreSQL** (if not installed)
   ```bash
   brew install postgresql@14
   brew services start postgresql@14
   ```

3. **Create dev database**
   ```bash
   createdb penpal_english_dev
   ```

4. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

5. **Initialize database**
   ```bash
   python3 -c "from penpal_english_bot import init_db, init_game_tables; init_db(); init_game_tables()"
   ```

6. **Run dev bot**
   ```bash
   python3 penpal_english_bot.py
   ```

## Environment Variables

### DEV (Local)
- **BOT_TOKEN**: Dev bot token from BotFather
- **DATABASE_URL**: `postgresql://username@localhost:5432/penpal_english_dev`
- **USE_WEBHOOK**: `false` (uses polling)

### PROD (Heroku)
- **BOT_TOKEN**: Production bot token (set via `heroku config`)
- **DATABASE_URL**: Heroku PostgreSQL (auto-configured)
- **USE_WEBHOOK**: `true` (uses webhook)

## Useful Commands

```bash
# Check Heroku config
heroku config -a penpal-english

# View Heroku logs
heroku logs --tail -a penpal-english

# Access Heroku database
heroku pg:psql -a penpal-english

# Access local dev database
psql penpal_english_dev

# Stop local bot
pkill -f "python3.*penpal_english_bot.py"
```

## Database Management

```bash
# Backup local dev database
pg_dump penpal_english_dev > backup_dev.sql

# Restore from backup
psql penpal_english_dev < backup_dev.sql

# Drop and recreate dev database
dropdb penpal_english_dev
createdb penpal_english_dev
python3 -c "from penpal_english_bot import init_db, init_game_tables; init_db(); init_game_tables()"
```

## Troubleshooting

### PostgreSQL not running
```bash
brew services start postgresql@14
```

### Port already in use
```bash
# Find process using port
lsof -i :8443

# Kill process
kill -9 <PID>
```

### Database connection issues
```bash
# Check PostgreSQL status
brew services list | grep postgresql

# Restart PostgreSQL
brew services restart postgresql@14
```
