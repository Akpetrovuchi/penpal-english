# Настройка ЮKassa для приёма платежей

## 1. Получение credentials из ЮKassa

1. Зайди в [личный кабинет ЮKassa](https://yookassa.ru/)
2. Перейди в раздел **Настройки** → **Данные для API**
3. Скопируй:
   - **shopId** (идентификатор магазина)
   - **Секретный ключ** (secret key)

## 2. Настройка переменных окружения

Добавь в `.env`:

```env
# YooKassa Configuration
YOOKASSA_SHOP_ID=123456
YOOKASSA_SECRET_KEY=live_xxxxxxxxxxxxxx

# Webhook Configuration (for production)
USE_WEBHOOK=false  # true для продакшена с webhook
WEBHOOK_URL=https://your-app-name.herokuapp.com/yookassa/webhook
PORT=8443
```

## 3. Настройка webhook в ЮKassa (для продакшена)

### Для локального тестирования:
Используй `USE_WEBHOOK=false` - бот будет работать в режиме polling, пользователи будут нажимать кнопку "Проверить оплату".

### Для продакшена на Heroku:

1. **Получи URL твоего приложения:**
   ```
   https://your-app-name.herokuapp.com
   ```

2. **Настрой webhook в ЮKassa:**
   - Зайди в [личный кабинет ЮKassa](https://yookassa.ru/)
   - Перейди в **Настройки** → **Уведомления**
   - Включи **HTTP-уведомления**
   - Добавь URL: `https://your-app-name.herokuapp.com/yookassa/webhook`
   - Выбери события: `payment.succeeded`, `payment.canceled`

3. **Обнови `.env` на Heroku:**
   ```bash
   heroku config:set USE_WEBHOOK=true -a your-app-name
   heroku config:set WEBHOOK_URL=https://your-app-name.herokuapp.com/yookassa/webhook -a your-app-name
   heroku config:set YOOKASSA_SHOP_ID=your_shop_id -a your-app-name
   heroku config:set YOOKASSA_SECRET_KEY=your_secret_key -a your-app-name
   ```

## 4. Тестирование

### Тестовые карты ЮKassa:

- **Успешная оплата:** `5555 5555 5555 4477`
  - CVC: любой
  - Срок: любой будущий
  
- **Отклонённая оплата:** `5555 5555 5555 4444`

### Режим тестирования:

В личном кабинете ЮKassa есть переключатель **Тестовый режим / Боевой режим**. 

Для тестирования используй **Тестовый режим** с тестовыми credentials.

## 5. Как работает оплата

### С webhook (продакшен):
1. Пользователь выбирает тариф
2. Бот создаёт платёж в ЮKassa
3. Пользователь переходит на страницу оплаты
4. После оплаты ЮKassa отправляет webhook на твой сервер
5. Бот автоматически активирует подписку и уведомляет пользователя

### Без webhook (локально):
1. Пользователь выбирает тариф
2. Бот создаёт платёж в ЮKassa
3. Пользователь переходит на страницу оплаты
4. После оплаты пользователь возвращается в бота и нажимает "Проверить оплату"
5. Бот проверяет статус и активирует подписку

## 6. Мониторинг платежей

Все платежи сохраняются в таблице `payments`:

```sql
SELECT * FROM payments ORDER BY created_at DESC LIMIT 10;
```

Поля:
- `user_id` - ID пользователя Telegram
- `payment_id` - ID платежа в ЮKassa
- `amount` - сумма платежа
- `status` - статус (pending, succeeded, canceled)
- `plan` - тариф (monthly, yearly)
- `created_at` - дата создания
- `paid_at` - дата оплаты

## 7. Безопасность

✅ Секретный ключ хранится в переменных окружения (не в коде)
✅ Webhook использует HTTPS
✅ Все платежи логируются в БД
✅ ЮKassa проверяет подпись webhook-запросов

## 8. Troubleshooting

### Платёж не активируется автоматически:
- Проверь, что webhook настроен в ЮKassa
- Проверь логи: `heroku logs --tail -a your-app-name`
- Убедись, что `USE_WEBHOOK=true` на Heroku

### Ошибка "YooKassa credentials not set":
- Проверь, что `YOOKASSA_SHOP_ID` и `YOOKASSA_SECRET_KEY` установлены
- Для локального: в `.env`
- Для Heroku: `heroku config -a your-app-name`

### Webhook не приходит:
- Проверь URL в настройках ЮKassa
- Убедись, что URL доступен извне
- Проверь, что приложение запущено: `heroku ps -a your-app-name`
