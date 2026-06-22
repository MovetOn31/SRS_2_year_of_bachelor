# важно: положить файл .env с токеном в папку (и не пушить .env в github иначе придется менять токен)
telegram_audio_classifier/.env

## Как запустить

### 1. Склонировать репозиторий

```bash
git clone <repo_url>
cd telegram_audio_classifier
```

### 2. Создать виртуальное окружение

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Скопировать `.env`

# важно: положить файл .env с токеном в папку (и не пушить .env в github иначе придется менять токен)
telegram_audio_classifier/.env

### 5. Проверить модель без Telegram

Положить тестовый файл сюда:

```text
data/voices/test.wav
```

Запустить подготовку аудио:

```bash
python test_audio_prepare.py
```

Запустить предсказание модели:

```bash
python test_model_predict.py
```

### 6. Запустить Telegram-бота

```bash
python -m bot.main
```

В Telegram написать боту:

```text
/start
```

Потом отправить voice message или `.wav` файл.
