from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from bot.config import (
    MAX_VOICE_DURATION_SECONDS,
    PROCESSED_DIR,
    TARGET_DURATION_SECONDS,
    TARGET_SAMPLE_RATE,
    VOICES_DIR,
)
from model_service.audio_utils import prepare_audio_to_wav
from model_service.model import format_prediction, predict_audio


router = Router()


@router.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "Привет! Я бот для классификации звуков.\n\n"
        "Отправь мне голосовое сообщение или загрузи .wav файл.\n"
        "Я подготовлю аудио, передам его в CNN-модель "
        "и верну predicted class, confidence и top-3 классов."
    )


@router.message(F.voice)
async def voice_handler(message: Message, bot: Bot):
    voice = message.voice

    if voice.duration > MAX_VOICE_DURATION_SECONDS:
        await message.answer(
            f"Голосовое слишком длинное.\n"
            f"Максимальная длительность: {MAX_VOICE_DURATION_SECONDS} секунд."
        )
        return

    await message.answer("Аудио получено. Обрабатываю...")

    user_id = message.from_user.id if message.from_user else 0
    message_id = message.message_id

    input_file_name = f"user_{user_id}_message_{message_id}.ogg"
    input_file_path: Path = VOICES_DIR / input_file_name

    output_file_name = f"user_{user_id}_message_{message_id}_prepared.wav"
    output_file_path: Path = PROCESSED_DIR / output_file_name

    try:
        telegram_file = await bot.get_file(voice.file_id)

        await bot.download_file(
            file_path=telegram_file.file_path,
            destination=input_file_path,
        )

        prepare_audio_to_wav(
            input_path=input_file_path,
            output_path=output_file_path,
            target_sample_rate=TARGET_SAMPLE_RATE,
            target_duration_seconds=TARGET_DURATION_SECONDS,
        )

        prediction = predict_audio(output_file_path)
        answer = format_prediction(prediction)

        await message.answer(answer)

    except Exception as exc:
        await message.answer(
            "Не удалось обработать голосовое сообщение.\n\n"
            f"Ошибка: {exc}"
        )


@router.message(F.document)
async def document_handler(message: Message, bot: Bot):
    document = message.document

    if not document.file_name:
        await message.answer(
            "Файл без имени. Попробуй отправить .wav файл еще раз."
        )
        return

    file_name_lower = document.file_name.lower()

    if not file_name_lower.endswith(".wav"):
        await message.answer(
            "Пока я принимаю только .wav файлы.\n"
            "Отправь файл с расширением .wav."
        )
        return

    await message.answer("Файл получен. Обрабатываю...")

    user_id = message.from_user.id if message.from_user else 0
    message_id = message.message_id

    input_file_name = f"user_{user_id}_message_{message_id}_{document.file_name}"
    input_file_path: Path = VOICES_DIR / input_file_name

    output_file_name = f"user_{user_id}_message_{message_id}_prepared.wav"
    output_file_path: Path = PROCESSED_DIR / output_file_name

    try:
        telegram_file = await bot.get_file(document.file_id)

        await bot.download_file(
            file_path=telegram_file.file_path,
            destination=input_file_path,
        )

        prepare_audio_to_wav(
            input_path=input_file_path,
            output_path=output_file_path,
            target_sample_rate=TARGET_SAMPLE_RATE,
            target_duration_seconds=TARGET_DURATION_SECONDS,
        )

        prediction = predict_audio(output_file_path)
        answer = format_prediction(prediction)

        await message.answer(answer)

    except Exception as exc:
        await message.answer(
            "Не удалось обработать .wav файл.\n\n"
            f"Ошибка: {exc}"
        )


@router.message(F.audio)
async def audio_handler(message: Message):
    await message.answer(
        "Ты отправил аудио как music/audio.\n\n"
        "Для теста отправь .wav именно как файл/document "
        "или отправь обычное voice message через микрофон Telegram."
    )


@router.message(F.text)
async def text_handler(message: Message):
    await message.answer(
        "Пока я принимаю только голосовые сообщения и .wav файлы.\n\n"
        "Отправь voice message или загрузи .wav файл."
    )


@router.message()
async def unknown_handler(message: Message):
    await message.answer(
        "Я пока умею принимать только voice message и .wav файлы."
    )
