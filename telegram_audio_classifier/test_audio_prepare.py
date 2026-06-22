from pathlib import Path

from bot.config import PROCESSED_DIR, TARGET_DURATION_SECONDS, TARGET_SAMPLE_RATE
from model_service.audio_utils import prepare_audio_to_wav


input_path = Path("data/voices/test.wav")
output_path = PROCESSED_DIR / "test_prepared.wav"

info = prepare_audio_to_wav(
    input_path=input_path,
    output_path=output_path,
    target_sample_rate=TARGET_SAMPLE_RATE,
    target_duration_seconds=TARGET_DURATION_SECONDS,
)

print(info)
