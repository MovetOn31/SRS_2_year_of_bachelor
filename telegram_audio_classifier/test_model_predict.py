from pathlib import Path

from model_service.model import format_prediction, predict_audio


wav_path = Path("data/processed/test_prepared.wav")

prediction = predict_audio(wav_path)

print(prediction)
print()
print(format_prediction(prediction))
