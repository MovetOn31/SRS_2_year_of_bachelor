from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(file_path: Path) -> tuple[torch.Tensor, int]:
    """
    Загружает аудиофайл через soundfile.

    Возвращает:
    - waveform: torch.Tensor формы [channels, samples]
    - sample_rate: частота дискретизации
    """
    audio, sample_rate = sf.read(str(file_path), dtype="float32")

    if audio.ndim == 1:
        waveform = torch.from_numpy(audio).unsqueeze(0)
    else:
        waveform = torch.from_numpy(audio.T)

    return waveform, sample_rate


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Делает mono из stereo/multichannel.
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform


def resample_audio(
    waveform: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """
    Приводит аудио к нужной частоте дискретизации.
    """
    if original_sample_rate == target_sample_rate:
        return waveform

    resampler = torchaudio.transforms.Resample(
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
    )

    return resampler(waveform)


def fix_audio_length(
    waveform: torch.Tensor,
    target_sample_rate: int,
    target_duration_seconds: int,
) -> torch.Tensor:
    """
    Приводит аудио к фиксированной длине.

    Если аудио длиннее - обрезает.
    Если короче - дополняет нулями.
    """
    target_num_samples = target_sample_rate * target_duration_seconds
    current_num_samples = waveform.shape[1]

    if current_num_samples > target_num_samples:
        waveform = waveform[:, :target_num_samples]

    elif current_num_samples < target_num_samples:
        padding_size = target_num_samples - current_num_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding_size))

    return waveform


def save_wav(
    waveform: torch.Tensor,
    sample_rate: int,
    output_path: Path,
) -> None:
    """
    Сохраняет mono wav.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = waveform.squeeze(0).cpu().numpy()
    sf.write(str(output_path), audio, sample_rate)


def prepare_audio_to_wav(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_duration_seconds: int,
) -> dict:
    """
    Полный pipeline подготовки аудио:

    1. Загружаем аудио.
    2. Делаем mono.
    3. Приводим к target_sample_rate.
    4. Обрезаем или дополняем до target_duration_seconds.
    5. Сохраняем в wav.
    """

    waveform, original_sample_rate = load_audio(input_path)

    original_channels = waveform.shape[0]
    original_num_samples = waveform.shape[1]
    original_duration = original_num_samples / original_sample_rate

    waveform = convert_to_mono(waveform)

    waveform = resample_audio(
        waveform=waveform,
        original_sample_rate=original_sample_rate,
        target_sample_rate=target_sample_rate,
    )

    waveform = fix_audio_length(
        waveform=waveform,
        target_sample_rate=target_sample_rate,
        target_duration_seconds=target_duration_seconds,
    )

    save_wav(
        waveform=waveform,
        sample_rate=target_sample_rate,
        output_path=output_path,
    )

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_sample_rate": original_sample_rate,
        "target_sample_rate": target_sample_rate,
        "original_channels": original_channels,
        "target_channels": 1,
        "original_duration_seconds": round(original_duration, 3),
        "target_duration_seconds": target_duration_seconds,
        "output_num_samples": waveform.shape[1],
    }


def build_mel_spectrogram_for_model(
    wav_path: Path,
    target_sample_rate: int = 22050,
    target_duration_seconds: int = 4,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Строит mel-спектрограмму так же, как в ноутбуке обучения.

    На выходе:
    torch.Tensor формы [1, 1, 128, time]
    """

    target_num_samples = target_sample_rate * target_duration_seconds

    y, _ = librosa.load(
        str(wav_path),
        sr=target_sample_rate,
        mono=True,
    )

    if len(y) < target_num_samples:
        y = np.pad(y, (0, target_num_samples - len(y)), mode="constant")
    else:
        y = y[:target_num_samples]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=target_sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    mel_tensor = torch.tensor(mel, dtype=torch.float32)

    # Было: [128, time]
    # Нужно CNN: [batch_size, channels, n_mels, time]
    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)

    return mel_tensor
