from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_service.audio_utils import build_mel_spectrogram_for_model


CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / .. / "ml" /  "models" / "best_UrbanSoundCNN_dev_split.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        x = F.relu(x, inplace=True)

        return x


class UrbanSoundCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.4,
        fc_dim: int = 128,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = self.final(x)

        return x


def load_model() -> UrbanSoundCNN:
    """
    Загружает модель и веса.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    model = UrbanSoundCNN(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


MODEL = load_model()


def predict_audio(wav_path: Path) -> dict:
    """
    Полный inference pipeline:

    wav
    ↓
    mel-спектрограмма
    ↓
    CNN
    ↓
    probabilities
    ↓
    predicted class + confidence + top-3
    """

    mel_tensor = build_mel_spectrogram_for_model(wav_path)
    mel_tensor = mel_tensor.to(DEVICE)

    with torch.no_grad():
        logits = MODEL(mel_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=3)

    predicted_index = top_indices[0].item()
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = top_probs[0].item()

    top_3 = []

    for prob, index in zip(top_probs, top_indices):
        top_3.append(
            {
                "class_name": CLASS_NAMES[index.item()],
                "confidence": round(prob.item(), 4),
            }
        )

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "top_3": top_3,
    }


def format_prediction(prediction: dict) -> str:
    """
    Делает красивый текст для Telegram.
    """

    predicted_class = prediction["predicted_class"]
    confidence = prediction["confidence"]
    top_3 = prediction["top_3"]

    lines = [
        "Результат классификации аудио:",
        "",
        f"Predicted class: {predicted_class}",
        f"Confidence: {round(confidence * 100, 2)}%",
        "",
        "Top-3 классов:",
    ]

    for i, item in enumerate(top_3, start=1):
        class_name = item["class_name"]
        item_confidence = round(item["confidence"] * 100, 2)
        lines.append(f"{i}. {class_name} - {item_confidence}%")

    return "\n".join(lines)
