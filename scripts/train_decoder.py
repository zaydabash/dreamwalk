#!/usr/bin/env python3
"""
Train Decoder Script

Trains the neural-decoder models (EEG-to-CLIP decoder, emotion classifier,
motif detector) on synthetic data and saves checkpoints for the service to load.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICE_DIR = REPO_ROOT / "services" / "neural-decoder"
sys.path.insert(0, str(SERVICE_DIR))

from decoders.eeg_to_clip import EEGToCLIPDecoder  # noqa: E402
from decoders.emotion_classifier import EmotionClassifier  # noqa: E402
from decoders.motif_detector import MotifDetector  # noqa: E402
from models.decoder_models import DecoderConfig  # noqa: E402
from synthetic_data.generator import SyntheticDataGenerator  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_BUILDERS = {
    "eeg_to_clip": lambda config: EEGToCLIPDecoder(config=config),
    "emotion_classifier": lambda config: EmotionClassifier(config=config),
    "motif_detector": lambda config: MotifDetector(config=config),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, applying a YAML config file if given"""
    parser = argparse.ArgumentParser(description="Train DreamWalk neural-decoder models on synthetic data")
    parser.add_argument("--config", type=Path, default=None, help="YAML file overriding the defaults below")
    parser.add_argument("--samples", type=int, default=10000, help="Number of synthetic training samples")
    parser.add_argument("--channels", type=int, default=8, help="Number of EEG channels to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data generation")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_BUILDERS),
        default=list(MODEL_BUILDERS),
        help="Which models to train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models" / "checkpoints",
        help="Directory to save trained model checkpoints",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.config:
        _apply_config_file(args)

    return args


def _apply_config_file(args: argparse.Namespace) -> None:
    """Override parsed args in place with values from a YAML config file"""
    import yaml

    with open(args.config) as f:
        overrides: Dict[str, Any] = yaml.safe_load(f) or {}

    for key in ("samples", "channels", "seed", "device", "models"):
        if key in overrides:
            setattr(args, key, overrides[key])
    if "output_dir" in overrides:
        args.output_dir = REPO_ROOT / overrides["output_dir"]


async def train_all(args: argparse.Namespace) -> None:
    """Generate synthetic training data and train/save each requested model"""
    config = DecoderConfig(device=args.device)
    generator = SyntheticDataGenerator(seed=args.seed)

    logger.info("Generating %d synthetic training samples (%d channels)", args.samples, args.channels)
    training_data = await generator.generate_training_data(n_samples=args.samples, n_channels=args.channels)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        logger.info("Training %s", name)
        model = MODEL_BUILDERS[name](config)
        await model.train_synthetic(training_data)

        checkpoint_path = args.output_dir / f"{name}.pth"
        await model.save_model(str(checkpoint_path))
        logger.info("Saved %s checkpoint to %s", name, checkpoint_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(train_all(args))


if __name__ == "__main__":
    main()
