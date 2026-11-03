"""Train the complete VLA model."""

from xense0vla.training.trainer import train
from xense0vla.training.vla import VLATraining


if __name__ == "__main__":
    train(VLATraining())
