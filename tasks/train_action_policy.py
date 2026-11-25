"""Train the independent dual-wrist action policy."""

from xense0vla.training.action_policy import ActionPolicyTraining
from xense0vla.training.trainer import train


if __name__ == "__main__":
    train(ActionPolicyTraining())
