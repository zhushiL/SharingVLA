"""Independent wrist/state robot controller."""

from .policy import RobotPolicy


class ActionPolicy(RobotPolicy):
    expected_standalone = True
