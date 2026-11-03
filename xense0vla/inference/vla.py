"""Task-conditioned complete VLA robot controller."""

from .policy import RobotPolicy


class VLAPolicy(RobotPolicy):
    expected_standalone = False
