"""
Module to keep constant values for end to end tests
"""
import os

class End2EndError(Exception):
    """End2End test exception"""

CONFIG_PREFIX = "end2end_component_"

FEDBIOMED_SCRIPTS = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "..", "scripts")
)
