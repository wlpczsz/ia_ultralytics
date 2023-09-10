import importlib
import sys

from ia_ultralytics.utils import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ia_ultralytics.yolo.v8'] = importlib.import_module('ia_ultralytics.models.yolo')

LOGGER.warning("WARNING ⚠️ 'ia_ultralytics.yolo.v8' is deprecated since '8.0.136' and will be removed in '8.1.0'. "
               "Please use 'ia_ultralytics.models.yolo' instead.")
