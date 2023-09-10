import importlib
import sys

from ia_ultralytics.utils import LOGGER

# Set modules in sys.modules under their old name
sys.modules['ia_ultralytics.yolo.engine'] = importlib.import_module('ia_ultralytics.engine')

LOGGER.warning("WARNING ⚠️ 'ia_ultralytics.yolo.engine' is deprecated since '8.0.136' and will be removed in '8.1.0'. "
               "Please use 'ia_ultralytics.engine' instead.")
