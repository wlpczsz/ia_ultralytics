# ia_ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.168'

from ia_ultralytics.models import RTDETR, SAM, YOLO
from ia_ultralytics.models.fastsam import FastSAM
from ia_ultralytics.models.nas import NAS
from ia_ultralytics.utils import SETTINGS as settings
from ia_ultralytics.utils.checks import check_yolo as checks
from ia_ultralytics.utils.downloads import download

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings'
