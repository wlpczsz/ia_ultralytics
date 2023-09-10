# ia_ultralytics YOLO 🚀, AGPL-3.0 license

from ia_ultralytics.utils import emojis


class HUBModelError(Exception):

    def __init__(self, message='Model not found. Please check model URL and try again.'):
        """Create an exception for when a model is not found."""
        super().__init__(emojis(message))
