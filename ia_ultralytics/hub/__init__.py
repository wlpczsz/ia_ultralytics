# ia_ultralytics YOLO 🚀, AGPL-3.0 license

import requests

from ia_ultralytics.data.utils import HUBDatasetStats
from ia_ultralytics.hub.auth import Auth
from ia_ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX
from ia_ultralytics.utils import LOGGER, SETTINGS


def login(api_key=''):
    """
    Log in to the ia_ultralytics HUB API using the provided API key.

    Args:
        api_key (str, optional): May be an API key or a combination API key and model ID, i.e. key_id

    Example:
        ```python
        from ia_ultralytics import hub

        hub.login('API_KEY')
        ```
    """
    Auth(api_key, verbose=True)


def logout():
    """
    Log out of ia_ultralytics HUB by removing the API key from the settings file. To log in again, use 'yolo hub login'.

    Example:
        ```python
        from ia_ultralytics import hub

        hub.logout()
        ```
    """
    SETTINGS['api_key'] = ''
    SETTINGS.save()
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo hub login'.")


def reset_model(model_id=''):
    """Reset a trained model to an untrained state."""
    r = requests.post(f'{HUB_API_ROOT}/model-reset', json={'apiKey': Auth().api_key, 'modelId': model_id})
    if r.status_code == 200:
        LOGGER.info(f'{PREFIX}Model reset successfully')
        return
    LOGGER.warning(f'{PREFIX}Model reset failure {r.status_code} {r.reason}')


def export_fmts_hub():
    """Returns a list of HUB-supported export formats."""
    from ia_ultralytics.engine.exporter import export_formats
    return list(export_formats()['Argument'][1:]) + ['ia_ultralytics_tflite', 'ia_ultralytics_coreml']


def export_model(model_id='', format='torchscript'):
    """Export a model to all formats."""
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(f'{HUB_API_ROOT}/v1/models/{model_id}/export',
                      json={'format': format},
                      headers={'x-api-key': Auth().api_key})
    assert r.status_code == 200, f'{PREFIX}{format} export failure {r.status_code} {r.reason}'
    LOGGER.info(f'{PREFIX}{format} export started ✅')


def get_export(model_id='', format='torchscript'):
    """Get an exported model dictionary with download URL."""
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(f'{HUB_API_ROOT}/get-export',
                      json={
                          'apiKey': Auth().api_key,
                          'modelId': model_id,
                          'format': format})
    assert r.status_code == 200, f'{PREFIX}{format} get_export failure {r.status_code} {r.reason}'
    return r.json()


def check_dataset(path='', task='detect'):
    """
    Function for error-checking HUB dataset Zip file before upload. It checks a dataset for errors before it is
    uploaded to the HUB. Usage examples are given below.

    Args:
        path (str, optional): Path to data.zip (with data.yaml inside data.zip). Defaults to ''.
        task (str, optional): Dataset task. Options are 'detect', 'segment', 'pose', 'classify'. Defaults to 'detect'.

    Example:
        ```python
        from ia_ultralytics.hub import check_dataset

        check_dataset('path/to/coco8.zip', task='detect')  # detect dataset
        check_dataset('path/to/coco8-seg.zip', task='segment')  # segment dataset
        check_dataset('path/to/coco8-pose.zip', task='pose')  # pose dataset
        ```
    """
    HUBDatasetStats(path=path, task=task).get_json()
    LOGGER.info(f'Checks completed correctly ✅. Upload this dataset to {HUB_WEB_ROOT}/datasets/.')
