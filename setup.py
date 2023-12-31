# ia_ultralytics YOLO 🚀, AGPL-3.0 license

import re
from pathlib import Path

import pkg_resources as pkg
from setuptools import find_packages, setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]


def get_version():
    file = PARENT / 'ia_ultralytics/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]


setup(
    name='ia_ultralytics',  # name of pypi package
    version=get_version(),  # version of pypi package
    python_requires='>=3.8',
    license='AGPL-3.0',
    description=('ia_ultralytics YOLOv8 for SOTA object detection, multi-object tracking, instance segmentation, '
                 'pose estimation and image classification.'),
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/ia_ultralytics/ia_ultralytics',
    project_urls={
        'Bug Reports': 'https://github.com/ia_ultralytics/ia_ultralytics/issues',
        'Funding': 'https://ia_ultralytics.com',
        'Source': 'https://github.com/ia_ultralytics/ia_ultralytics'},
    author='鹏程万里转瞬至',
    author_email='1744182303@qq.com',
    packages=find_packages(),  # required
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require={
        'dev': [
            'ipython',
            'check-manifest',
            'pytest',
            'pytest-cov',
            'coverage',
            'mkdocs-material',
            'mkdocstrings[python]',
            'mkdocs-redirects',  # for 301 redirects
            'mkdocs-ultralytics-plugin>=0.0.27',  # for meta descriptions and images, dates and authors
        ],
        'export': [
            'coremltools>=7.0.b1',
            'openvino-dev>=2023.0',
            'tensorflowjs',  # automatically installs tensorflow
        ], },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows', ],
    keywords='machine-learning, deep-learning, vision, ML, DL, AI, YOLO, YOLOv3, YOLOv5, YOLOv8, HUB, ia_ultralytics',
    entry_points={'console_scripts': ['ia_yolo = ia_ultralytics.cfg:entrypoint', 'ia_ultralytics = ia_ultralytics.cfg:entrypoint']})
