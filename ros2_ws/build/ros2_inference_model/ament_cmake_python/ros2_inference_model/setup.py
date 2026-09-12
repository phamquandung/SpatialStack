from setuptools import find_packages
from setuptools import setup

setup(
    name='ros2_inference_model',
    version='0.0.0',
    packages=find_packages(
        include=('ros2_inference_model', 'ros2_inference_model.*')),
)
