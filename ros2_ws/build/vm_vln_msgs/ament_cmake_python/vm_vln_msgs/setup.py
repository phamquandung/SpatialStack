from setuptools import find_packages
from setuptools import setup

setup(
    name='vm_vln_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('vm_vln_msgs', 'vm_vln_msgs.*')),
)
