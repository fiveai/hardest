import os
from typing import List

from setuptools import find_packages
from setuptools import setup

NAME = "hardest"
__version__ = None

repository_dir = os.path.dirname(__file__)


def filter_requirements_args(requirements: List[str]) -> List[str]:
    """Filter out arguments from a requirements list."""
    return [line for line in requirements if not line.strip().startswith("--") and len(line.strip()) > 0]


with open(os.path.join(repository_dir, "README.md")) as fh:
    README = fh.read()

with open(os.path.join(repository_dir, "requirements.txt")) as fh:
    requirements = filter_requirements_args(fh.readlines())

with open(os.path.join(repository_dir, "tests-requirements.txt")) as fh:
    tests_requirements = filter_requirements_args(fh.readlines())

setup(
    name=NAME,
    description="The HARDness ESTimation package: ranks an object detection dataset by the expected hardness of the "
    "images.",
    long_description_content_type="text/markdown",
    long_description=README,
    author="Jonathan Sadeghi",
    author_email="jonathan.sadeghi@five.ai",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: POSIX :: Linux",
    ],
    include_package_data=True,
    install_requires=requirements,
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.0.0",  # TOOD: do this the github ci way
    test_suite="tests",
    tests_require=tests_requirements,
    python_requires=">=3.8",
    extras_require={"test": tests_requirements},
)
