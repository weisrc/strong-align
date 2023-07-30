import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="strong-align",
    py_modules=["strong_align"],
    version="0.0.1",
    description="Forced alignment using Wav2Vec2",
    readme="README.md",
    python_requires=">=3.8",
    author="Wei",
    url="https://github.com/weisrc/strong-align",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
)