#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()


setup(
    name="pytest-gherkin",
    version="0.1.7",
    author="Nick Farrell",
    author_email="nicholas.farrell@gmail.com",
    maintainer="Nick Farrell",
    maintainer_email="nicholas.farrell@gmail.com",
    license="MIT",
    url="https://github.com/nicois/pytest-gherkin",
    description="A flexible framework for executing BDD gherkin tests",
    long_description=read("README.rst"),
    packages=["pytest_gherkin"],
    python_requires=">=3.6",
    install_requires=["pytest>=5.0.0", "gherkin-official>=4.1.3"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: Pytest",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing :: BDD",
        "Topic :: Text Processing",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    entry_points={"pytest11": ["gherkin = pytest_gherkin.plugin"]},
)
