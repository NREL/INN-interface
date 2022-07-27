from setuptools import setup

from os import path
from io import open


metadata = dict(
    name="INN_interface",
    version="0.1.4",
    description="Neural net and airfoil interface",
    author="Andrew Glaws",
    packages=["INN_interface"],
    python_requires=">=3.8",
    zip_safe=True,
    )

setup(**metadata)
