from setuptools import setup, find_packages
import codecs
import os

current = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(current, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.1.0"
DESCRIPTION = "Offloading CPU intensive model scoring to a pool of workers"

# Setting up
setup(
    name="MLPool",
    version=VERSION,
    author="ET",
    author_email="<titov.1994@list.ru>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[],
    keywords=["python", "machine learning", "multiprocessing"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
