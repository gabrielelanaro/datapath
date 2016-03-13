import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "trails",
    version = "0.0.1",
    author = "Gabriele Lanaro",
    author_email = "gabriele.lanaro@gmail.com",
    description = ("Design and structure data pipelines"),
    install_requires=['dask','joblib'],
    license = "MIT",
    keywords = "data pipelines",
    url = "https://github.com/gabrielelanaro/trails",
    packages=['trails', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)
