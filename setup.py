from setuptools import setup, find_packages
import os


def find_datafiles(path):
    return [
        (os.path.join("etc", d), [os.path.join(d, f) for f in files])
        for d, folders, files in os.walk(path)
    ]


setup(
    name="nmprepr",
    version="0.2",
    description="Neural Motion Planning",
    packages=find_packages(),
    data_files=find_datafiles("mpenv/assets"),
)
