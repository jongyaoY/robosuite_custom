# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="robosuite_custom",
    packages=[
        package
        for package in find_packages()
        if package.startswith("robosuite_custom")
    ],
    install_requires=[
        # "stable-baselines3[extra]==2.0.0",
        # "gymnasium==0.28.1",
        # "open3d",
        # "moviepy==1.0.0",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="robosuite_custom: Add-ons for robosuite",
    author="Zongyao Yi",
    url="https://github.com/jongyaoY/robosuite_custom",
    author_email="zongyao.yi@dfki.de",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
