from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name='inverted_encoding',
    version='0.0.19',
    description='Implementation of inverted encoding model as described in Scotti, Chen, & Golomb',
    packages=['inverted_encoding'],
    url="https://github.com/paulscotti/inverted_encoding",
    author="Paul S. Scotti",
    author_email="scottibrain@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "sklearn"],
)

# pip3 install check-manifest twine 
# pip3 install -e .

# python setup.py bdist_wheel sdist

# check-manifest --c
# git add MANIFEST.in

# twine upload --skip-existing dist/*
# or try cookiecutter
# tutorial on publishing python packages: www.youtube.com/watch?v=GIF3LaRqgXo