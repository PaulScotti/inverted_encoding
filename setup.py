from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name='inverted_encoding',
    version='0.0.1',
    description='Implementation of inverted encoding model as described in Scotti, Chen, & Golomb',
    py_modules=["inverted_encoding"],
    package_dir={'':'src'},
    url="https://github.com/paulscotti/inverted_encoding",
    author="Paul S. Scotti",
    author_email="scottibrain@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "sklearn.model_selection",
    ],
)

# pip3 install check-manifest twine 

# python setup.py bdist_wheel sdist

# check-manifest --create
# git add MANIFEST.in

# twine upload dist/*
# or try cookiecutter
# tutorial on publishing python packages: www.youtube.com/watch?v=GIF3LaRqgXo