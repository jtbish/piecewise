import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="piecewise",
    version="0.1.2",
    author="Jordan T. Bishop",
    author_email="jordanbishop26@gmail.com",
    description="Michigan-style Learning Classifier Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtbish/piecewise",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "pandas"
    ],
    python_requires='>=3.6',
)
