import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tqa_training_lib",
    version="0.0.2",
    author="Luke Westfall et al",
    author_email="lwestfall1@gmail.com",
    description="Common functions for tweetqa model training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sweng480-team23/tqa-training-lib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
