import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gello",
    version="0.0.1",
    author="Xiaodao Lin",
    author_email="Xiaodao_lin@protonmail.com",
    description="Gello and FACTR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/z-yf17/galbot.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    license="MIT",
    install_requires=[
        "numpy",
    ],
)
