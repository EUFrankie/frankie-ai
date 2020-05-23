import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

all_packages = setuptools.find_packages()

setuptools.setup(
    name="frankie-ai",
    version="0.0.1",
    author="Jan Kleine-Klatte",
    author_email="jakl5749@gmail.com",
    description="A package to use Frankie AI components.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EUFrankie/frankie-ai",
    packages=all_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy >= 1.18.4',
        'pandas >= 1.0.3',
        'tensorflow == 2.2.0',
        'transformers == 2.9.1'
    ],
    python_requires='>=3.6',
)