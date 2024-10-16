from setuptools import setup, find_packages

setup(
    name="pyautograd",
    version="2.0.0",
    packages=find_packages(),
    install_requires=["numpy"],
    author="Affan Hamid",
    author_email="affanhamid007@gmail.com",
    description="A complete autodiff library in python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/affanhamid/pyautograd",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
