from setuptools import setup, find_packages

setup(
    name="html2md",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["cffi"],
    package_data={
        "html2md": ["htmltomarkdown.so"],
    },
    include_package_data=True,
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python module to convert HTML to Markdown using Go",
    long_description="This module provides a Python interface to convert HTML to Markdown using a Go library.",
    url="https://github.com/yourusername/htmltomarkdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

