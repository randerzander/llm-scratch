from setuptools import setup, find_packages

setup(
    name='llm_scratch',
    version='0.1.5',
    packages=find_packages(),
    description='A collection of LLM utils',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Randy Gelhausen',
    author_email='rgelhau@gmail.com',
    url='https://github.com/randerzander/llm_scratch',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
