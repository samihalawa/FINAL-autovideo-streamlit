from setuptools import setup, find_packages

setup(
    name="your_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'package_name>=1.2.3',
        'another_package>=2.0.0',
    ],
    python_requires='>=3.7',
) 