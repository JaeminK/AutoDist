from setuptools import setup, find_packages

setup(
    name='autodist',  # Replace with your desired package name
    version='0.1',
    packages=find_packages(where='src'),  # Automatically find packages in the src directory
    package_dir={'': 'src'},
    install_requires=[
        'torch',
    ],
)