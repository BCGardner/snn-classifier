from setuptools import setup

setup(
    name='snn-classifier',
    version='0.2',
    author='Brian Gardner',
    author_email='brgardner@hotmail.co.uk',
    description='Supervised learning in multilayer spiking neural networks for pattern recognition',
    long_description=open('README.md').read(),
    url='https://github.com/BCGardner/snn-classifier',
    license='LICENSE.txt',
    packages=['snncls'],
    python_requires='>=3',
    install_requires=[
        "numpy",
        "matplotlib",
        "python-mnist",
        "scikit-learn",
        "scikit-image"
    ],
)
