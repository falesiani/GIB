from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='GIB',
    version='0.0.1',
    install_requires=['numpy',
                        'torch',
                        'scipy',
                        'sklearn',
                        'matplotlib',
                        'torchvision',
                        'joblib',
                        'pandas',
                        'tensorboard',
                        'blitz-bayesian-pytorch',
                        'quadprog',
                        'pytorch-lightning',
                        ],
    packages=['GIB'],
    author='Francesco Alesiani, Shujian Yu, Xi Yu',
    author_email='francesco.alesiani@neclab.eu',
    url='https://github.com/falesiani/GIB',
    zip_safe=False,
    description='Gated Information Bottleneck for Generalization in Sequential Environments',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: NEC Laboratories Europe GmbH",
    ],
)
