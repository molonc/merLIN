import os
import setuptools

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS :: MacOS X",
    "License  :: Cofidential",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

install_requires = [line.rstrip() for line in open(
    os.path.join(os.path.dirname(__file__), "requirements.txt"))]

dependency_links = [\
        'git+https://github.com/spacetx/starfish.git@master#egg=startfish-0']

setuptools.setup(
    name="merfish",
    version="0.0.1",
    description="Pipelines and pipeline components for the analysis of image-based transcriptomics data",
    author="George Emanuel",
    author_email="emanuega0@gmail.com",
    license="Confidential",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    entry_points={
        'console_scripts': "merfish=merfish_code.merfish_code:merfish"
    },
    classifiers=CLASSIFIERS
)
