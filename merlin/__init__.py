from __future__ import annotations

import dotenv
import os
import glob
import json
import importlib

from merlin.core import dataset

# Initialize environment variables to None
DATA_HOME = None
ANALYSIS_HOME = None
PARAMETERS_HOME = None
ANALYSIS_PARAMETERS_HOME = None
CODEBOOK_HOME = None
DATA_ORGANIZATION_HOME = None
POSITION_HOME = None
MICROSCOPE_PARAMETERS_HOME = None
FPKM_HOME = None
SNAKEMAKE_PARAMETERS_HOME = None

envPath = os.path.join(os.getenv("MERLIN_ENV_PATH") or os.path.expanduser('~'), '.merlinenv')

if os.path.exists(envPath):
    dotenv.load_dotenv(envPath)

    try:
        DATA_HOME = os.path.expanduser(os.environ.get('DATA_HOME'))
        ANALYSIS_HOME = os.path.expanduser(os.environ.get('ANALYSIS_HOME'))
        PARAMETERS_HOME = os.path.expanduser(os.environ.get('PARAMETERS_HOME'))
        ANALYSIS_PARAMETERS_HOME = os.sep.join(
                [PARAMETERS_HOME, 'analysis'])
        CODEBOOK_HOME = os.sep.join(
                [PARAMETERS_HOME, 'codebooks'])
        DATA_ORGANIZATION_HOME = os.sep.join(
                [PARAMETERS_HOME, 'dataorganization'])
        POSITION_HOME = os.sep.join(
                [PARAMETERS_HOME, 'positions'])
        MICROSCOPE_PARAMETERS_HOME = os.sep.join(
                [PARAMETERS_HOME, 'microscope'])
        FPKM_HOME = os.sep.join([PARAMETERS_HOME, 'fpkm'])
        SNAKEMAKE_PARAMETERS_HOME = os.sep.join(
            [PARAMETERS_HOME, 'snakemake'])

    except TypeError:
        print('MERlin environment appears corrupt. Please run ' +
              '\'merlin --configure .\' in order to configure the environment.')
else:
    print(f'Unable to find MERlin environment file at {envPath}. Please run ' +
          '\'merlin --configure .\' in order to configure the environment.')


def store_env(dataHome, analysisHome, parametersHome):
    with open(envPath, 'w') as f:
        f.write(f'DATA_HOME={dataHome}\n')
        f.write(f'ANALYSIS_HOME={analysisHome}\n')
        f.write(f'PARAMETERS_HOME={parametersHome}\n')


class IncompatibleVersionException(Exception):
    pass


def version():
    try:
        import importlib.metadata
        return importlib.metadata.version('merlin')
    except ImportError:
        # Fallback for Python < 3.8
        import pkg_resources
        return pkg_resources.get_distribution('merlin').version


def is_compatible(testVersion: str, baseVersion: str = None) -> bool:
    """ Determine if testVersion is compatible with baseVersion

    Args:
        testVersion: the version identifier to test, as the string 'x.y.z'
            where x is the major version, y is the minor version,
            and z is the patch.
        baseVersion: the version to check testVersion's compatibility. If  not
            specified then the current MERlin version is used as baseVersion.
    Returns: True if testVersion are compatible, otherwise false.
    """
    if baseVersion is None:
        baseVersion = version()
    return testVersion.split('.')[0] == baseVersion.split('.')[0]


def get_analysis_datasets(maxDepth=2) -> list[dataset.DataSet]:
    """ Get a list of all datasets currently stored in analysis home.

    Args:
        maxDepth: the directory depth to search for datasets.
    Returns: A list of the dataset objects currently within analysis home.
    """
    metadataFiles = []
    for d in range(1, maxDepth+1):
        metadataFiles += glob.glob(os.path.join(
            ANALYSIS_HOME, *['*']*d, 'dataset.json'))

    def load_dataset(jsonPath) -> dataset.DataSet:
        with open(jsonPath, 'r') as f:
            metadata = json.load(f)
            analysisModule = importlib.import_module(metadata['module'])
            analysisTask = getattr(analysisModule, metadata['class'])
            return analysisTask(metadata['dataset_name'])

    return [load_dataset(m) for m in metadataFiles]
