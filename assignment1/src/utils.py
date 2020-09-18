#!/usr/bin/env python3
"""
utils functions
"""

from glob import glob
from os.path import abspath, join
from loguru import logger
from typing import List
from pathlib import Path


def get_glob(glob_rel_path: str) -> List[str]:
    """
    get glob file list for given path
    """
    logger.info("getting files using glob")
    complete_path: str = join(
        abspath(join(Path(__file__).absolute(), '../..')), glob_rel_path)
    files = glob(complete_path)
    return files
