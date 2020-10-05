#!/usr/bin/env python
"""
tests for helpers
"""

#################################
# for handling relative imports #
#################################
if __name__ == '__main__':
    import sys
    from pathlib import Path
    current_file = Path(__file__).resolve()
    root = next(elem for elem in current_file.parents
                if str(elem).endswith('src'))
    sys.path.append(str(root))
    # remove the current file's directory from sys.path
    try:
        sys.path.remove(str(current_file.parent))
    except ValueError:  # Already removed
        pass
#################################

import numpy as np
import matplotlib.pyplot as plt 
from word2vec import sigmoid
from utils.utils import softmax, normalizeRows
from typing import Any
from loguru import logger


def test_sigmoid(x: Any, name: str) -> None:
    """
    test sigmoid implementation
    """
    y = sigmoid(x)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Sigmoid {name}")
    plt.show()


def test_softmax(x: Any) -> None:
    """
    test softmax implementation
    """
    res = softmax(x)
    for row in res:
        row_sum = np.sum(row)
        if not np.isclose(row_sum, 1):
            logger.info(f'{row}: {row_sum}')
            raise RuntimeError('sum of softmax not close to 1')
    logger.info(f'\nsoftmax:\n{res}')


def test_normalize(x: Any) -> None:
    """
    test normalization function
    """
    logger.info(f'\nnormalization input:\n{x}')
    res = normalizeRows(x)
    logger.info(f'\nnormalization output:\n{res}')


def test_all():
    """
    test all helper functions
    """
    test_normalize(np.array([[8, 7, 9], [16, 18, 22]]))
    test_normalize([[3, 2, 1], [1, 2, 3]])
    test_softmax([[3, 2, 1], [1, 2, 3]])
    test_softmax(np.array([[8, 7, 9], [16, 18, 22]]))
    test_sigmoid(np.linspace(-10, 10, 100), 'numpy array')
    test_sigmoid(np.linspace(-10, 10, 100).tolist(), 'scalar')


if __name__ == "__main__":
    test_all()
