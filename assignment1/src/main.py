#!/usr/bin/env python3
"""
main file

entry point for running assignment 1
"""

from clean import clean
from train import train


def main() -> None:
    """
    main entry point
    """
    clean_data = clean()
    train(clean_data)


if __name__ == '__main__':
    main()
