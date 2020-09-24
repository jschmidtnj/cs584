#!/usr/bin/env python3
"""
classification type (books.py)
"""

from enum import Enum
from typing import Dict


class BookType(Enum):
    """
    Enum to store nlp types
    """
    dostoyevsky = "dostoyevsky"
    doyle = "doyle"
    austen = "austen"


class StartEnd:
    """
    start and end lines for book
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


start_end_map: Dict[BookType, StartEnd] = {
    BookType.dostoyevsky: StartEnd(186, 35959),
    BookType.doyle: StartEnd(62, 12681),
    BookType.austen: StartEnd(94, 80104),
}

class_map: Dict[BookType, str] = {
    BookType.dostoyevsky: 'Fyodor Dostoyevsky',
    BookType.doyle: 'Arthur Conan Doyle',
    BookType.austen: 'Jane Austen',
}
