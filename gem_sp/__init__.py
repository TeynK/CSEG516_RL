# splendor_game/__init__.py

"""
Splender Game Logic Package
===========================

This package contains the core game logic for Splendor, implemented in pure Python.

Modules:
- constants.py: Game constants (scores, counts).
- card.py: Data structures for DevelopmentCards and NobleTiles.
- board.py: The game board state (gem stacks, face-up cards, nobles).
- player.py: The player state (gems, cards, score, reserved cards).
- actions.py: Action validation logic (e.g., is_valid_take_three).
- game.py: The main game engine (state machine, step(), reset()).

"""

from .constants import WINNING_SCORE, MAX_GEMS_PER_PLAYER
from .card import DevelopmentCard, NobleTile
from .board import Board
from .player import Player
from .game import SplendorGame

__all__ = [
    'SplendorGame',
    'Board',
    'Player',
    'DevelopmentCard',
    'NobleTile',
    'WINNING_SCORE',
    'MAX_GEMS_PER_PLAYER',
]