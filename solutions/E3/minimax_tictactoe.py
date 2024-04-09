import numpy as np
import copy
from utils import check_board, generate_next_moves


def MiniMaxTicTacToe(board: np.ndarray, depth: int, move_max: bool):
    pay = check_board(board, (1 if not move_max else -1))
    if pay or not depth:
        return pay * (depth + 1)
    best = -1000 if move_max else 1000
    moves = generate_next_moves(board)
    for move in moves:
        new_board = copy.copy(board)
        new_board[*move] = 1 if move_max else -1
        if move_max:
            best = max(best, MiniMaxTicTacToe(new_board, depth - 1, not move_max))
        else:
            best = min(best, MiniMaxTicTacToe(new_board, depth - 1, not move_max))
    return best


def AlfaBeta(board: np.ndarray, depth: int, move_max: bool, alfa=-1e6, beta=1e6):
    pay = check_board(board, (1 if not move_max else -1))
    if pay or not depth:
        return pay * (depth + 1)
    moves = generate_next_moves(board)
    if move_max:
        for move in moves:
            new_board = copy.copy(board)
            new_board[*move] = 1 if move_max else -1
            alfa = max(alfa, AlfaBeta(new_board, depth - 1, not move_max, alfa, beta))
            if alfa >= beta:
                return alfa
        return alfa
    else:
        for move in moves:
            new_board = copy.copy(board)
            new_board[*move] = 1 if move_max else -1
            beta = min(beta, AlfaBeta(new_board, depth - 1, not move_max, alfa, beta))
            if alfa >= beta:
                return beta
        return beta
