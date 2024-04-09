import numpy as np
from itertools import product
import random
def check_board(board:np.ndarray,who_is_moving:int):
    result = who_is_moving*10
    if np.count_nonzero(board)==9:
        return 0.01*who_is_moving
    for row in board:
        if (row==who_is_moving).sum()==3:
            return result
    for row in board.transpose():
        if (row==who_is_moving).sum()==3:
            return result
    if (board.diagonal()==who_is_moving).sum()==3:
        return result
    elif (np.fliplr(board).diagonal()==who_is_moving).sum()==3:
        return result
    return 0


def generate_next_moves(board:np.ndarray):
    states = []
    comb =  product([0,1,2],[0,1,2])
    for i in comb:
        if not board[*i]:
            states.append(i)
    return states
