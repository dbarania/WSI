import numpy as np
import random
from minimax_tictactoe import MiniMaxTicTacToe, AlfaBeta
from utils import check_board, generate_next_moves
import copy

temp_board = np.array([[-1, 1, -1], [-1, 1, 1], [0, 0, -1]])


class TicTacToe:
    PLAYER = -1
    COMPUTER = 1

    def __init__(self, dificulty:int) -> None:
        self.board = np.zeros((3, 3))
        self.move_counter = 0
        self.who_is_moving = None
        self.dificulty = dificulty

    @staticmethod
    def str_cell(value):
        if np.sign(value) == 1:
            return "X"
        elif np.sign(value) == -1:
            return "O"
        else:
            return " "

    def __str__(self) -> str:
        r = ""
        for row in self.board:
            s = f"{self.str_cell(row[0])}|{self.str_cell(row[1])}|{self.str_cell(row[2])}\n{'-' * 5}\n"
            r += s
        return r.rstrip("----\n")

    def clear_data(self):
        self.board = np.zeros((3, 3))
        self.move_counter = 0
        self.who_is_moving = None

    def player_move(self):
        while True:
            row = int(input("Choose a row: ")) - 1
            column = int(input("Choose column: ")) - 1
            if not self.board[row][column]:
                self.board[row][column] = self.who_is_moving
                return
            else:
                print("Invalid input try again")
                continue

    def computer_move(self,algorithm = MiniMaxTicTacToe,debug = False):
        moves = (generate_next_moves(self.board))
        random.shuffle(moves)
        if len(moves):
            best_move = moves[0]
            score = -100
            d = dict()
            for move in moves:
                new_board = copy.copy(self.board)
                new_board[*move] = 1

                result = algorithm(new_board, self.dificulty, False)
                if result > score:
                    best_move = move
                    score = result
                d[move] = result
            self.board[*best_move] = self.who_is_moving
            if debug:
                print(d)

    def update_move(self):
        if self.who_is_moving == self.COMPUTER:
            self.who_is_moving = self.PLAYER
        else:
            self.who_is_moving = self.COMPUTER
    def sum_game(self, result):
        if -1<result<1:
            print("The game enden as a draw")
        elif result<-1:
            print("The X has won!")
        else:
            print("The O has won!")

    def game(self):
        if not self.who_is_moving:
            self.who_is_moving = random.choice([self.PLAYER, self.COMPUTER])
        state_of_game = False
        print(self)
        print("\n")
        while not state_of_game:
            self.move_counter += 1
            match self.who_is_moving:
                case self.PLAYER:
                    self.player_move()
                case self.COMPUTER:
                    self.computer_move()
            state_of_game = check_board(self.board, self.who_is_moving)
            print(self)
            print("\n")
            self.update_move()
        self.sum_game(state_of_game)

if __name__=="__main__":
    game = TicTacToe(6)
    game.game()