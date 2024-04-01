import random
class TicTacToe:
    X = -1
    O = 1
    str_x = "X"
    str_o = "O"
    PLAYER = 1
    COMPUTER = 2

    def __init__(self) -> None:
        self.board = [0,0,0,0,0,0,0,0,0]
        self.file = open("game.log","+a")
        self.whose_turn = 0
    
    def close_file(self):
        self.file.close()

    def write_log(self,msg:str):
        self.file.writelines(msg)

    def update_whose_turn(self):
        if self.whose_turn==self.PLAYER:
            self.whose_turn = self.COMPUTER
        else:
            self.whose_turn = self.PLAYER
    
    def clear_board(self):
        self.board = [0,0,0,0,0,0,0,0,0]

    def check_finish(self,move:int):
        s1 = self.board[0:3].count(move)==3
        s2 = self.board[3:6].count(move)==3
        s3 = self.board[6:9].count(move)==3
        s4 = self.board[0:7:3].count(move)==3 
        s5 = self.board[1:8:3].count(move)==3 
        s6 = self.board[2:9:3].count(move)==3
        s7 = self.board[0:9:4].count(move)==3 
        s8 = self.board[2:7:2].count(move)==3
        if s1 or s2 or s3 or s4 or s5 or s6 or s7 or s8:
            return True
        return False

    def validate_choice(self,inp:str):
        try:
            move = int(inp)-1
            if not self.board[move]:
                self.board[move] = self.whose_turn
            else:
                self.kill_game()
        except:
            self.kill_game()
    def intro_msg(self):
        pass    
    def run(self):
        self.intro_msg()
        continue_game = True
        while continue_game:
            self.whose_turn = random.choice((TicTacToe.PLAYER,TicTacToe.COMPUTER))


    def kill_game(self):
        pass