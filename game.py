import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1
"""
game.py est simplement une copie du class Othello qui a été défini et donné dans le code de base dans othello.py.
J'ai simplement fait une copie dans un autre fichier pour éviter le circular call (ce qui m'a causé des problèmes).
"""
class Othello:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3, 3], self.board[4, 4] = WHITE, WHITE
        self.board[3, 4], self.board[4, 3] = BLACK, BLACK
        self.current_player = BLACK

    def get_valid_moves(self, player):
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(row, col, player):
                    valid_moves.append((row, col))
        return valid_moves

    def is_valid_move(self, row, col, player):
        if self.board[row, col] != EMPTY:
            return False
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            flipped = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == -player:
                flipped.append((r, c))
                r += dr
                c += dc
            if flipped and 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
                return True
        return False

    def apply_move(self, move, player):
        if move is None:
            return
        row, col = move
        self.board[row, col] = player  # Place la pièce
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            flipped = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == -player:
                flipped.append((r, c))
                r += dr
                c += dc
            if flipped and 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
                for fr, fc in flipped:
                    self.board[fr, fc] = player  # Retourne les pièces

    def is_game_over(self):
        return not self.get_valid_moves(BLACK) and not self.get_valid_moves(WHITE)

