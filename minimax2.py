import numpy as np
# Importing the Othello class from the main module.
from othello import Othello

# Définition des constantes
EMPTY = 0
BLACK = 1
WHITE = -1
DEPTH = 6  # Profondeur du Minimax

COEFFICIENTS = {
    "piece_diff": 10,
    "mobility": 5
}

def improved_evaluate(board, player):
    board_weights = np.array([
        [500, -150, 30, 10, 10, 30, -150, 500],
        [-150, -250, 0, 0, 0, 0, -250, -150],
        [30, 0, 1, 2, 2, 1, 0, 30],
        [10, 0, 2, 16, 16, 2, 0, 10],
        [10, 0, 2, 16, 16, 2, 0, 10],
        [30, 0, 1, 2, 2, 1, 0, 30],
        [-150, -250, 0, 0, 0, 0, -250, -150],
        [500, -150, 30, 10, 10, 30, -150, 500]
    ])

    piece_diff = np.sum(board == player) - np.sum(board == -player)

    board_weight_score = np.sum(board_weights * (board == player)) - np.sum(board_weights * (board == -player))

    game = Othello()
    game.board = board.copy()
    mobility = len(game.get_valid_moves(player)) -- len(game.get_valid_moves(-player))

    return COEFFICIENTS["piece_diff"] * piece_diff + board_weight_score + COEFFICIENTS["mobility"] * mobility


def improved_minimax(board, depth, maximizing, player):

    game = Othello()
    game.board = board.copy()

    if depth == 0 or game.is_game_over():
        return improved_evaluate(game.board, player), None

    valid_moves = game.get_valid_moves(player)
    best_move = None

    if maximizing:
        max_eval = float("-inf")

        for move in valid_moves:
            new_board = board.copy()
            game.apply_move(move, player)
            eval_score, _ = improved_minimax(new_board, depth - 1, False, -player)

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

        return max_eval, best_move

    else:
        min_eval = float("inf")

        for move in valid_moves:
            new_board = board.copy()
            game.apply_move(move, player)
            eval_score, _ = improved_minimax(new_board, depth - 1, True, -player)

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

        return min_eval, best_move


def improved_minimax_ai(board, player):
    _, best_move = improved_minimax(board, DEPTH, True, player)
    return best_move
