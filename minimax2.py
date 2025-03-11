import numpy as np
from game import Othello

# Définition des constantes
EMPTY = 0
BLACK = 1
WHITE = -1

COEFFICIENTS = {
    "piece_diff": 12,
    "mobility": 8
}

def improved_evaluate(board, current_player):
    """
        Evaluation function that combines:
        1. Piece difference,
        2. Board positional weights, and
        3. Mobility (number of legal moves).

        Parameters:
            board: the game state of the board
            player: the current player, white or black.

        Returns:
            COEFFICIENTS["piece_diff"] * piece_diff + board_weight_score + COEFFICIENTS["mobility"] * mobility:
            float containing the evaluation value representing the desirability of the game state.
        """
    board_weights = np.array([
        [100, -20, 10, 5, 5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [10, -2, 1, 2, 2, 1, -2, 10],
        [5, -2, 2, -1, -1, 2, -2, 5],
        [5, -2, 2, -1, -1, 2, -2, 5],
        [10, -2, 1, 2, 2, 1, -2, 10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10, 5, 5, 10, -20, 100]
    ])

    piece_diff = np.sum(board == current_player) - np.sum(board == -current_player)

    board_weight_score = np.sum(board_weights * (board == current_player)) - np.sum(board_weights * (board == -current_player))

    game = Othello()
    game.board = board.copy()
    mobility = len(game.get_valid_moves(current_player)) -- len(game.get_valid_moves(-current_player))

    return COEFFICIENTS["piece_diff"] * piece_diff + board_weight_score + COEFFICIENTS["mobility"] * mobility


def improved_minimax(board, depth, maximizing, player, root_player):
    """
    Recursive alpha-beta pruning algorithm using the improved evaluation function.

    Parameters:
        board: the current board state
        depth: the depth level of the algorithm
        maximizing: True if maximizing player, false if minimizing player (opponent).
        player: Color of the current player, white or black.
        root_player: Color of the AI's pieces.

    Returns:
        (max_eval/min_eval, best_move): a tuple containing what the evaluation of the best move is as well as the
        corresponding best move.
    """

    game = Othello()
    game.board = board.copy()

    if depth == 0 or game.is_game_over():
        return improved_evaluate(game.board, root_player), None

    valid_moves = game.get_valid_moves(player)

    if not valid_moves:
        return improved_evaluate(game.board, root_player), None

    best_move = None

    if maximizing:
        max_eval = float("-inf")

        for move in valid_moves:
            new_game = Othello()
            new_game.board = board.copy()
            new_game.apply_move(move, player)
            eval_score, _ = improved_minimax(new_game.board, depth - 1, False, -player, root_player)

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

        return max_eval, best_move

    else:
        min_eval = float("inf")

        for move in valid_moves:
            new_game = Othello()
            new_game.board = board.copy()
            new_game.apply_move(move, player)
            eval_score, _ = improved_minimax(new_game.board, depth - 1, True, -player, root_player)

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

        return min_eval, best_move


# Helper function to simplify launching in the main othello.py.
def improved_minimax_ai(board, player):
    _, best_move = improved_minimax(board, 6, True, player, player)
    return best_move
