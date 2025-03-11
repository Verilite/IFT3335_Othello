import numpy as np
from game import Othello

"""
Sources consulted:
https://www.ffothello.org/informatique/algorithmes/
Livre de Russell et Norvig: Artificial Intelligence: A Modern Approach, 4th US ed.
https://zzutk.github.io/docs/reports/2014.04%20-%20Searching%20Algorithms%20in%20Playing%20Othello.pdf
"""

# Global constants (make sure they match your main othello.py)
EMPTY = 0
BLACK = 1
WHITE = -1

# Global coefficients that can be modified at runtime
COEFFICIENTS = {
    "piece_diff": 12,  # Coefficient for piece difference
    "mobility": 8  # Coefficient for mobility
}


def improved_evaluate(board, current_player):
    """
    Evaluation function that combines:
    1. Piece difference,
    2. Board positional weights, and
    3. Mobility (number of legal moves).

    Parameters:
        board: the game state of the board
        current_player: the current player, white or black.

    Returns:
        COEFFICIENTS["piece_diff"] * piece_diff + board_weight_score + COEFFICIENTS["mobility"] * mobility:
        float containing the evaluation value representing the desirability of the game state.
    """
    board_weights = np.array([
        [100, -20, 10, 5, 5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [10, -2, 15, 2, 2, 15, -2, 10],
        [5, -2, 2, -1, -1, 2, -2, 5],
        [5, -2, 2, -1, -1, 2, -2, 5],
        [10, -2, 15, 2, 2, 15, -2, 10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10, 5, 5, 10, -20, 100]
    ])

    # Criterion 1: Piece difference
    piece_diff = np.sum(board == current_player) - np.sum(board == -current_player)

    # Criterion 2: Board weighted score
    board_weight_score = np.sum(board_weights * (board == current_player)) - np.sum(board_weights * (board == -current_player))

    # Criterion 3: Mobility – difference in the number of legal moves
    game = Othello()
    game.board = board.copy()
    mobility = len(game.get_valid_moves(current_player)) - len(game.get_valid_moves(-current_player))

    # Combine criteria using global coefficients
    return COEFFICIENTS["piece_diff"] * piece_diff + board_weight_score + COEFFICIENTS["mobility"] * mobility


def improved_alpha_beta(board, depth, alpha, beta, maximizing, player, root_player):
    """
    Recursive alpha-beta pruning algorithm using the improved evaluation function.

    Parameters:
        board: the current board state
        depth: the depth level of the algorithm
        alpha: Alpha value for pruning, defaults to negative infinity.
        beta: Beta value for pruning, defaults to positive infinity.
        maximizing: True if maximizing player, false if minimizing player (opponent).
        player: Color of the current player, white or black.
        root_player: Color of the AI's pieces.

    Returns:
        (max_eval/min_eval, best_move): a tuple containing what the evaluation of the best move is as well as the
        corresponding best move.
    """
    game = Othello()
    game.board = board.copy()

    # Terminal condition: reached depth limit or game over.
    if depth == 0 or game.is_game_over():
        return improved_evaluate(game.board, root_player), None

    valid_moves = game.get_valid_moves(player)
    if not valid_moves:
        # No legal moves: return the evaluation of the board.
        return improved_evaluate(game.board, root_player), None

    best_move = None

    if maximizing:
        max_eval = float("-inf")

        for move in valid_moves:
            new_game = Othello()
            new_game.board = board.copy()
            new_game.apply_move(move, player)
            eval_score, _ = improved_alpha_beta(new_game.board, depth - 1, alpha, beta, False, -player, root_player)

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break   # Beta cutoff

        return max_eval, best_move

    else:
        min_eval = float("inf")

        for move in valid_moves:
            new_game = Othello()
            new_game.board = board.copy()
            new_game.apply_move(move, player)
            eval_score, _ = improved_alpha_beta(new_game.board, depth - 1, alpha, beta, True, -player, root_player)

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta <= alpha:
                break   # Alpha cutoff

        return min_eval, best_move


# Helper function to simplify launching in the main othello.py.
def improved_alpha_beta_ai(board, player):
    """
    AI wrapper that uses the alpha-beta algorithm with a fixed depth of 7.

    Parameters:
        board: the current board state
        player: Color of the corresponding player. Can be white or black.

    Returns:
        best_move: the best possible move in the current state based on the alpha-beta pruning algorithm.
    """
    _, best_move = improved_alpha_beta(board, 7, float("-inf"), float("inf"), True, player, player)
    return best_move
