import numpy as np
import math
import random
from game import Othello

EMPTY = 0
BLACK = 1
WHITE = -1


class monte_carlo_node:
    """
    Node class for the Monte Carlo Tree Search algorithm. It contains all the information as well as functions
    that are needed to perform the search.

    get_moves: returns the different possible moves that the player can make at that position.
    is_fully_expanded: Checks if any of the possible moves have not been explored further.
    is_terminal_state: Checks if the board is in a terminal state (game over).
    best_child: Determines the best child node using the UCT formula. Source: https://stackoverflow.com/questions/36664993/mcts-uct-with-a-scoring-system
    expand: expands all possible nodes from the selected node. Each node is selected at random.
    update: updates the total number of simulated wins and visits to be used in the evaluation.
    """
    def __init__(self, board, player, move=None, parent=None):
        self.board = board.copy()
        self.player = player
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = self.get_moves()

    def get_moves(self):
        game = Othello()
        game.board = self.board.copy()
        moves = game.get_valid_moves(self.player)

        if not moves:
            moves.append(None)
        return moves

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal_state(self):
        game = Othello()
        game.board = self.board.copy()
        return game.is_game_over()

    # https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
    def best_child(self, c_param=math.sqrt(2)):
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        new_board = self.board.copy()
        game = Othello()
        game.board = new_board
        if move is not None:
            game.apply_move(move, self.player)

        child_node = monte_carlo_node(game.board, -self.player, move, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result


def deploy_MC(board, player):
    """
    Function to deploy the Monte Carlo Tree Search.

    Parameters:
        board: the current board
        player: the current player

    Returns:
        A different value based on the obtained score. If the score is greater than 0 (more of the player's pieces
        than the opponents), then it will return 1. If it is a tie, then it will return 0.5. If it is lesser, then
        it will return 0.
    """
    game = Othello()
    game.board = board.copy()
    current_player = player

    while not game.is_game_over():
        valid_moves = game.get_valid_moves(current_player)
        if not valid_moves:
            current_player = -current_player
            continue

        move = random.choice(valid_moves)
        if move is not None:
            game.apply_move(move, current_player)

        current_player = -current_player

    score = np.sum(game.board == player) - np.sum(game.board == -player)

    if score > 0:
        return 1.0
    elif score == 0:
        return 0.5
    else:
        return 0.0


def monte_carlo_tree_search(root, iterations=10000):
    """
    The actual tree search, using the information contained within the node.

    Parameters:
        root: the root node of the tree.
        iterations: the total number of iterations that MC will go through, defined as 10000 for this project.

    Returns:
        best_move: the best possible move from the node after all possible iterations.

    """
    for _ in range(iterations):
        node = root

        while not node.is_terminal_state() and node.is_fully_expanded():
            node = node.best_child()

        if not node.is_terminal_state():
            node = node.expand()

        result = deploy_MC(node.board, node.player)

        while node is not None:
            node.update(result)
            result = 1 - result
            node = node.parent

    best_move = None
    best_visits = -1
    for child in root.children:
        if child.visits > best_visits:
            best_visits = child.visits
            best_move = child.move

    return best_move


# Helper function to simplify launching in the main othello.py.
def monte_carlo_ai(board, player, iterations=10000):
    root = monte_carlo_node(board, player)
    return monte_carlo_tree_search(root, iterations)
