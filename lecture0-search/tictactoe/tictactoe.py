"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None

players = {X: 1, O: -1}


def copy_board(board):
    return [
        board[0][:],
        board[1][:],
        board[2][:]
    ]


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    counter = {
        X: 0,
        O: 0
    }
    has_empty = False

    for list in board:
        for value in list:
            if value == EMPTY:
                has_empty = True
            else:
                counter[value] += 1

    if not has_empty:
        return 'the game is already over'

    return min(counter, key=counter.get)


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_moves = set()

    for i, list in enumerate(board):
        for j, value in enumerate(list):
            if value == EMPTY:
                possible_moves.add((i, j))
    
    return possible_moves if possible_moves else None


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = copy_board(board)
    row = action[0]
    col = action[1]

    if not (0 <= row < len(new_board)):
        raise NameError(f'Row {row} does not exist')
    
    if not (0 <= col < len(new_board[row])):
        raise NameError(f'Column {col} does not exist')
    
    if new_board[row][col] is not EMPTY:
        raise NameError(f'Cell ({row},{col}) is already filled')
    
    new_board[row][col] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        target_row = board[i][0]
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return target_row
        
        target_col = board[0][i]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return target_col

    target_diagonal1 = board[0][0]
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return target_diagonal1

    target_diagonal2 = board[0][2]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return target_diagonal2


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return actions(board) is None or winner(board) is not None


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_player = winner(board)
    return players.get(winner_player, 0)


def min_value(board):
    if (terminal(board)):
        return (utility(board), None)

    v = None
    optimal_action = None

    for action in actions(board):
        max = max_value(result(board, action))
        if v is None:
            v = max[0]
            optimal_action = action
        elif max[0] < v:
            v = max[0]
            optimal_action = action
    
    return (v, optimal_action)


def max_value(board):
    if (terminal(board)):
        return (utility(board), None)

    v = None
    optimal_action = None

    for action in actions(board):
        min = min_value(result(board, action))
        
        if v is None:
            v = min[0]
            optimal_action = action
        elif min[0] > v:
            v = min[0]
            optimal_action = action

    return (v, optimal_action)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if (terminal(board)):
        return None

    if player(board) == X:
        return max_value(board)[1]
    else:
        return min_value(board)[1]
