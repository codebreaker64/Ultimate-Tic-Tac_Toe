from utils import State, Action

import time
import numpy as np

class StudentAgent:
    Score = float
    Move = tuple[int, int, int, int]
    
    def __init__(self):
        """Instantiates your agent.
        """

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        start_time = time.time()
        time_limit = 2.5
        
        best_score = float('-inf')
        best_move = None
        
        for move in state.get_all_valid_actions():
            if time.time() - start_time > time_limit:
                break
            new_state = state.change_state(move)
            score, _ = self.minimax(state=new_state, depth=3, alpha=float('-inf'), beta=float('inf'), maximizingPlayer=False)

            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def minimax (self, state: State, depth: int, alpha: Score, beta: Score, maximizingPlayer: bool) -> tuple[Score, Move]:
        best_move = None
        
        if depth == 0 or state.is_terminal():
            return self.evaluate(state), best_move

        if maximizingPlayer:
            maxEval = float('-inf')
            for move in state.get_all_valid_actions():
                child = state.change_state(move)
                eval, _ = self.minimax(child, depth - 1, alpha, beta, False)
                if eval > maxEval:
                    maxEval = eval
                    best_move = move
                alpha = max(alpha, maxEval)
                if beta <= alpha:
                        break 
            return maxEval, best_move

        else:
            minEval = float('inf')
            for move in state.get_all_valid_actions():
                child = state.change_state(move)
                eval, _ = self.minimax(child, depth - 1, alpha, beta, True)
                if eval < minEval:
                    minEval = eval
                    best_move = move
                beta = min(beta, minEval)
                if beta <= alpha:
                    break  
            return minEval, best_move
        
    def evaluate(self, state: State) -> int:
        def two_connected(line, player):
            return (
                np.count_nonzero(line == player) == 2 and np.count_nonzero(line == 0) == 1
            )

        def score_two_connected(board, player, weight):
            score = 0
            for i in range(3):
                score += two_connected(board[i], player) * weight
                score += two_connected([board[0][i], board[1][i], board[2][i]], player) * weight
            diag1 = [board[0][0], board[1][1], board[2][2]]
            diag2 = [board[0][2], board[1][1], board[2][0]]
            score += two_connected(diag1, player) * weight
            score += two_connected(diag2, player) * weight
            return score

        def control_center(board, player, weight):
            return weight if board[1][1] == player else 0

        def control_corners(board, player, weight):
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            return sum(weight for i, j in corners if board[i][j] == player)

        def has_free_move():
            if state.prev_local_action is None:
                return True
            prev_row, prev_col = state.prev_local_action
            return state.local_board_status[prev_row][prev_col] != 0

        def count_forks(board, player):
            forks = 0
            lines = [
                [board[0][0], board[0][1], board[0][2]], 
                [board[1][0], board[1][1], board[1][2]],
                [board[2][0], board[2][1], board[2][2]],
                [board[0][0], board[1][0], board[2][0]], 
                [board[0][1], board[1][1], board[2][1]],
                [board[0][2], board[1][2], board[2][2]],
                [board[0][0], board[1][1], board[2][2]], 
                [board[0][2], board[1][1], board[2][0]],
            ]
            two_counts = 0
            for line in lines:
                if np.count_nonzero(line == player) == 2 and np.count_nonzero(line == 0) == 1:
                    two_counts += 1
            return 1 if two_counts >= 2 else 0 

        def mobility_bonus():
            open_boards = sum(1 for i in range(3) for j in range(3) if state.local_board_status[i][j] == 0)
            return open_boards * WEIGHTS["MOBILITY"]

        def meta_board_occupation(player):
            return np.count_nonzero(state.local_board_status == player)

        WEIGHTS = {
            "GLOBAL_TWO_CONNECTED": 250,    # Prioritize creating winning connections
            "BLOCKING_THREAT": 3000,        # Lower defensive focus
            "LOCAL_TWO_CONNECTED": 35,      # Still important, but less focus than global
            "GLOBAL_CENTER": 150,           # Aggressively control the center
            "LOCAL_CENTER": 25,             # Prioritize local centers for offense
            "GLOBAL_CORNER": 40,            # Corner control for wider reach
            "LOCAL_CORNER": 10,             # Corner placements can create forks
            "FREE_MOVE": 70,                # Free move should help offensive play
            "FORK_BONUS": 1000,             # High reward for fork creation
            "MOBILITY": 20,                 # More freedom for attack
            "WIN_BONUS": 100000,            # Large win bonus
            "META_WIN_CHANCE": 300,         # Favor winning the meta-board
            "META_CONTROL": 200,            # Focus on controlling the meta-board
            "CENTER_BOARD_CONTROL": 130     # Center control is critical for offense
        }  


        if state.is_terminal():
            if state.terminal_utility() == 1.0:
                return WEIGHTS["WIN_BONUS"]
            elif state.terminal_utility() == 0.0:
                return -WEIGHTS["WIN_BONUS"]
            else:
                return 0

        score = 0
        status = state.local_board_status
        board = state.board

        # Near-win on meta-board
        score += score_two_connected(status, 1, WEIGHTS["META_WIN_CHANCE"])
        score -= score_two_connected(status, 2, WEIGHTS["BLOCKING_THREAT"])

        # Meta-board occupation
        score += meta_board_occupation(1) * WEIGHTS["META_CONTROL"]
        score -= meta_board_occupation(2) * WEIGHTS["META_CONTROL"]

        # Center board control
        center_board = board[1][1]
        center_status = status[1][1]
        if center_status == 1:
            score += WEIGHTS["CENTER_BOARD_CONTROL"]
        elif center_status == 2:
            score -= WEIGHTS["CENTER_BOARD_CONTROL"]
        else:
            score += score_two_connected(center_board, 1, WEIGHTS["LOCAL_TWO_CONNECTED"])
            score -= score_two_connected(center_board, 2, WEIGHTS["LOCAL_TWO_CONNECTED"])

        # Existing global heuristics
        score += control_center(status, 1, WEIGHTS["GLOBAL_CENTER"])
        score -= control_center(status, 2, WEIGHTS["GLOBAL_CENTER"])
        score += control_corners(status, 1, WEIGHTS["GLOBAL_CORNER"])
        score -= control_corners(status, 2, WEIGHTS["GLOBAL_CORNER"])
        score += count_forks(status, 1) * WEIGHTS["FORK_BONUS"]
        score -= count_forks(status, 2) * WEIGHTS["FORK_BONUS"]
        score += mobility_bonus()

        # Local board heuristics
        for i in range(3):
            for j in range(3):
                if status[i][j] != 0:
                    continue
                local = board[i][j]
                score += score_two_connected(local, 1, WEIGHTS["LOCAL_TWO_CONNECTED"])
                score -= score_two_connected(local, 2, WEIGHTS["LOCAL_TWO_CONNECTED"])
                score += control_center(local, 1, WEIGHTS["LOCAL_CENTER"])
                score -= control_center(local, 2, WEIGHTS["LOCAL_CENTER"])
                score += control_corners(local, 1, WEIGHTS["LOCAL_CORNER"])
                score -= control_corners(local, 2, WEIGHTS["LOCAL_CORNER"])

        # Free move
        if has_free_move():
            score += WEIGHTS["FREE_MOVE"]

        return score
