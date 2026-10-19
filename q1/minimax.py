import math
import random

from .state import QuartoState


def evaluate_board(state, player):
    """Pontua vitórias e linhas de três peças com atributo compartilhado."""
    winner = state.check_win()
    if winner == player:
        return 1000
    if winner == 3 - player:
        return -1000

    score = 0
    for line in QuartoState.LINES:
        pieces = [state.board[i] for i in line if state.board[i] is not None]
        if len(pieces) == 3:
            p0, p1, p2 = pieces
            if (p0 & p1 & p2) > 0 or ((~p0 & ~p1 & ~p2) & 15) > 0:
                score += 10

    return score if state.current_player == player else -score


def minimax(state, depth, alpha, beta, maximizing_player, root_player):
    if depth == 0 or state.is_terminal():
        return evaluate_board(state, root_player), None

    legal_actions = state.get_legal_actions()
    best_action = random.choice(legal_actions)

    if maximizing_player:
        best_score = -math.inf
        for action in legal_actions:
            score, _ = minimax(
                state.apply_action(action), depth - 1, alpha, beta,
                False, root_player,
            )
            if score > best_score:
                best_score, best_action = score, action
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return best_score, best_action

    best_score = math.inf
    for action in legal_actions:
        score, _ = minimax(
            state.apply_action(action), depth - 1, alpha, beta,
            True, root_player,
        )
        if score < best_score:
            best_score, best_action = score, action
        beta = min(beta, score)
        if beta <= alpha:
            break
    return best_score, best_action
