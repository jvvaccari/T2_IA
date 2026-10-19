import copy
import math
import random


class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.untried_actions = state.get_legal_actions()
        self.visits = 0
        self.wins = 0.0

    def ucb1(self, exploration_weight=1.41):
        if self.visits == 0 or self.parent is None:
            return math.inf
        return (
            self.wins / self.visits
            + exploration_weight
            * math.sqrt(math.log(self.parent.visits) / self.visits)
        )


def mcts(root_state, iterations):
    root = MCTSNode(root_state)

    for _ in range(iterations):
        node = root

        while not node.untried_actions and node.children:
            node = max(node.children, key=lambda child: child.ucb1())

        if node.untried_actions and not node.state.is_terminal():
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            child = MCTSNode(
                node.state.apply_action(action),
                parent=node,
                action_taken=action,
            )
            node.children.append(child)
            node = child

        simulated_state = copy.deepcopy(node.state)
        while not simulated_state.is_terminal():
            action = random.choice(simulated_state.get_legal_actions())
            simulated_state = simulated_state.apply_action(action)

        winner = simulated_state.check_win()
        while node is not None:
            node.visits += 1
            if winner == 3 - node.state.current_player:
                node.wins += 1
            elif winner is None:
                node.wins += 0.5
            node = node.parent

    return max(root.children, key=lambda child: child.visits).action_taken
