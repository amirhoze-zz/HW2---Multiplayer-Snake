from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from scipy.spatial.distance import cityblock
from enum import Enum
import copy


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    if not state.snakes[player_index].alive:
        return 0


    max_dis = state.board_size.height + state.board_size.width

    if len(state.fruits_locations) == 0:
        return state.snakes[player_index].length * max_dis

    closest_fruit_dis = min(cityblock(state.snakes[player_index].head, fruit) for fruit in state.fruits_locations)
    closest_snake_dis = min(cityblock(state.snakes[player_index].head, s.head) for s in state.snakes if s.index != player_index)

    greedy_val = (1/closest_fruit_dis) * max_dis + (state.snakes[player_index].length * max_dis) - (1/closest_snake_dis)
    return greedy_val


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """
    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """
        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def get_action(self, state: GameState) -> GameAction:
        max_val = -np.inf
        best_action = None
        for my_action in [GameAction.LEFT, GameAction.STRAIGHT, GameAction.RIGHT]:
            next_state = self.TurnBasedGameState(state, my_action)
            minimax_res = self.rb_multi_minimax(next_state, 3)
            if minimax_res > max_val:
                best_action = my_action
                max_val = minimax_res
        return best_action

    def rb_multi_minimax(self, turnState: TurnBasedGameState, d: int):
        if turnState.game_state.is_terminal_state:
            if turnState.game_state.current_winner.player_index == self.player_index:
                return np.inf
            else:
                return -np.inf
        if d == 0:
            return heuristic(turnState.game_state, self.player_index)

        if turnState.turn == self.Turn.AGENT_TURN:
            curMax = -np.inf
            for action in turnState.game_state.get_possible_actions(self.player_index):
                child_state = self.TurnBasedGameState(turnState.game_state, action)
                child_state_minimax_val = self.rb_multi_minimax(child_state, d-1)
                curMax = max(curMax, child_state_minimax_val)
            return curMax
        else:
            curMin = np.inf
            for opponents_actions in turnState.game_state.get_possible_actions_dicts_given_action(turnState.agent_action, self.player_index):
                next_state = get_next_state(turnState.game_state, opponents_actions)
                child_state = self.TurnBasedGameState(next_state, None)
                child_state_minimax_val = self.rb_multi_minimax(child_state, d-1)
                curMin = min(curMin, child_state_minimax_val)
            return curMin

class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        max_val = -np.inf
        best_action = None
        for my_action in [GameAction.LEFT, GameAction.STRAIGHT, GameAction.RIGHT]:
            next_state = self.TurnBasedGameState(state, my_action)
            minimax_res = self.alpha_beta(next_state, 3, -np.inf, np.inf)
            if minimax_res > max_val:
                best_action = my_action
                max_val = minimax_res
        return best_action
        pass

    def alpha_beta(self,turnState: MinimaxAgent.TurnBasedGameState, d, a, b):
        if turnState.game_state.is_terminal_state:
            if turnState.game_state.current_winner.player_index == self.player_index:
                return np.inf
            else:
                return -np.inf
        if d == 0:
            return heuristic(turnState.game_state, self.player_index)

        if turnState.turn == self.Turn.AGENT_TURN:
            curMax = -np.inf
            for action in turnState.game_state.get_possible_actions(self.player_index):
                child_state = self.TurnBasedGameState(turnState.game_state, action)
                child_state_minimax_val = self.alpha_beta(child_state, d-1, a, b)
                curMax = max(curMax, child_state_minimax_val)
                a = max(a, curMax)
                if curMax >= b:
                    return np.inf
            return curMax
        else:
            curMin = np.inf
            for opponents_actions in turnState.game_state.get_possible_actions_dicts_given_action(turnState.agent_action, self.player_index):
                next_state = get_next_state(turnState.game_state, opponents_actions)
                child_state = self.TurnBasedGameState(next_state, None)
                child_state_minimax_val = self.alpha_beta(child_state, d-1, a, b)
                curMin = min(curMin, child_state_minimax_val)
                if curMin <= a:
                    return -np.inf
            return curMin

def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    N = 50
    limit_sideways_steps = 10
    sideways = 0 # TODO-amir what is this variable
    movement_options = [GameAction.STRAIGHT, GameAction.RIGHT, GameAction.LEFT]
    #a state is a vector of length N
    #start state is random
    state_game_strategy = np.random.choice(movement_options, N, p=[1 / 3, 1 / 3, 1 / 3])
    current_state_game_strategy = state_game_strategy
    # a for loop for deciding the movement in each turn
    for i in range(0, N): #TODO-amir should we start fromm 0?
        state_game_strategy = current_state_game_strategy
        best_score = -float("inf")
        best_states = []
        for move in movement_options:
            temp_state = copy.deepcopy(state_game_strategy)
            temp_state[i] = move
            new_score = get_fitness(tuple.convert(copy.deepcopy(temp_state))) #TODO-amir check if this conversion is valid
            if new_score > best_score:
                best_score = new_score
                state_game_strategy = temp_state
                best_states = [state_game_strategy]
            elif new_score == best_score:
                best_states.append(temp_state)
        if best_score > get_fitness(tuple.convert(copy.deepcopy(current_state_game_strategy))): #TODO-amir check if this conversion is valid
            current_state_game_strategy = best_states[np.random.choice(range(len(best_states)))]
            #reset sideways
            sideways = 0
        elif best_score == get_fitness(tuple.convert(copy.deepcopy(current_state_game_strategy))) and sideways <= limit_sideways_steps: #TODO-amir check if this conversion is valid
            current_state_game_strategy = best_states[np.random.choice(range(len(best_states)))]
            sideways = sideways + 1
    print(current_state_game_strategy)


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()

