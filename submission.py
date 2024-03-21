from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random, time
import math


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    agent = env.get_robot(robot_id)

    available_packages = [p for p in env.packages if p.on_board]
    sorted_packages = sorted(available_packages, key=lambda p: manhattan_distance(agent.position, p.position))
    sorted_chargers = sorted(env.charge_stations,
                             key=lambda station: manhattan_distance(agent.position, station.position))

    if agent.package is None:
        if manhattan_distance(agent.position, sorted_packages[0].position) <= agent.battery:
            return 5 * agent.battery + 50 * agent.credit - manhattan_distance(agent.position,
                                                                              sorted_packages[0].position)
        else:
            return 5 * agent.battery + 50 * agent.credit - manhattan_distance(agent.position,
                                                                              sorted_chargers[0].position)
    else:
        if manhattan_distance(agent.position, agent.package.destination) <= agent.battery:
            return manhattan_distance(agent.package.position, agent.package.destination) - manhattan_distance(
                agent.position, agent.package.destination) + 5 * agent.battery + 50 * agent.credit + 35
        else:
            return 5 * agent.battery + 50 * agent.credit - manhattan_distance(agent.position,
                                                                              sorted_chargers[0].position)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def __init__(self, time_limit=1):
        self.time_limit = time_limit
        self.agent_id = None
    def RB_minimax(self, env: WarehouseEnv, agent_id, depth):
        if env.done() or depth == 0 or (time.time() - self.start_time) > self.time_limit:
            return smart_heuristic(env, agent_id)

        children = self.successors(env, agent_id)
        if agent_id == self.agent_id:  # Maximize for the first agent
            best_value = float('-inf')
            for child_env in children[1]:
                value = self.RB_minimax(child_env, (agent_id + 1) % 2, depth - 1)
                best_value = max(best_value, value)
            return best_value
        else:  # Minimize for the second agent
            best_value = float('inf')
            for child_env in children[1]:
                value = self.RB_minimax(child_env, (agent_id + 1) % 2, depth - 1)
                best_value = min(best_value, value)
            return best_value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        operators, children = self.successors(env, agent_id)
        best_operator = None
        best_value = float('-inf')
        self.agent_id = agent_id

        for op, child_env in zip(operators, children):
            value = self.RB_minimax(child_env, agent_id, depth=2)
            if value > best_value:
                best_value = value
                best_operator = op

        return best_operator


class AgentAlphaBeta(Agent):

    def __init__(self, time_limit=1):
        self.time_limit = time_limit
        self.agent_id = None

    def RB_alpha_beta(self, env: WarehouseEnv, agent_id, alpha, beta, depth):
        if env.done() or depth == 0 or (time.time() - self.start_time) > self.time_limit:
            return smart_heuristic(env, agent_id)

        children = self.successors(env, agent_id)
        if agent_id == self.agent_id:  # Maximize for the first agent
            best_value = float('-inf')
            for child_env in children[1]:
                value = self.RB_alpha_beta(child_env, (agent_id + 1) % 2, alpha, beta, depth - 1)
                best_value = max(best_value, value)
                alpha = max(best_value, alpha)
                if best_value >= beta:
                    return math.inf
            return best_value
        else:  # Minimize for the second agent
            best_value = float('inf')
            for child_env in children[1]:
                value = self.RB_alpha_beta(child_env, (agent_id + 1) % 2, alpha, beta, depth - 1)
                best_value = min(best_value, value)
                beta = min(best_value, beta)
                if best_value <= alpha:
                    return -math.inf
            return best_value

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        operators, children = self.successors(env, agent_id)
        best_operator = None
        best_value = float('-inf')
        self.agent_id = agent_id

        for op, child_env in zip(operators, children):
            value = self.RB_alpha_beta(child_env, agent_id, -math.inf, math.inf, depth=4)
            if value > best_value:
                best_value = value
                best_operator = op

        return best_operator


class AgentExpectimax(Agent):
    def __init__(self, time_limit=1):
        self.time_limit = time_limit
        self.agent_id = None

    def weight(self, op):
        if op in ['pick up', 'move east']:
            return 2
        return 1

    def RB_expectimax(self, env: WarehouseEnv, agent_id, depth):
        if (time.time() - self.start_time) > self.start_time or env.done() or depth == 0:
            return smart_heuristic(env, self.agent_id)

        children = self.successors(env, agent_id)

        if agent_id == self.agent_id:  # Maximize for the first agent
            best_value = float('-inf')

            for child_env in children[1]:
                value = self.RB_expectimax(child_env, (agent_id + 1) % 2, depth - 1)
                best_value = max(best_value, value)

            return best_value
        else:  # Minimize for the second agent
            ops = env.get_legal_operators(agent_id)

            ops_weights = [self.weight(op) for op in ops]
            probabilities = [float(op_weight) / sum(ops_weights) for op_weight in ops_weights]

            expect = 0

            for op, op_weight in zip(ops, probabilities):
                if (time.time() - self.start_time) > self.time_limit:
                    return smart_heuristic(env, self.agent_id)

                cloned_env = env.clone()
                cloned_env.apply_operator(agent_id, op)

                expect += op_weight * self.RB_expectimax(cloned_env, (agent_id + 1) % 2, depth - 1)

            return expect

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        operators, children = self.successors(env, agent_id)
        best_operator = None
        best_value = float('-inf')
        self.agent_id = agent_id

        for op, child_env in zip(operators, children):
            value = self.RB_expectimax(child_env, agent_id, depth=4)
            if value > best_value:
                best_value = value
                best_operator = op

        return best_operator


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)