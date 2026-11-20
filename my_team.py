# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import sys
from collections import Counter, deque

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Actions, Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveTacticalAgent', second='DefensiveTacticalAgent', num_training=0):
    """
    Returns a team of two tactical agents.
    Creates a dedicated SharedTeamState per team instance to completely eliminate
    any crosstalk or color/territory confusion between teams or across matches.
    """
    shared = SharedTeamState(is_red)
    return [
        eval(first)(first_index, shared_state=shared),
        eval(second)(second_index, shared_state=shared)
    ]


# Backward compatibility aliases
OffensiveReflexAgent = None
DefensiveReflexAgent = None


#####################
# Shared Team State #
#####################

class SharedTeamState:
    """
    Shared blackboard for a team instance. Holds precomputed map topology,
    corridor entrance clustering (top, mid, bot), exact Bayesian belief tracking,
    food loss events, enemy invasion statistics, and opponent profiling.
    """
    def __init__(self, is_red):
        self.is_red = is_red
        self.initialized = False
        self.width = 0
        self.height = 0
        self.walls = None
        self.legal_positions = []
        self.legal_neighbors = {}

        # Boundaries
        self.home_border_x = 0
        self.enemy_border_x = 0
        self.home_boundaries = []
        self.enemy_boundaries = []

        # 3 Strategic Corridors: 'top', 'mid', 'bot'
        self.corridors = {'top': [], 'mid': [], 'bot': []}
        self.corridor_ambush_points = {'top': None, 'mid': None, 'bot': None}

        # Dead ends: pos -> (junction_pos, depth)
        self.dead_ends = {}

        # Belief tracking: opponent_index -> Counter of position probabilities
        self.beliefs = {}
        self.last_seen_pos = {}
        self.last_food_defending = None
        self.food_eaten_positions = deque(maxlen=5)

        # Entrance corridors and invasion tendency tracking
        self.entrance_stats = Counter()

        # Opponent behavioral profiling
        self.opp_turns_as_pacman = Counter()
        self.opp_total_turns = Counter()
        self.opp_food_eaten = Counter()

    def setup(self, game_state, is_red, distancer):
        self.is_red = is_red
        if self.initialized:
            return

        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height

        # Legal positions and adjacency
        self.legal_positions = [
            (x, y) for x in range(self.width)
            for y in range(self.height)
            if not self.walls[x][y]
        ]
        self.legal_neighbors = {}
        for pos in self.legal_positions:
            x, y = pos
            nbrs = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and not self.walls[nx][ny]:
                    nbrs.append((nx, ny))
            self.legal_neighbors[pos] = nbrs

        # Boundaries:
        # Red side: x < mid_x (0 to mid_x - 1). Red border is mid_x - 1.
        # Blue side: x >= mid_x (mid_x to width - 1). Blue border is mid_x.
        mid_x = self.width // 2
        if self.is_red:
            self.home_border_x = mid_x - 1
            self.enemy_border_x = mid_x
        else:
            self.home_border_x = mid_x
            self.enemy_border_x = mid_x - 1

        self.home_boundaries = [
            (self.home_border_x, y)
            for y in range(self.height)
            if not self.walls[self.home_border_x][y]
        ]
        self.enemy_boundaries = [
            (self.enemy_border_x, y)
            for y in range(self.height)
            if not self.walls[self.enemy_border_x][y]
        ]

        # Classify the 3 Strategic Corridors
        self._classify_corridors()

        # Dead-end precomputation
        self._precompute_dead_ends()

        # Initialize beliefs and profile counters
        opponents = game_state.get_blue_team_indices() if self.is_red else game_state.get_red_team_indices()
        for opp in opponents:
            start_pos = game_state.get_initial_agent_position(opp)
            self.beliefs[opp] = util.Counter()
            self.opp_total_turns[opp] = 0
            self.opp_turns_as_pacman[opp] = 0
            if start_pos is not None:
                self.beliefs[opp][start_pos] = 1.0
                self.last_seen_pos[opp] = start_pos
            else:
                for p in self.legal_positions:
                    if not self.is_home_side(p):
                        self.beliefs[opp][p] = 1.0
                self.beliefs[opp].normalize()

        self.initialized = True

    def get_zone(self, y):
        """
        Maps y-coordinate to one of three natural map sectors on defaultCapture:
        Top: y >= 11 (6-7 dots + Blue capsule @ (25, 10))
        Mid: 6 <= y <= 10 (7-8 dots + direct crossings)
        Bot: y <= 5 (6-7 dots + Red capsule @ (6, 5))
        """
        if y >= 11:
            return 'top'
        elif y >= 6:
            return 'mid'
        else:
            return 'bot'

    def _classify_corridors(self):
        self.corridors = {'top': [], 'mid': [], 'bot': []}
        for b in self.home_boundaries:
            z = self.get_zone(b[1])
            self.corridors[z].append(b)

        for z in ['top', 'mid', 'bot']:
            cells = self.corridors[z]
            if cells:
                cells_sorted = sorted(cells, key=lambda c: c[1])
                self.corridor_ambush_points[z] = cells_sorted[len(cells_sorted) // 2]
            else:
                self.corridor_ambush_points[z] = self.home_boundaries[0] if self.home_boundaries else None

    def _precompute_dead_ends(self):
        degree = {p: len(self.legal_neighbors[p]) for p in self.legal_positions}
        queue = deque([p for p, deg in degree.items() if deg <= 1])
        pruned = set()
        parent = {}

        while queue:
            u = queue.popleft()
            pruned.add(u)
            for v in self.legal_neighbors[u]:
                if v not in pruned:
                    degree[v] -= 1
                    parent[u] = v
                    if degree[v] == 1:
                        queue.append(v)

        for u in pruned:
            curr = u
            path = [curr]
            while curr in parent:
                curr = parent[curr]
                path.append(curr)
                if curr not in pruned:
                    break
            junction = curr
            for d, p in enumerate(reversed(path[:-1])):
                self.dead_ends[p] = (junction, len(path) - 1 - d)

    def is_home_side(self, pos):
        if pos is None:
            return True
        x, _ = pos
        mid_x = self.width // 2
        return (x < mid_x) if self.is_red else (x >= mid_x)

    def record_invasion_point(self, pos, agent, weight=1):
        if pos is None or not self.home_boundaries:
            return
        closest_b = min(self.home_boundaries, key=lambda b: agent.get_maze_distance(pos, b))
        self.entrance_stats[closest_b] += weight

    def get_opponent_team_style(self, game_state, opponents):
        pacman_count = sum(1 for opp in opponents if game_state.get_agent_state(opp).is_pacman)
        if pacman_count >= 2:
            return 'DUAL_OFFENSE'

        high_attack_count = 0
        total_sample = 0
        for opp in opponents:
            total = max(1, self.opp_total_turns[opp])
            total_sample += total
            if (self.opp_turns_as_pacman[opp] / total) > 0.35:
                high_attack_count += 1

        if high_attack_count >= 2:
            return 'DUAL_OFFENSE'
        elif high_attack_count == 0 and total_sample > 60 and pacman_count == 0:
            return 'DUAL_DEFENSE'
        return 'STANDARD'

    def update_beliefs(self, game_state, agent):
        my_idx = agent.index
        my_pos = game_state.get_agent_position(my_idx)
        opponents = agent.get_opponents(game_state)
        team_indices = agent.get_team(game_state)

        # 1. Defending food loss detection
        curr_food_defending = agent.get_food_you_are_defending(game_state).as_list()
        if self.last_food_defending is not None:
            lost = set(self.last_food_defending) - set(curr_food_defending)
            for food_pos in lost:
                self.food_eaten_positions.append(food_pos)
                self.record_invasion_point(food_pos, agent, weight=2)

                invaders = [
                    opp for opp in opponents
                    if game_state.get_agent_state(opp).is_pacman
                    or game_state.get_agent_position(opp) is None
                ]
                for inv in invaders:
                    if game_state.get_agent_position(inv) is None:
                        self.beliefs[inv] = util.Counter({food_pos: 1.0})
                        self.last_seen_pos[inv] = food_pos
                        self.opp_food_eaten[inv] += 1
                        break
        self.last_food_defending = curr_food_defending

        # 2. Update each opponent belief & profile statistics
        for opp in opponents:
            self.opp_total_turns[opp] += 1
            opp_state = game_state.get_agent_state(opp)
            if opp_state.is_pacman:
                self.opp_turns_as_pacman[opp] += 1

            exact_pos = game_state.get_agent_position(opp)
            if exact_pos is not None:
                self.beliefs[opp] = util.Counter({exact_pos: 1.0})
                self.last_seen_pos[opp] = exact_pos
                if self.is_home_side(exact_pos):
                    self.record_invasion_point(exact_pos, agent, weight=1)
            else:
                new_belief = util.Counter()
                for p, prob in self.beliefs[opp].items():
                    if prob > 0:
                        nbrs = self.legal_neighbors.get(p, [p]) + [p]
                        p_nbr = prob / len(nbrs)
                        for n in nbrs:
                            new_belief[n] += p_nbr

                for tm in team_indices:
                    tm_pos = game_state.get_agent_position(tm)
                    if tm_pos is not None:
                        for p in list(new_belief.keys()):
                            if util.manhattan_distance(p, tm_pos) <= 5:
                                new_belief[p] = 0.0

                noisy_distances = game_state.get_agent_distances()
                if noisy_distances is not None and opp < len(noisy_distances):
                    noisy_dist = noisy_distances[opp]
                    if my_pos is not None and noisy_dist is not None:
                        for p in list(new_belief.keys()):
                            if new_belief[p] > 0:
                                true_dist = agent.get_maze_distance(my_pos, p)
                                prob_sensor = game_state.get_distance_prob(true_dist, noisy_dist)
                                new_belief[p] *= prob_sensor

                is_pac = opp_state.is_pacman
                for p in list(new_belief.keys()):
                    if new_belief[p] > 0:
                        in_home = self.is_home_side(p)
                        if is_pac and not in_home:
                            new_belief[p] = 0.0
                        elif not is_pac and in_home:
                            new_belief[p] = 0.0

                total = sum(new_belief.values())
                if total > 0:
                    new_belief.normalize()
                    self.beliefs[opp] = new_belief
                else:
                    reset_belief = util.Counter()
                    for p in self.legal_positions:
                        in_home = self.is_home_side(p)
                        if is_pac and in_home:
                            reset_belief[p] = 1.0
                        elif not is_pac and not in_home:
                            reset_belief[p] = 1.0
                    reset_belief.normalize()
                    self.beliefs[opp] = reset_belief

                if self.beliefs[opp]:
                    best_pos = self.beliefs[opp].arg_max()
                    self.last_seen_pos[opp] = best_pos

    def get_most_likely_pos(self, opp_index):
        if opp_index in self.beliefs and self.beliefs[opp_index]:
            return self.beliefs[opp_index].arg_max()
        return self.last_seen_pos.get(opp_index, None)


########################
# Base Tactical Agent  #
########################

class TacticalCaptureAgent(CaptureAgent):
    """
    Base class providing tactical utilities, BFS pathfinding, dead-end validation,
    strict home-boundary enforcement, and score interpretation.
    """
    def __init__(self, index, time_for_computing=.1, shared_state=None):
        super().__init__(index, time_for_computing)
        self.shared_state = shared_state
        self.recent_positions = deque(maxlen=8)

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        if self.shared_state is None:
            self.shared_state = SharedTeamState(self.red)
        self.shared_state.setup(game_state, self.red, self.distancer)

    def is_home_side(self, pos):
        if pos is None:
            return True
        x, _ = pos
        mid_x = self.shared_state.width // 2
        return (x < mid_x) if self.red else (x >= mid_x)

    def is_winning(self, game_state):
        raw_score = game_state.get_score()
        return (raw_score > 0) if self.red else (raw_score < 0)

    def get_team_score(self, game_state):
        raw_score = game_state.get_score()
        return raw_score if self.red else -raw_score

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def get_dangerous_ghosts(self, game_state):
        dangerous = []
        my_pos = game_state.get_agent_position(self.index)
        for opp in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp)
            if not opp_state.is_pacman:
                scared = opp_state.scared_timer
                if scared <= 2:
                    pos = opp_state.get_position()
                    if pos is None:
                        pos = self.shared_state.get_most_likely_pos(opp)
                        if pos is not None and self.is_home_side(pos):
                            pos = None
                    if pos is not None and not self.is_home_side(pos) and my_pos is not None:
                        dist = self.get_maze_distance(my_pos, pos)
                        dangerous.append((pos, dist, scared))
        return dangerous

    def get_scared_ghosts(self, game_state):
        scared_ghosts = []
        my_pos = game_state.get_agent_position(self.index)
        for opp in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp)
            if not opp_state.is_pacman and opp_state.scared_timer > 2:
                pos = opp_state.get_position()
                if pos is not None and not self.is_home_side(pos) and my_pos is not None:
                    dist = self.get_maze_distance(my_pos, pos)
                    scared_ghosts.append((pos, dist, opp_state.scared_timer))
        return scared_ghosts

    def get_invaders(self, game_state):
        invaders = []
        my_pos = game_state.get_agent_position(self.index)
        for opp in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp)
            pos = opp_state.get_position()
            is_pac = opp_state.is_pacman
            carrying = opp_state.num_carrying

            if pos is not None:
                if is_pac or self.is_home_side(pos):
                    dist = self.get_maze_distance(my_pos, pos)
                    invaders.append((opp, pos, dist, carrying))
            else:
                b_pos = self.shared_state.get_most_likely_pos(opp)
                if is_pac and b_pos is not None and self.is_home_side(b_pos):
                    dist = self.get_maze_distance(my_pos, b_pos)
                    invaders.append((opp, b_pos, dist, carrying))
                elif not is_pac and b_pos is not None and self.is_home_side(b_pos):
                    prob_on_home = sum(
                        self.shared_state.beliefs[opp][p]
                        for p in self.shared_state.beliefs[opp]
                        if self.is_home_side(p)
                    )
                    if prob_on_home > 0.4:
                        dist = self.get_maze_distance(my_pos, b_pos)
                        invaders.append((opp, b_pos, dist, carrying))
        return invaders

    def bfs_shortest_path(self, start_pos, target_positions, forbidden_positions=None):
        if not target_positions or start_pos is None:
            return None, float('inf'), None

        targets = set(target_positions)
        if start_pos in targets:
            return Directions.STOP, 0, start_pos

        forbidden = set(forbidden_positions) if forbidden_positions else set()
        queue = deque([(start_pos, None, 0)])
        visited = {start_pos}

        while queue:
            curr, first_action, dist = queue.popleft()
            if curr in targets:
                return first_action, dist, curr

            x, y = curr
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.direction_to_vector(action)
                next_pos = (int(x + dx), int(y + dy))

                if next_pos not in visited and next_pos not in forbidden:
                    if next_pos in self.shared_state.legal_neighbors.get(curr, []):
                        visited.add(next_pos)
                        act = first_action if first_action is not None else action
                        queue.append((next_pos, act, dist + 1))

        return None, float('inf'), None

    def is_dead_end_safe(self, my_pos, target_pos, dangerous_ghosts):
        if target_pos not in self.shared_state.dead_ends:
            return True

        junction, depth = self.shared_state.dead_ends[target_pos]
        dist_to_target = self.get_maze_distance(my_pos, target_pos)
        escape_steps = dist_to_target + depth

        for g_pos, _, scared in dangerous_ghosts:
            if scared <= 2:
                ghost_dist_to_junction = self.get_maze_distance(g_pos, junction)
                if ghost_dist_to_junction <= escape_steps + 1:
                    return False
        return True

    def get_anti_oscillation_penalty(self, next_pos, action, current_direction):
        penalty = 0.0
        if action == Directions.STOP:
            penalty += 50.0

        if current_direction is not None and current_direction != Directions.STOP:
            rev = Directions.REVERSE.get(current_direction, None)
            if action == rev:
                penalty += 4.0

        count = self.recent_positions.count(next_pos)
        if count > 0:
            penalty += count * 15.0

        return penalty


#############################
# Offensive Tactical Agent  #
#############################

class OffensiveTacticalAgent(TacticalCaptureAgent):
    """
    Championship offensive agent optimized for defaultCapture layout.
    Features:
    - Seppuku countermeasure: breaks out of dead-end caging by charging ghost to respawn.
    - 3-Sector dynamic evaluation: targets open, high-density sectors ('top', 'mid', 'bot').
    - Opponent profile awareness: rushes capsule against turtling teams.
    - Flanking escapes: exits via alternate corridors when pursued.
    """
    def choose_action(self, game_state):
        self.shared_state.update_beliefs(game_state, self)

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        self.recent_positions.append(my_pos)

        legal_actions = game_state.get_legal_actions(self.index)
        legal_actions = [a for a in legal_actions if a != Directions.STOP] or [Directions.STOP]

        food_grid = self.get_food(game_state)
        food_list = food_grid.as_list()
        food_left = len(food_list)
        capsules = self.get_capsules(game_state)
        carrying = my_state.num_carrying
        time_left = game_state.data.timeleft // 4

        dangerous_ghosts = self.get_dangerous_ghosts(game_state)
        min_ghost_dist = min([d for _, d, _ in dangerous_ghosts], default=999)

        min_dist_to_home = min(
            [self.get_maze_distance(my_pos, h) for h in self.shared_state.home_boundaries],
            default=1
        )

        # -------------------------------------------------------------
        # Decision 0: Anti-Caging Seppuku Countermeasure
        # -------------------------------------------------------------
        # Only commit seppuku if strictly locked in a 2-tile oscillation cage
        # with zero carrying food!
        seppuku_needed, seppuku_action = self.check_seppuku_needed(my_pos, dangerous_ghosts, carrying)
        if seppuku_needed and seppuku_action in legal_actions:
            return seppuku_action

        # -------------------------------------------------------------
        # Decision 1: End-Game Lockdown
        # -------------------------------------------------------------
        if time_left <= 50 and self.is_winning(game_state) and carrying == 0:
            action = self.choose_safe_return_action(game_state, my_pos, dangerous_ghosts)
            if action in legal_actions:
                return action

        # -------------------------------------------------------------
        # Decision 2: Immediate Critical Return
        # -------------------------------------------------------------
        must_return_win = (food_left <= 2 and carrying > 0)
        must_return_clock = (carrying > 0 and time_left <= min_dist_to_home + 4)
        should_return_carrying = (carrying >= 4) or (carrying >= 1 and min_ghost_dist <= 3)

        if must_return_win or must_return_clock or should_return_carrying:
            action = self.choose_safe_return_action(game_state, my_pos, dangerous_ghosts)
            if action is not None:
                return action

        # -------------------------------------------------------------
        # Decision 3: Hunt Scared Ghost
        # -------------------------------------------------------------
        scared_ghosts = self.get_scared_ghosts(game_state)
        if scared_ghosts:
            closest_scared = min(scared_ghosts, key=lambda g: g[1])
            g_pos, g_dist, s_timer = closest_scared
            if g_dist <= 4 and s_timer > g_dist + 2:
                action, dist, _ = self.bfs_shortest_path(my_pos, [g_pos])
                if action in legal_actions:
                    return action

        # -------------------------------------------------------------
        # Decision 4: Capsule Rush (or Against Turtling Opponents)
        # -------------------------------------------------------------
        opponents = self.get_opponents(game_state)
        opp_style = self.shared_state.get_opponent_team_style(game_state, opponents)
        turtle_capsule_rush = (capsules and opp_style == 'DUAL_DEFENSE')

        if (capsules and min_ghost_dist <= 5) or turtle_capsule_rush:
            safe_capsules = []
            for cap in capsules:
                cap_dist = self.get_maze_distance(my_pos, cap)
                ghost_can_reach_first = any(
                    self.get_maze_distance(g_pos, cap) <= cap_dist
                    for g_pos, _, _ in dangerous_ghosts
                )
                if not ghost_can_reach_first or turtle_capsule_rush:
                    safe_capsules.append((cap_dist, cap))

            if safe_capsules:
                safe_capsules.sort()
                best_cap = safe_capsules[0][1]
                action, dist, _ = self.bfs_shortest_path(
                    my_pos, [best_cap],
                    forbidden_positions=self.get_lethal_positions(dangerous_ghosts)
                )
                if action in legal_actions:
                    return action

        # -------------------------------------------------------------
        # Decision 5: Strategic Food Collection with 3-Sector Optimization
        # -------------------------------------------------------------
        action = self.choose_offensive_move(game_state, my_pos, food_list, capsules, dangerous_ghosts, opp_style)
        if action in legal_actions:
            return action

        return self.choose_fallback_action(game_state, legal_actions, dangerous_ghosts)

    def check_seppuku_needed(self, my_pos, dangerous_ghosts, carrying):
        """
        Detects if our Pacman is genuinely trapped in a cage (dead end) with an enemy ghost
        camping the exit. Only commits seppuku if carrying == 0 and strictly oscillating
        between 2 tiles for at least 8 moves, guaranteeing zero accidental suicides.
        """
        if my_pos not in self.shared_state.dead_ends or carrying > 0:
            return False, None

        if len(self.recent_positions) < 8:
            return False, None

        # Verify strict 2-tile oscillation: A, B, A, B, A, B, A, B
        p = list(self.recent_positions)
        is_oscillating = (
            p[-1] == p[-3] == p[-5] == p[-7] and
            p[-2] == p[-4] == p[-6] == p[-8] and
            p[-1] != p[-2]
        )
        if not is_oscillating:
            return False, None

        junction, depth = self.shared_state.dead_ends[my_pos]
        ghost_blocking = any(
            self.get_maze_distance(g_pos, junction) <= 1
            for g_pos, _, scared in dangerous_ghosts
            if scared <= 1
        )
        if ghost_blocking:
            action, _, _ = self.bfs_shortest_path(my_pos, [junction])
            return True, action

        return False, None

    def get_lethal_positions(self, dangerous_ghosts):
        lethal = set()
        for g_pos, dist, scared in dangerous_ghosts:
            if scared <= 1 and dist <= 3:
                lethal.add(g_pos)
                for nbr in self.shared_state.legal_neighbors.get(g_pos, []):
                    lethal.add(nbr)
        for h in self.shared_state.home_boundaries:
            lethal.discard(h)
        return lethal

    def choose_safe_return_action(self, game_state, my_pos, dangerous_ghosts):
        lethal_positions = self.get_lethal_positions(dangerous_ghosts)

        scored_boundaries = []
        for h in self.shared_state.home_boundaries:
            dist_to_h = self.get_maze_distance(my_pos, h)
            min_g_to_h = min(
                [self.get_maze_distance(g_pos, h) for g_pos, _, _ in dangerous_ghosts],
                default=999
            )
            safety_margin = min_g_to_h - dist_to_h
            score = dist_to_h - max(0, safety_margin) * 2.0
            scored_boundaries.append((score, h))

        scored_boundaries.sort()
        best_goals = [h for _, h in scored_boundaries[:4]]

        action, dist, _ = self.bfs_shortest_path(
            my_pos, best_goals,
            forbidden_positions=lethal_positions
        )
        if action is not None:
            return action

        action, dist, _ = self.bfs_shortest_path(my_pos, best_goals)
        return action

    def choose_offensive_move(self, game_state, my_pos, food_list, capsules, dangerous_ghosts, opp_style):
        lethal_positions = self.get_lethal_positions(dangerous_ghosts)

        if opp_style == 'DUAL_OFFENSE' and not dangerous_ghosts:
            lethal_positions = set()

        safe_food = []
        for food in food_list:
            if opp_style == 'DUAL_OFFENSE' or self.is_dead_end_safe(my_pos, food, dangerous_ghosts):
                safe_food.append(food)

        if not safe_food:
            if capsules:
                act, _, _ = self.bfs_shortest_path(my_pos, capsules, lethal_positions)
                if act is not None:
                    return act
            return self.choose_safe_return_action(game_state, my_pos, dangerous_ghosts)

        # -------------------------------------------------------------
        # 3-Zone Analysis: 'top', 'mid', 'bot'
        # -------------------------------------------------------------
        zones = {'top': [], 'mid': [], 'bot': []}
        for f in safe_food:
            z = self.shared_state.get_zone(f[1])
            zones[z].append(f)

        ghost_in_zone = {'top': False, 'mid': False, 'bot': False}
        for g_pos, dist, scared in dangerous_ghosts:
            if scared <= 2:
                gz = self.shared_state.get_zone(g_pos[1])
                ghost_in_zone[gz] = True
                for z_name, corr_cells in self.shared_state.corridors.items():
                    if any(self.get_maze_distance(g_pos, c) <= 3 for c in corr_cells):
                        ghost_in_zone[z_name] = True

        capsule_in_zone = {'top': False, 'mid': False, 'bot': False}
        for cap in capsules:
            cz = self.shared_state.get_zone(cap[1])
            capsule_in_zone[cz] = True

        zone_scores = {}
        for z_name in ['top', 'mid', 'bot']:
            food_count = len(zones[z_name])
            score = food_count * 3.0
            if capsule_in_zone[z_name]:
                score += 8.0
            if ghost_in_zone[z_name]:
                score -= 16.0
            zone_scores[z_name] = score

        best_zone = max(zone_scores, key=zone_scores.get)

        scored_food = []
        for food in safe_food:
            dist = self.get_maze_distance(my_pos, food)
            cluster_count = sum(
                1 for other in safe_food
                if other != food and self.get_maze_distance(food, other) <= 3
            )
            score = dist - (cluster_count * 0.8)

            fz = self.shared_state.get_zone(food[1])
            if fz == best_zone:
                score -= 6.0
            if ghost_in_zone[fz]:
                score += 12.0

            scored_food.append((score, food))

        scored_food.sort()
        candidate_targets = [f for _, f in scored_food[:5]]

        action, dist, target_reached = self.bfs_shortest_path(
            my_pos, candidate_targets,
            forbidden_positions=lethal_positions
        )
        if action is not None:
            return action

        action, dist, _ = self.bfs_shortest_path(my_pos, safe_food)
        return action

    def choose_fallback_action(self, game_state, legal_actions, dangerous_ghosts):
        best_action = None
        best_val = -float('inf')
        curr_dir = game_state.get_agent_state(self.index).configuration.direction

        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_state(self.index).get_position()

            if dangerous_ghosts:
                min_g = min(self.get_maze_distance(next_pos, g_pos) for g_pos, _, _ in dangerous_ghosts)
            else:
                min_g = 10

            dist_home = min(self.get_maze_distance(next_pos, h) for h in self.shared_state.home_boundaries)

            val = (min_g * 10.0) - dist_home
            val -= self.get_anti_oscillation_penalty(next_pos, action, curr_dir)

            if val > best_val:
                best_val = val
                best_action = action

        return best_action or random.choice(legal_actions)


#############################
# Defensive Tactical Agent  #
#############################

class DefensiveTacticalAgent(TacticalCaptureAgent):
    """
    Championship defensive agent optimized for defaultCapture layout.
    Features:
    - Relentless Interception & Elimination: Destroys invaders before they can score.
    - True Deadlock Caging: If an invader has 0 food, depth <= 2, and no food left in pocket,
      holds the doorway to neutralize them without allowing respawn.
    - Home Power Capsule Protection: Defends (6, 5) or (25, 10).
    - Pre-Emptive Bottleneck Ambush: Rushes to corridor bottleneck when opponent nears border.
    - Strict Home Boundary Lock: Never crosses into enemy territory.
    """
    def __init__(self, index, time_for_computing=.1, shared_state=None):
        super().__init__(index, time_for_computing, shared_state)
        self.patrol_index = 0

    def get_home_actions(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        home_actions = []
        for a in actions:
            if a == Directions.STOP:
                continue
            succ = self.get_successor(game_state, a)
            succ_pos = succ.get_agent_state(self.index).get_position()
            if self.is_home_side(succ_pos):
                home_actions.append(a)

        if home_actions:
            return home_actions
        return [Directions.STOP]

    def bfs_shortest_path_home(self, start_pos, target_positions, extra_forbidden=None):
        forbidden = set(extra_forbidden) if extra_forbidden else set()
        for p in self.shared_state.legal_positions:
            if not self.is_home_side(p):
                forbidden.add(p)
        return self.bfs_shortest_path(start_pos, target_positions, forbidden_positions=forbidden)

    def choose_action(self, game_state):
        self.shared_state.update_beliefs(game_state, self)

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        self.recent_positions.append(my_pos)

        legal_home_actions = self.get_home_actions(game_state)
        invaders = self.get_invaders(game_state)
        scared_timer = my_state.scared_timer

        # -------------------------------------------------------------
        # Case 1: Scared Defense (Enemy ate power capsule)
        # -------------------------------------------------------------
        if scared_timer > 1 and invaders:
            action = self.choose_scared_defense_action(game_state, my_pos, invaders, legal_home_actions)
            if action in legal_home_actions:
                return action

        # -------------------------------------------------------------
        # Case 2: Protect Our Home Power Capsule
        # -------------------------------------------------------------
        home_capsules = self.get_capsules_you_are_defending(game_state)
        if home_capsules and invaders:
            target_cap = home_capsules[0]
            threat_invader = [inv for inv in invaders if self.get_maze_distance(inv[1], target_cap) <= 5]
            if threat_invader:
                action, _, _ = self.bfs_shortest_path_home(my_pos, [target_cap])
                if action in legal_home_actions:
                    return action

        # -------------------------------------------------------------
        # Case 3: Interception & Tactical Caging
        # -------------------------------------------------------------
        if invaders:
            invaders.sort(key=lambda inv: (-inv[3], inv[2]))
            target_opp, inv_pos, inv_dist, carrying = invaders[0]

            # Dead-end Caging Logic:
            # ONLY cage if:
            # 1. Invader has 0 food
            # 2. There is ZERO defending food inside that dead-end branch
            # 3. Branch depth <= 2
            # 4. Only 1 invader on our side
            if inv_pos in self.shared_state.dead_ends and self.is_home_side(inv_pos):
                junction, depth = self.shared_state.dead_ends[inv_pos]
                food_defending = self.get_food_you_are_defending(game_state).as_list()

                # Check if there is any defended food inside this dead-end branch
                food_in_branch = any(
                    f in self.shared_state.dead_ends and self.shared_state.dead_ends[f][0] == junction
                    for f in food_defending
                )

                if self.is_home_side(junction) and carrying == 0 and not food_in_branch and depth <= 2:
                    if my_pos == junction:
                        if len(invaders) > 1:
                            action, _, _ = self.bfs_shortest_path_home(my_pos, [inv_pos])
                            if action in legal_home_actions:
                                return action
                        else:
                            # Hold the doorway to contain the harmless 0-food invader!
                            if Directions.STOP in game_state.get_legal_actions(self.index):
                                return Directions.STOP
                    else:
                        action, _, _ = self.bfs_shortest_path_home(my_pos, [junction])
                        if action in legal_home_actions:
                            return action

            # Otherwise: Relentlessly intercept and eliminate the invader!
            action = self.intercept_invader(game_state, my_pos, inv_pos, carrying, legal_home_actions)
            if action in legal_home_actions:
                return action

        # -------------------------------------------------------------
        # Case 4: Food Eaten Response (Investigate recently eaten food)
        # -------------------------------------------------------------
        if self.shared_state.food_eaten_positions:
            investigate_pos = self.shared_state.food_eaten_positions[-1]
            if self.is_home_side(investigate_pos) and self.get_maze_distance(my_pos, investigate_pos) > 1:
                action, _, _ = self.bfs_shortest_path_home(my_pos, [investigate_pos])
                if action in legal_home_actions:
                    return action

        # -------------------------------------------------------------
        # Case 5: Pre-Emptive Corridor Ambush & Patrol
        # -------------------------------------------------------------
        action = self.patrol_chokepoints(game_state, my_pos, legal_home_actions)
        if action in legal_home_actions:
            return action

        return random.choice(legal_home_actions)

    def intercept_invader(self, game_state, my_pos, inv_pos, carrying, legal_home_actions):
        if not self.is_home_side(inv_pos):
            inv_pos = min(self.shared_state.home_boundaries, key=lambda b: self.get_maze_distance(inv_pos, b))

        if carrying > 0:
            best_escape = min(
                self.shared_state.home_boundaries,
                key=lambda b: self.get_maze_distance(inv_pos, b)
            )
            inv_dist_to_escape = self.get_maze_distance(inv_pos, best_escape)
            my_dist_to_escape = self.get_maze_distance(my_pos, best_escape)

            if my_dist_to_escape <= inv_dist_to_escape:
                action, _, _ = self.bfs_shortest_path_home(my_pos, [best_escape])
                if action in legal_home_actions:
                    return action

        action, _, _ = self.bfs_shortest_path_home(my_pos, [inv_pos])
        return action

    def choose_scared_defense_action(self, game_state, my_pos, invaders, legal_home_actions):
        best_action = None
        best_val = -float('inf')
        curr_dir = game_state.get_agent_state(self.index).configuration.direction

        for action in legal_home_actions:
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_state(self.index).get_position()

            if not self.is_home_side(next_pos):
                continue

            min_inv_dist = min(self.get_maze_distance(next_pos, inv[1]) for inv in invaders)

            if min_inv_dist <= 1:
                val = -1000.0
            else:
                val = -abs(min_inv_dist - 2.5) * 10.0

            val -= self.get_anti_oscillation_penalty(next_pos, action, curr_dir)

            if val > best_val:
                best_val = val
                best_action = action

        return best_action or random.choice(legal_home_actions)

    def patrol_chokepoints(self, game_state, my_pos, legal_home_actions):
        food_defending = self.get_food_you_are_defending(game_state).as_list()
        boundaries = self.shared_state.home_boundaries

        if not boundaries:
            return random.choice(legal_home_actions)

        # Detect which corridor the enemy is currently approaching on their side
        approaching_corridor = None
        min_opp_dist = 999
        opponents = self.get_opponents(game_state)
        for opp in opponents:
            opp_pos = self.shared_state.get_most_likely_pos(opp)
            if opp_pos is not None and not self.is_home_side(opp_pos):
                for z, ambush_pt in self.shared_state.corridor_ambush_points.items():
                    if ambush_pt:
                        d = self.get_maze_distance(opp_pos, ambush_pt)
                        if d < min_opp_dist:
                            min_opp_dist = d
                            approaching_corridor = z

        if approaching_corridor and min_opp_dist <= 6:
            ambush_spot = self.shared_state.corridor_ambush_points[approaching_corridor]
            if ambush_spot:
                action, _, _ = self.bfs_shortest_path_home(my_pos, [ambush_spot])
                if action in legal_home_actions:
                    return action

        def score_boundary(b):
            dists = [self.get_maze_distance(b, f) for f in food_defending] if food_defending else [0]
            food_score = min(dists) * 1.5 + (sum(dists) / len(dists)) if food_defending else 0.0

            tendency = self.shared_state.entrance_stats.get(b, 0)
            for other_b in boundaries:
                if abs(other_b[1] - b[1]) <= 2:
                    tendency += self.shared_state.entrance_stats.get(other_b, 0) * 0.5

            return food_score - (tendency * 6.0)

        sorted_boundaries = sorted(boundaries, key=score_boundary)
        top_patrol_points = sorted_boundaries[:2]
        target = top_patrol_points[self.patrol_index % len(top_patrol_points)]

        if self.get_maze_distance(my_pos, target) <= 1:
            self.patrol_index += 1
            target = top_patrol_points[self.patrol_index % len(top_patrol_points)]

        action, _, _ = self.bfs_shortest_path_home(my_pos, [target])
        if action in legal_home_actions:
            return action

        return random.choice(legal_home_actions)


# Link aliases for create_team backward compatibility
OffensiveReflexAgent = OffensiveTacticalAgent
DefensiveReflexAgent = DefensiveTacticalAgent
