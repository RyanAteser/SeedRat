   import json
import torch
from agent.jepa import JEPA
import random
import numpy as np
from collections import deque

# Define MOVEMENT_MAP if it's not defined globally elsewhere
# This is crucial for detect_scent_gradient and simulate_move
MOVEMENT_MAP = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}


class SeedRat:  # Assuming your class is named SeedRat
    def __init__(self, goal):
        self.pos = (0, 0)
        self.hunger = 0
        self.fear = 0
        self.curiosity = 1.0
        self.energy = 100.0
        self.age = 0
        self.goal = goal
        self.food_acquired = False
        self.failed_actions = set()
        self.plan_confidence = 0.5
        self.je_pa_errors = []
        self.doors_entered = 0  # Track how many doors the rat has entered
        self.last_known_burrow = None
        self.burrow_pos = None  # ADDED: To resolve 'burrow_pos' reference
        self.burrow_nearby = False  # ADDED: To resolve 'burrow_nearby' reference

        # Enhanced investigation state
        self.walls_examined = set()  # Track which walls we've investigated
        self.investigation_count = 0
        self.last_action = None
        self.stuck_counter = 0  # Track how long we've been in same position

        self.memory = []
        self.working_memory = []
        self.episodic_memory = deque(maxlen=200)
        self.mom_memory = []

        # Initialize JEPA first to get proper dimensions
        # Use a more realistic initial input_dim if possible,
        # or ensure it's re-initialized after encode_state_enhanced is stable.
        # For now, keeping your placeholder logic.
        example_state_dim_placeholder = 10 # This will be overwritten below
        self.jepa = JEPA(input_dim=example_state_dim_placeholder)

        try:
            self.jepa.load_state_dict(torch.load("models/jepa.pth"))
            self.jepa.eval()
            self.log_memory("Loaded pre-trained JEPA model.")
        except Exception:
            self.log_memory("No pre-trained JEPA model found.")

        try:
            with open("mom_memory/mom_learned.json", "r") as f:
                learned_rules = json.load(f)
            self.mom_memory += learned_rules
        except FileNotFoundError:
            print("⚠️ No mom_learned.json found. Using default memory.")
            self.mom_memory = [
                "Avoid noise.",
                "Food smells good.",
                "Dark burrows are safe.",
                "Watch where others go.",
                "Sometimes walls have hidden passages.",
                "Dig when trapped.",
                "Follow stronger scents."
            ]

        # Now properly initialize JEPA with correct dimensions after mom_memory load attempt
        # Ensure that encode_state_enhanced calculates the correct vector size
        # This means the dimensions provided to JEPA are correct for your state representation.
        example_state = self.encode_state_enhanced({"up":" ", "down":" ", "left":" ", "right":" "}, 0, 0, False)
        self.jepa = JEPA(input_dim=example_state.shape[0])


        self.concept_map = {}
        self.active_strategy = "systematic_exploration"
        # Subgoals are managed by the arbitration layer, but default if not set
        self.subgoals = ["locate_food_source", "secure_food"]
        self.current_subgoal = 0
        self.high_level_goal = "secure_food" # Initialize high-level goal

    def update_emotions(self, tile):
        if tile == "F":
            self.hunger = 0
            self.energy += 20
        elif tile == "N":
            self.fear += 2
            self.energy -= 10
        elif tile == "B":
            self.fear = max(0, self.fear - 1)
            self.energy -= 1
        else:
            self.hunger += 1
            self.curiosity += 0.5
            self.energy -= 2

    def update_emotions_and_state(self, tile, context):
        self.update_emotions(tile)
        self.age += 1
        self.curiosity = min(max(self.curiosity, 0.1), 2.0)
        self.memory.append({"tile": tile, "context": context})
        if len(self.memory) > 50:
            self.memory.pop(0)
        if isinstance(context, torch.Tensor):
            self.episodic_memory.append(context)
        if tile == "B":
            self.last_known_burrow = self.pos  # ✅ Remember burrow location
            self.burrow_pos = self.pos  # Keep burrow_pos updated if it's referenced

    def encode_state_enhanced(self, surroundings, food_scent=0, danger_scent=0, burrow_nearby=False):
        dir_map = {"Wall": 0, "F": 1, "N": 2, "B": 3, " ": 4, "WeakWall": 0.5, "Passage": 4.5, "Crack": 0.6} # Added Crack
        vec = [dir_map.get(surroundings.get(d, " "), 4) for d in ["up", "down", "left", "right"]]
        vec += [
            self.hunger, self.fear, self.curiosity, self.energy / 100.0,
            food_scent, danger_scent, int(burrow_nearby), self.plan_confidence,
            len(self.walls_examined), self.stuck_counter / 10.0  # Add investigation state
        ]
        # Update self.burrow_nearby attribute if it exists, for consistency
        self.burrow_nearby = burrow_nearby  # ADDED: To keep attribute consistent with function param
        return torch.tensor(vec, dtype=torch.float32)

    def plan_path_to_burrow(self, env):
        if not self.last_known_burrow:
            self.log_memory("❌ I don’t remember where the burrow is.")
            return None

        from collections import deque

        def simulate_move_check_wall(pos, direction, current_env):
            x, y = pos
            dx, dy = MOVEMENT_MAP[direction]
            new_x, new_y = x + dx, y + dy
            if 0 <= new_y < current_env.height and 0 <= new_x < current_env.width:
                if current_env.grid[new_y][new_x] not in ["Wall", "#"]:
                    return (new_x, new_y)
            return pos  # No move if blocked or out of bounds

        start = self.pos
        goal = self.last_known_burrow
        queue = deque([(start, [])])
        visited = set()
        visited.add(start) # Add start to visited initially

        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path  # list of ['up', 'right', ...]

            for direction in ["up", "down", "left", "right"]:
                new_pos = simulate_move_check_wall(current, direction, env)
                if new_pos != current and new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, path + [direction]))

        self.log_memory("⚠️ No path to burrow found.")
        return None

    def detect_scent_gradient(self, env, scent_type="F"):
        """Enhanced scent detection with directional components"""
        # Ensure env has get_scent_map and it can handle 'burrow' type
        # Or that scent_map is directly accessible like env.scent_map[scent_type]
        current_scent = env.get_scent_map(scent_type, self.pos)
        gradients = {}

        directions = ["up", "down", "left", "right"]
        for direction in directions:
            # Using the rat's own simulate_move for consistency
            test_pos = self.simulate_move(self.pos, direction, env)
            if test_pos != self.pos:  # Valid move (not blocked by grid edge, but still check walls below)
                # Check if the tile at test_pos is a wall in the actual environment grid
                # Assuming env.grid is (row, col) i.e. (y,x)
                if env.grid[test_pos[1]][test_pos[0]] not in ["Wall", "#"]:
                    neighbor_scent = env.get_scent_map(scent_type, test_pos)
                    gradients[direction] = neighbor_scent - current_scent
                else:
                    # If it's a wall, no scent gradient can be followed this way
                    gradients[direction] = -float('inf') # Strong penalty for walls
            else:
                # If simulate_move returns same position (e.g., out of bounds or current wall)
                gradients[direction] = -float('inf') # Strong penalty

        return gradients

    def investigate_wall(self, direction, env):
        """Investigate a wall more thoroughly"""
        wall_key = f"{self.pos}_{direction}"

        if wall_key in self.walls_examined:
            return "already_examined"

        self.walls_examined.add(wall_key)
        self.investigation_count += 1
        self.energy -= 1  # Small energy cost for investigation

        # Simulate finding hidden passages or weak walls
        investigation_roll = random.random()

        # Higher chance of finding something if we're really stuck
        desperation_bonus = min(self.stuck_counter * 0.05, 0.3)

        if investigation_roll < (0.15 + desperation_bonus):  # 15% base chance + desperation
            discovery_type = random.choice(["weak_wall", "small_passage", "crack"])
            self.log_memory(f"Found {discovery_type} in {direction} wall after investigation!")
            return discovery_type
        else:
            self.log_memory(f"Thoroughly examined {direction} wall - solid barrier.")
            return "solid_wall"

    def attempt_special_action(self, action_type, direction, env):
        """Attempt digging, climbing, or squeezing through passages"""
        energy_cost = {"dig": 5, "climb": 3, "squeeze": 2}
        success_rate = {"dig": 0.3, "climb": 0.4, "squeeze": 0.7} # These rates are quite low

        if self.energy < energy_cost.get(action_type, 5):
            return False, "insufficient_energy"

        self.energy -= energy_cost[action_type]

        # In a real environment, this would modify the grid
        # For simulation, we just report success/failure
        if random.random() < success_rate[action_type]:
            self.log_memory(f"Successfully {action_type}ed through {direction}!")
            return True, "success"
        else:
            self.log_memory(f"Failed to {action_type} through {direction}.")
            return False, "failed"

    def choose_enhanced_action(self, surroundings, env, scents):
        """Enhanced action selection with investigation capabilities"""
        all_walls = all(tile == "Wall" for tile in surroundings.values())

        # *** No stuck_counter management here, it's in the run loop ***

        # NEW PRIORITY 1: Return to Burrow if high_level_goal is return_to_burrow
        # and we know the burrow location.
        if self.high_level_goal == "return_to_burrow":
            if self.pos == self.burrow_pos:
                self.log_memory("🕳️ At burrow, resting.")
                # If at the burrow and goal is to rest, simply stay.
                # The LLM's prompt for 'rest_in_burrow' should also guide this.
                return "stay" # 'stay' is a custom action for your environment to handle

            if self.last_known_burrow:
                burrow_path = self.plan_path_to_burrow(env)
                if burrow_path:
                    next_direction = burrow_path[0]
                    self.log_memory(f"🗺️ Following A* path to burrow: {next_direction}")
                    return next_direction
                else:
                    self.log_memory("🚫 No A* path to burrow, attempting burrow scent gradient.")
                    burrow_gradients = self.detect_scent_gradient(env, "burrow")
                    valid_directions = [d for d, tile in surroundings.items() if tile != "Wall"]

                    # Find best direction based on burrow scent gradient
                    # Ensure only valid, non-wall directions are considered
                    navigable_burrow_directions = [
                        d for d in valid_directions
                        if burrow_gradients.get(d, -float('inf')) > -float('inf') # Exclude impossible moves
                    ]

                    if navigable_burrow_directions:
                        # Prioritize directions with positive gradient, then any non-negative
                        positive_gradients_dirs = [d for d in navigable_burrow_directions if burrow_gradients[d] > 0]
                        if positive_gradients_dirs:
                            best_direction = max(positive_gradients_dirs,
                                                 key=lambda d: burrow_gradients[d])
                            self.log_memory(f" burrow scent: {best_direction}")
                            return best_direction
                        elif navigable_burrow_directions: # No positive, but still navigable
                            best_direction = max(navigable_burrow_directions,
                                                 key=lambda d: burrow_gradients[d])
                            self.log_memory(f" burrow scent (no positive, but navigable): {best_direction}")
                            return best_direction

            self.log_memory("🤷 Can't find burrow or path, falling back to general strategy.")
            # If no path found or stuck trying, fall through to other strategies
            # This might lead to exploration to find the burrow again.

        # If completely surrounded and haven't investigated much, prioritize investigation
        # This logic still uses 'all_walls' but doesn't modify stuck_counter
        if all_walls and len(self.walls_examined) < 4:
            unexamined_directions = []
            for direction in ["up", "down", "left", "right"]:
                wall_key = f"{self.pos}_{direction}"
                if wall_key not in self.walls_examined:
                    unexamined_directions.append(direction)

            if unexamined_directions:
                # Choose direction with strongest scent gradient for investigation
                # (Still using food scent here as a guide for WHERE to investigate)
                gradients = self.detect_scent_gradient(env, "F")
                best_direction = max(unexamined_directions,
                                     key=lambda d: gradients.get(d, -999))
                return f"investigate_{best_direction}"

        # If we've investigated walls and found weaknesses, try special actions
        # This also uses 'all_walls' and 'self.stuck_counter' but doesn't modify stuck_counter
        if all_walls and self.stuck_counter > 5:
            # Try digging in direction of strongest food scent (if any) or randomly
            gradients = self.detect_scent_gradient(env, "F")
            # Filter for directions that are not -inf (i.e., not solid walls in next step based on scent calculation)
            potential_dig_climb_dirs = [d for d, g in gradients.items() if g != -float('inf')]

            if potential_dig_climb_dirs:
                best_scent_direction = max(potential_dig_climb_dirs, key=lambda d: gradients[d])
            else: # If all directions are solid walls based on scent, pick randomly
                best_scent_direction = random.choice(["up", "down", "left", "right"])


            if self.energy > 10 and random.random() < 0.4:
                self.log_memory(f"Stuck override: Trying to dig {best_scent_direction}")
                return f"dig_{best_scent_direction}"
            elif self.energy > 6 and random.random() < 0.3:
                self.log_memory(f"Stuck override: Trying to climb {best_scent_direction}")
                return f"climb_{best_scent_direction}"
            else:
                self.log_memory("Stuck but low energy for special actions, falling back to LLM.")


        # Regular movement with scent gradient following (for food or general exploration)
        if not all_walls: # Only move if not completely surrounded
            # If high-level goal is secure_food, follow food scent. Otherwise, might explore or return to burrow (already handled)
            if self.high_level_goal == "secure_food":
                gradients = self.detect_scent_gradient(env, "F")
            else: # For other goals, e.g., explore, or if no food scent, use general movement
                # You might want another scent type here for general exploration or just follow LLM
                gradients = self.detect_scent_gradient(env, "F") # Fallback to food scent if no other gradient
                # Could implement 'explore_gradient' based on 'novelty' here

            valid_directions = [d for d, tile in surroundings.items() if tile not in ["Wall", "#"]] # Also exclude '#' from valid moves

            # Only follow gradients if they are positive and there are valid directions
            if valid_directions:
                positive_gradients_dirs = [d for d in valid_directions if gradients.get(d, 0) > 0]
                if positive_gradients_dirs:
                    # Follow positive scent gradient
                    best_direction = max(positive_gradients_dirs, key=lambda d: gradients[d])
                    self.log_memory(f"Following scent gradient: {best_direction}")
                    return best_direction
                else:
                    # If no positive gradient but still valid directions, choose based on existing scent or just move
                    # Avoid moving into known less favorable areas.
                    # This is where LLM prediction should ideally guide.
                    pass # Fall through to LLM prediction if no clear positive gradient
            else:
                pass # Fall through if no valid directions, might trigger investigation/stuck behavior

        # Fallback to original prediction system if no specific enhanced action applies
        # This will query the LLM and the JEPA model
        self.log_memory("Falling back to LLM prediction.")
        return self.predict_outcomes(surroundings, env, self.goal)


    def recall_similar_context(self, current_context):
        scores = []
        for memory in self.episodic_memory:
            if not isinstance(memory, torch.Tensor):
                continue
            if memory.shape != current_context.shape:
                # print(f"Warning: Memory shape mismatch. Current: {current_context.shape}, Memory: {memory.shape}")
                continue
            diff = torch.abs(current_context - memory).sum().item()
            scores.append((diff, memory))
        scores.sort()
        return scores[0][1] if scores else None

    def generate_hypotheses(self, surroundings, scents):
        hypotheses = []
        if scents.get("food", 0) > 0.3:
            hypotheses.append({"type": "food_location", "prediction": "Food likely nearby", "confidence": scents["food"]})
        if scents.get("danger", 0) > 0.3:
            hypotheses.append({"type": "danger_alert", "prediction": "Possible threat detected", "confidence": scents["danger"]})
        if sum(1 for v in surroundings.values() if v == "Wall") >= 3:
            hypotheses.append({"type": "spatial_constraint", "prediction": "In constrained space", "confidence": 0.8})

        # New hypotheses for investigation
        if len(self.walls_examined) > 0:
            hypotheses.append({"type": "investigation_progress", "prediction": f"Examined {len(self.walls_examined)} walls", "confidence": 0.9})
        if self.stuck_counter > 3:
            hypotheses.append({"type": "trapped_state", "prediction": "Need alternative escape method", "confidence": min(self.stuck_counter / 10.0, 0.9)})

        # Add hypothesis for burrow proximity
        if self.burrow_nearby:
            hypotheses.append({"type": "burrow_proximity", "prediction": "Burrow is very close, prioritize return.", "confidence": 0.95})
        elif self.last_known_burrow:
            hypotheses.append({"type": "burrow_known", "prediction": "Burrow location is known, consider pathing.", "confidence": 0.7})


        return hypotheses

    def plan_with_world_model(self, surroundings, env, goal):
        # Pass the correct scent map to choose_enhanced_action
        # Ensure 'scents' dictionary has both 'food' and 'burrow' if needed
        scent_data = {
            "food": env.get_scent_map("F", self.pos),
            "burrow": env.get_scent_map("burrow", self.pos) if self.burrow_pos else 0 # Pass burrow scent if burrow is known
        }
        best_action = self.choose_enhanced_action(surroundings, env, scent_data)

        # Adjust confidence based on situation
        if all(tile == "Wall" for tile in surroundings.values()):
            if len(self.walls_examined) < 4:
                self.plan_confidence = 0.7  # Higher confidence in investigation strategy
            else:
                self.plan_confidence = 0.3  # Lower confidence when truly stuck
        else:
            self.plan_confidence = 0.8  # High confidence when paths available

        return best_action

    def predict_outcomes(self, surroundings, env, goal):
        possible_actions = ["up", "down", "left", "right"]
        best_score = float('-inf')
        best_action = None

        # Here's where burrow_nearby is correctly calculated and passed as an argument
        burrow_nearby_current = any(tile == "B" for tile in surroundings.values())
        context_vec = self.encode_state_enhanced(surroundings,
                                                 env.get_scent_map("F", self.pos),
                                                 env.get_scent_map("N", self.pos),
                                                 burrow_nearby_current) # Ensure all scents are passed

        similar_memory = self.recall_similar_context(context_vec)
        if similar_memory is not None:
            self.log_memory("I remember being here. Past memory helps guide me.")

        for action in possible_actions:
            new_pos = self.simulate_move(self.pos, action, env)
            # Ensure the simulated move actually results in a different position
            # and that the new position isn't a wall
            if new_pos == self.pos or env.grid[new_pos[1]][new_pos[0]] in ["Wall", "#"]:
                continue # Skip invalid or blocked moves

            x, y = new_pos
            tile = env.grid[y][x] # Get the tile type at the new position
            temp_surroundings = { # Recreate surroundings for the predicted new position
                "up": env.grid[y - 1][x] if y > 0 else "Wall",
                "down": env.grid[y + 1][x] if y < env.height - 1 else "Wall",
                "left": env.grid[y][x - 1] if x > 0 else "Wall",
                "right": env.grid[y][x + 1] if x < env.width - 1 else "Wall",
            }
            food_scent_next = env.get_scent_map("F", new_pos)
            danger_scent_next = env.get_scent_map("N", new_pos)
            burrow_nearby_next = any(tile_val == "B" for tile_val in temp_surroundings.values())
            target_vec = self.encode_state_enhanced(temp_surroundings, food_scent_next, danger_scent_next, burrow_nearby_next)

            with torch.no_grad():
                loss = self.jepa.compute_loss(context_vec.unsqueeze(0), target_vec.unsqueeze(0))
                score = -loss.item()

            if score > best_score:
                best_score = score
                best_action = action

        # If best_action is still None after checking all valid moves, try fallback
        if best_action is None:
            return self._fallback_action(surroundings)
        return best_action

    def simulate_move(self, pos, action, env):
        x, y = pos
        dx, dy = MOVEMENT_MAP[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_y < env.height and 0 <= new_x < env.width:
            return (new_x, new_y)
        return pos # Return current position if out of bounds

    def _fallback_action(self, surroundings):
        """
        Instinct-driven fallback logic when no confident action is available.
        Prioritizes escape, then exploration, then memory, then desperation.
        """
        valid_dirs = [d for d, t in surroundings.items() if t not in ["Wall", "#"]]

        if not valid_dirs:
            # Fully trapped, check unexamined walls
            unexamined = [d for d in ["up", "down", "left", "right"]
                          if f"{self.pos}_{d}" not in self.walls_examined]
            if unexamined:
                self.log_memory("Fallback: Investigating unexamined wall.")
                return f"investigate_{random.choice(unexamined)}"

            self.log_memory("Fallback: Trapped. Attempting to dig instinctively.")
            return f"dig_{random.choice(['up', 'down', 'left', 'right'])}"

        # Instinct priority — simulated mammalian preferences
        # If the goal is to return to burrow, prioritize moving towards known burrow direction
        if self.high_level_goal == "return_to_burrow" and self.last_known_burrow:
            # Simple heuristic: calculate a general direction to burrow
            burrow_dx = self.last_known_burrow[0] - self.pos[0]
            burrow_dy = self.last_known_burrow[1] - self.pos[1]

            if abs(burrow_dx) > abs(burrow_dy): # More horizontal distance
                if burrow_dx > 0 and "right" in valid_dirs:
                    return "right"
                elif burrow_dx < 0 and "left" in valid_dirs:
                    return "left"
            else: # More vertical distance or equal
                if burrow_dy > 0 and "down" in valid_dirs:
                    return "down"
                elif burrow_dy < 0 and "up" in valid_dirs:
                    return "up"


        instincts = ["left", "right", "down", "up"]  # e.g., prefers wall-following
        for instinct in instincts:
            if instinct in valid_dirs:
                self.log_memory(f"Fallback: Using instinct to move {instinct}.")
                return instinct

        # As final fallback: random valid direction
        fallback_choice = random.choice(valid_dirs)
        self.log_memory(f"Fallback: Random direction {fallback_choice} selected.")
        return fallback_choice

    def learn_from_outcome(self, expected_state, actual_state):
        self.jepa.train()
        optimizer = torch.optim.Adam(self.jepa.parameters(), lr=1e-3)

        optimizer.zero_grad()
        loss = self.jepa.compute_loss(expected_state.unsqueeze(0), actual_state.unsqueeze(0))
        loss.backward()
        optimizer.step()

        self.log_memory(f"Learned from outcome. Loss: {loss.item():.4f}")

    def get_reflection(self):
        return {
            "current_strategy": self.active_strategy,
            "plan_confidence": self.plan_confidence,
            "key_concepts": dict(list(self.concept_map.items())[:5]),
            "recent_performance": "insufficient_data", # This should be dynamically updated
            "exploration_exploitation_balance": 0.5, # This should be dynamically updated
            "active_subgoal": self.subgoals[self.current_subgoal] if self.current_subgoal < len(self.subgoals) else "goal_complete",
            "walls_examined": len(self.walls_examined),
            "stuck_counter": self.stuck_counter,
            "investigation_count": self.investigation_count,
            "burrow_pos": self.burrow_pos # Add burrow pos to reflection
        }

    def log_memory(self, monologue):
        # Check if self.memory is None or not a list, initialize if needed
        if not hasattr(self, 'memory') or not isinstance(self.memory, list):
            self.memory = []
        self.memory.append(monologue)
        if len(self.memory) > 50:
            self.memory.pop(0)

    @property
    def current_strategy(self):
        return self.active_strategy

    @current_strategy.setter
    def current_strategy(self, value):
        self.active_strategy = value
