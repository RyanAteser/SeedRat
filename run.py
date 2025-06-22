from agent.jepa import JEPA
import json
import os
import time

import pygame
import torch

from agent.jepa import JEPA
from agent.seedrat import SeedRat
from env.world import Environment
from plt import plot_learning_curve


class GoalArbitrationLayer:
    def __init__(self):
        self.last_known_food_locations = []
        self.tile_values = {}
        self.value_decay_rate = 0.95

        # Initialize attributes used in arbitrate_goal
        self.current_high_level_goal = "explore_environment" # Default starting goal
        self.current_subplan_step = "move_randomly"        # Default starting subplan step
        self.goal_history = []
        self.subplan_history = []
        self.food_search_attempts = 0
        self.return_to_burrow_attempts = 0

    def update_tile_values(self, position, scent_strength, found_food=False):
        # Ensure position is hashable (e.g., a tuple)
        if not isinstance(position, tuple):
            position = tuple(position)

        if found_food:
            self.tile_values[position] = 10.0  # boost
            if position not in self.last_known_food_locations:
                self.last_known_food_locations.append(position)
            # Keep the list from growing indefinitely, e.g., last 10
            if len(self.last_known_food_locations) > 10:
                self.last_known_food_locations.pop(0)
        else:
            old_value = self.tile_values.get(position, 0.0)
            # New value is influenced by current scent and decayed old value
            new_value = max(scent_strength, old_value * self.value_decay_rate)
            self.tile_values[position] = new_value

    def value_at(self, position):
        return self.tile_values.get(position, 0.0)

    def decay_all_values(self):
        # Create a list of keys to avoid RuntimeError: dictionary changed size during iteration
        for pos in list(self.tile_values.keys()):
            self.tile_values[pos] *= self.value_decay_rate
            if self.tile_values[pos] < 0.01: # Remove if value too low
                del self.tile_values[pos]

    def arbitrate_goal(self, rat, env, scent_strength, prediction_error):
        """
        Arbitrates the rat's current high-level goal and immediate sub-plan step
        based on its internal state and environmental cues.

        Args:
            rat: The rat agent object (SeedRat instance).
            env: The environment object.
            scent_strength: The current food scent strength at rat's position.
            prediction_error: The JEPA prediction error.
        """
        # Update goal and subplan history
        self.goal_history.append(self.current_high_level_goal)
        self.subplan_history.append(self.current_subplan_step)

        # Ensure rat has a burrow_pos for comparison
        if not hasattr(rat, 'burrow_pos') or rat.burrow_pos is None:
            # Fallback if burrow_pos isn't set (e.g., set to rat's starting pos if no explicit burrow)
            rat.burrow_pos = (2,15) # Example: set a default burrow position if not found

        # Determine if burrow is nearby based on current surroundings
        surroundings = env.get_surroundings(rat.pos) # Assuming env has this method
        burrow_nearby = any(t == "B" for t in surroundings.values())
        rat.burrow_nearby = burrow_nearby # Optionally update rat's attribute

        # --- High-Level Goal Arbitration ---
        prev_high_level_goal = self.current_high_level_goal # For logging changes

        if rat.food_acquired: # Check if rat just acquired food
            self.current_high_level_goal = "return_to_burrow" # High priority to return
            rat.log_memory("Arbitration: Food acquired! Prioritizing 'return_to_burrow'.")
        elif rat.pos == rat.burrow_pos and rat.hunger < 20 and rat.energy > 60:
            # If in burrow, not hungry, and good energy, reset food acquired and rest
            if self.current_high_level_goal != "rest_and_digest":
                rat.food_acquired = False # Reset for next food foraging cycle
                self.current_high_level_goal = "rest_and_digest"
                rat.log_memory("Arbitration: In burrow, fed, and rested. Shifting to 'rest_and_digest'.")
        elif rat.energy < 30: # Low energy, go home
            self.current_high_level_goal = "return_to_burrow"
            self.return_to_burrow_attempts += 1
            rat.log_memory(f"Arbitration: Energy low ({rat.energy:.1f}). Prioritizing 'return_to_burrow'.")
        elif rat.hunger > 60: # Very hungry, immediate food search
            self.current_high_level_goal = "secure_food"
            self.food_search_attempts += 1
            rat.log_memory(f"Arbitration: Hunger very high ({rat.hunger:.1f}). Prioritizing 'secure_food'.")
        elif rat.fear > 50 and burrow_nearby: # High fear, seek safety if burrow is near
            self.current_high_level_goal = "seek_safety"
            rat.log_memory(f"Arbitration: Fear high ({rat.fear:.1f}) and burrow nearby. Prioritizing 'seek_safety'.")
        elif rat.fear > 50: # High fear, but no burrow nearby - try to evade
            self.current_high_level_goal = "evade_danger"
            rat.log_memory(f"Arbitration: Fear high ({rat.fear:.1f}) and no burrow near. Prioritizing 'evade_danger'.")
        elif scent_strength > 0.4: # Strong food scent, prioritize following it
            self.current_high_level_goal = "secure_food" # Still secure_food, but subplan will be specific
            rat.log_memory(f"Arbitration: Strong food scent ({scent_strength:.2f}). Prioritizing 'secure_food'.")
        elif prediction_error < 0.15 and rat.curiosity < 1.0: # JEPA is confident and curiosity not driving
            self.current_high_level_goal = "exploit_prediction" # Follow JEPA's confident path
            rat.log_memory(f"Arbitration: JEPA confident ({prediction_error:.2f}). Exploiting prediction.")
        elif rat.curiosity > 1.2: # High curiosity, explore
            self.current_high_level_goal = "explore_environment"
            rat.log_memory(f"Arbitration: Curiosity high ({rat.curiosity:.1f}). Prioritizing 'explore_environment'.")
        else: # Default exploration
            self.current_high_level_goal = "explore_environment"
            rat.log_memory("Arbitration: Defaulting to 'explore_environment'.")

        if prev_high_level_goal != self.current_high_level_goal:
            rat.log_memory(f"🎯 High-Level Goal Changed: {prev_high_level_goal} -> {self.current_high_level_goal}")


        # --- Sub-Plan Step Arbitration based on High-Level Goal ---
        prev_subplan_step = self.current_subplan_step # For logging changes

        if self.current_high_level_goal == "secure_food":
            if scent_strength > 0.6:
                self.current_subplan_step = "follow_gradient"
            elif len(self.last_known_food_locations) > 0 and rat.hunger > 30:
                self.current_subplan_step = "navigate_to_known_food" # New subplan step
            else:
                self.current_subplan_step = "scan_quadrant" # Or 'move_randomly_for_food'
        elif self.current_high_level_goal == "return_to_burrow":
            if rat.pos == rat.burrow_pos:
                self.current_subplan_step = "rest_in_burrow"

        elif self.current_high_level_goal == "seek_safety":
            self.current_subplan_step = "hide_in_burrow"
        elif self.current_high_level_goal == "evade_danger":
            self.current_subplan_step = "move_away_from_danger" # New subplan step
        elif self.current_high_level_goal == "rest_and_digest":
            self.current_subplan_step = "idle_in_burrow"
        elif self.current_high_level_goal == "exploit_prediction":
            self.current_subplan_step = "simulate_and_follow"
        elif self.current_high_level_goal == "explore_environment":
            if rat.curiosity > 0.8:
                self.current_subplan_step = "explore_novel_tile"
            elif rat.stuck_counter > 2:
                self.current_subplan_step = "reorient_and_move"
            else:
                self.current_subplan_step = "move_randomly"
        else:
            self.current_subplan_step = "idle" # Fallback subplan

        if prev_subplan_step != self.current_subplan_step:
            rat.log_memory(f"➡️ Subplan Step Changed: {prev_subplan_step} -> {self.current_subplan_step}")

        return self.current_high_level_goal, self.current_subplan_step

    # Removed generate_subplan and execute_next_substep as current_subplan_step directly
    # indicates the immediate next action *type* for the LLM to interpret, not a sequence.
    # If true multi-step subplans are needed, this class would need more sophisticated state management.



class ActionResult:
    """Structured action result for better error handling"""
    def __init__(self, success: bool, new_position: tuple,
                 result_type: str, message: str = ""):
        self.success = success
        self.new_position = new_position
        self.result_type = result_type
        self.message = message

def save_reflection_logs(reflection_log, output_dir="dashboard"):
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(os.path.join(output_dir, "reflection_logs.jsonl"), "w") as f:
            for entry in reflection_log:
                f.write(json.dumps(entry) + "\n")
        print("✅ Reflection logs saved.")
    except Exception as e:
        print(f"⚠️ Error saving reflection logs: {e}")

import random

def detect_doors(surroundings):
    """Returns list of directions where a door ('D') is found in surroundings."""
    return [direction for direction, tile in surroundings.items() if tile == "D"]

def enhanced_llm_monologue_with_action(
        hunger, fear, curiosity, energy,
        memory, surroundings, mom_memory, goal,
        food_scent, danger_scent, burrow_nearby, predicted_plan, jepa_prediction,
        current_strategy, plan_confidence, active_subgoal, key_concepts, age,
        walls_examined, stuck_counter, failed_actions,
        current_high_level_goal, current_subplan_step
):
    """
    Simulates an LLM generating a monologue and choosing an action,
    now guided by the high-level goal and subplan step.
    """

    # 🔍 Add door awareness
    door_directions = detect_doors(surroundings)
    door_hint = (
        f"I detect a door nearby to the {', '.join(door_directions)}."
        if door_directions else
        "There are no visible doors nearby."
    )

    thought_process = f"My overarching goal is to {current_high_level_goal}. "
    thought_process += f"The immediate task is to {current_subplan_step}. "
    thought_process += f" {door_hint} "

    # 🧠 Choose an action based on subplan
    if current_subplan_step == 'follow_gradient':
        thought_process += f"I detect food scent strength of {food_scent:.2f}. I should move towards the highest gradient."
        action = random.choice(['move_up', 'move_down', 'move_left', 'move_right'])  # placeholder
    elif current_subplan_step == 'scan_quadrant':
        thought_process += "My objective is to explore this quadrant. I should try to move to a novel area."
        if door_directions:
            action = f"move_{random.choice(door_directions)}"
            thought_process += f" Since I see a door, I will attempt to go through it by moving {action}."
        else:
            action = random.choice(['move_up', 'move_down', 'move_left', 'move_right', 'idle'])
    elif current_subplan_step == 'move_randomly':
        thought_process += "No clear directive, so I'll move randomly to discover new areas."
        action = random.choice(['move_up', 'move_down', 'move_left', 'move_right'])
    elif current_subplan_step == 'navigate_home':
        thought_process += f"I need to return to the burrow. Current energy: {energy:.1f}. Prioritizing direct path."
        action = "move_down"  # placeholder
    elif current_subplan_step == 'simulate_and_follow':
        thought_process += f"JEPA predicts: {jepa_prediction}. I will follow this prediction."
        action = predicted_plan
    else:
        thought_process += "No specific subplan step. Defaulting to general survival instincts."
        action = predicted_plan

    # 📉 Risk & exploration modifiers
    if hunger > 70:
        thought_process += " Hunger is very high; finding food is paramount."
        if food_scent > 0.5 and current_subplan_step != 'follow_gradient':
            thought_process += " I sense strong food scent, should I adjust to follow it?"
    if fear > 50:
        thought_process += " Fear is elevated; need to be cautious and potentially seek safety."
    if curiosity > 2.0:
        thought_process += " Curiosity is piqued; I should explore novel surroundings."
    if stuck_counter > 0:
        thought_process += f" I've been stuck {stuck_counter} times recently. I should try a different approach."
        if action in failed_actions:
            possible_actions = [a for a in ['move_up', 'move_down', 'move_left', 'move_right'] if a not in failed_actions]
            if possible_actions:
                action = random.choice(possible_actions)
                thought_process += f" Avoiding failed action. New chosen action: {action}."
            else:
                thought_process += " All directions seem to be failed actions. I'm truly stuck!"

    # 🎯 Confidence calculation
    confidence = plan_confidence + (0.1 if food_scent > 0.5 else 0) - (0.1 if fear > 50 else 0)
    confidence = max(0.1, min(0.95, confidence))  # clamp to valid range

    prediction_text = f"I predict that taking '{action}' will lead to progress in '{current_subplan_step}'."
    strategy_update = current_strategy

    return {
        "thought": thought_process,
        "action": action,
        "prediction": prediction_text,
        "strategy_update": strategy_update,
        "confidence": confidence
    }


def extract_field(lines, field_name, default_value):
    """Extract a specific field from LLM response lines"""
    field_line = next((line for line in lines if line.startswith(field_name)), "")
    return field_line.replace(field_name, "").strip() if field_line else default_value

def validate_action(action, available_actions, fallback_action):
    """Validate and normalize LLM action to match expected format"""

    # Normalize "move_up" to "up"
    if action.startswith("move_"):
        action = action.replace("move_", "")

    valid_basic_actions = ["up", "down", "left", "right"]
    valid_prefixes = ["investigate_", "dig_", "climb_", "squeeze_"]

    # Check basic movement
    if action in valid_basic_actions:
        return action

    # Check special action with valid format
    for prefix in valid_prefixes:
        if action.startswith(prefix):
            direction = action.split("_")[1]
            if direction in valid_basic_actions:
                return action

    print(f"⚠️ Invalid or unavailable action received from LLM: '{action}'. Falling back to predicted plan: '{fallback_action}'")
    return fallback_action


def fallback_reasoning(hunger, fear, curiosity, danger_scent, food_scent,
                       stuck_counter, predicted_plan, current_strategy, surroundings):
    """Fallback reasoning when LLM is unavailable"""
    if danger_scent > 0.5:
        return {
            "thought": "Danger detected! Engaging evasive maneuvers based on scent analysis.",
            "action": "left",
            "prediction": "Moving away from danger source to ensure survival.",
            "strategy_update": "danger_avoidance",
            "confidence": 0.8
        }
    elif food_scent > 0.3:
        return {
            "thought": "Food scent detected. Prioritizing nutritional objective for survival.",
            "action": predicted_plan,
            "prediction": "Following scent gradient should lead to food source.",
            "strategy_update": "food_seeking",
            "confidence": 0.7
        }
    elif stuck_counter > 3:
        investigation_directions = [d for d in ["up", "down", "left", "right"]
                                    if surroundings.get(d) == "Wall"]
        chosen_dir = random.choice(investigation_directions) if investigation_directions else "up"
        return {
            "thought": f"Stuck for {stuck_counter} steps. Investigating walls for hidden passages or weaknesses.",
            "action": f"investigate_{chosen_dir}",
            "prediction": "Investigation may reveal structural weaknesses or hidden passages.",
            "strategy_update": "exploration",
            "confidence": 0.6
        }
    else:
        return {
            "thought": f"Executing {current_strategy} strategy with systematic approach.",
            "action": predicted_plan,
            "prediction": "Continuing with planned action sequence.",
            "strategy_update": current_strategy,
            "confidence": 0.5
        }

def handle_enhanced_action_with_feedback(action, rat, env, llm_result):
    """Enhanced action handler that incorporates LLM predictions and provides detailed feedback"""

    # Log the LLM's prediction before executing
    rat.log_memory(f"🧠 AI Prediction: {llm_result['prediction']}")
    rat.log_memory(f"🎯 Action Confidence: {llm_result['confidence']:.1%}")

    # Validate action format
    if not isinstance(action, str) or len(action) == 0:
        rat.log_memory("❌ Invalid action format received")
        return create_action_result(False, rat.pos, "invalid_action", "Action format error")

    # Handle investigation actions
    if action.startswith("investigate_"):
        return handle_investigation_action(action, rat, env, llm_result)

    # Handle special actions (dig, climb, squeeze)
    elif action.startswith(("dig_", "climb_", "squeeze_")):
        return handle_special_action(action, rat, env, llm_result)

    # Handle regular movement
    else:
        return handle_regular_movement_with_feedback(action, rat, env, llm_result)

def handle_investigation_action(action, rat, env, llm_result):
    """Handle investigation actions with LLM feedback integration"""
    direction = action.split("_")[1] if "_" in action else None

    if direction not in ["up", "down", "left", "right"]:
        rat.log_memory("❌ Invalid investigation direction")
        return create_action_result(False, rat.pos, "invalid_direction", "Invalid direction")

    result = rat.investigate_wall(direction, env)

    # Enhanced investigation feedback with LLM prediction comparison
    investigation_messages = {
        "weak_wall": f"🔍 Discovery: {direction} wall shows weakness! LLM predicted: {llm_result['prediction']}",
        "small_passage": f"🔍 Found: Small passage to {direction}! This aligns with curiosity-driven exploration.",
        "crack": f"🔍 Analysis: Crack detected in {direction} wall suggests hollow space beyond.",
        "already_examined": f"🔍 Memory: {direction} wall previously analyzed. Confidence was {llm_result['confidence']:.1%}",
        "solid_wall": f"🔍 Assessment: {direction} wall completely solid. Strategy may need adjustment."
    }

    message = investigation_messages.get(result, f"Investigated {direction} wall")
    rat.log_memory(message)

    # Update strategy based on investigation results
    if result in ["weak_wall", "small_passage", "crack"]:
        rat.log_memory(f"💡 Investigation successful! Updating strategy from {rat.current_strategy}")
        rat.current_strategy = "targeted_breakthrough"

    return create_action_result(True, rat.pos, result, message)

def handle_special_action(action, rat, env, llm_result):
    """Handle special actions with LLM reasoning integration"""
    action_type, direction = action.split("_", 1)

    if direction not in ["up", "down", "left", "right"]:
        rat.log_memory(f"❌ Invalid direction for {action_type}: {direction}")
        return create_action_result(False, rat.pos, "invalid_direction", f"Invalid {action_type} direction")

    success, result = rat.attempt_special_action(action_type, direction, env)

    if success:
        new_pos = get_new_position(rat.pos, direction, env)
        success_messages = {
            "dig": f"⛏️ Success: Breakthrough achieved! LLM reasoning was correct: {llm_result['thought'][:50]}...",
            "climb": f"🧗 Success: Overcame obstacle using advanced problem-solving!",
            "squeeze": f"🤏 Success: Navigated tight passage with precision!"
        }
        rat.log_memory(success_messages.get(action_type, f"Successfully {action_type}ed {direction}"))

        # Reward successful LLM prediction
        if llm_result['confidence'] > 0.7:
            rat.log_memory("🎯 High-confidence LLM prediction proved correct! Boosting curiosity reward.")
            rat.curiosity += 0.2

        return create_action_result(True, new_pos, f"{action_type}_success", "Special action succeeded")
    else:
        failure_messages = {
            "dig": f"⛏️ Failed: {result}. LLM prediction: {llm_result['prediction']}",
            "climb": f"🧗 Failed: {result}. Reassessing approach.",
            "squeeze": f"🤏 Failed: {result}. Space too narrow."
        }
        rat.log_memory(failure_messages.get(action_type, f"Failed to {action_type} {direction}"))
        return create_action_result(False, rat.pos, f"{action_type}_failed", result)

def handle_regular_movement_with_feedback(action, rat, env, llm_result):
    """Handle regular movement with LLM prediction feedback"""
    direction_map = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0)
    }

    if action in direction_map:
        dx, dy = direction_map[action]
        new_x = rat.pos[0] + dx
        new_y = rat.pos[1] + dy

        # Check bounds
        if 0 <= new_x < env.width and 0 <= new_y < env.height:
            target_tile = env.grid[new_y][new_x]

            # Normalize tile case for comparison
            normalized_tile = str(target_tile).lower()

            # Allow movement if not a wall
            if normalized_tile not in ["wall", "#"]:
                rat.log_memory(f"🚶 Moving {action} to ({new_x}, {new_y}). LLM confidence: {llm_result['confidence']:.1%}")

                # Reward or warn based on prediction
                if normalized_tile == "f" and "food" in llm_result['prediction'].lower():
                    rat.log_memory("🎯 LLM correctly predicted food location! Excellent reasoning.")
                elif normalized_tile == "n" and "danger" in llm_result['prediction'].lower():
                    rat.log_memory("⚠️ LLM correctly predicted danger! Superior threat assessment.")

                return create_action_result(True, (new_x, new_y), target_tile, f"Moved {action}")
            else:
                rat.log_memory(f"🚫 Cannot move {action} - wall detected: '{target_tile}' (normalized: '{normalized_tile}').")
                return create_action_result(False, rat.pos, "blocked_by_wall", "Wall blocking movement")
        else:
            rat.log_memory(f"🚫 Cannot move {action} - boundary limit reached.")
            return create_action_result(False, rat.pos, "out_of_bounds", "Boundary reached")

    # Invalid action fallback
    rat.log_memory(f"❓ Unknown action: {action}")
    return create_action_result(False, rat.pos, "unknown_action", "Action not recognized")


def create_action_result(success, position, result_type, message):
    """Create standardized action result"""
    return ActionResult(success, position, result_type, message)

def get_new_position(current_pos, direction, env):
    """Calculate new position after successful action"""
    x, y = current_pos
    direction_map = {
        "up": (x, max(0, y - 1)),
        "down": (x, min(env.height - 1, y + 1)),
        "left": (max(0, x - 1), y),
        "right": (min(env.width - 1, x + 1), y)
    }
    return direction_map.get(direction, current_pos)

def get_surroundings(pos, env):
    """Get surroundings for a position with bounds checking"""
    x, y = pos
    surroundings = {}
    surroundings["up"] = env.grid[y - 1][x] if y > 0 else "Wall"
    surroundings["down"] = env.grid[y + 1][x] if y < env.height - 1 else "Wall"
    surroundings["left"] = env.grid[y][x - 1] if x > 0 else "Wall"
    surroundings["right"] = env.grid[y][x + 1] if x < env.width - 1 else "Wall"
    return surroundings


def handle_stuck_situation(rat, env, surroundings, food_scent, current_goal):
    """
    Advanced stuck handling with multiple fallback strategies
    """
    available_moves = []

    # Find all non-wall moves
    for direction in ["up", "down", "left", "right"]:
        tile = surroundings.get(direction, "Wall").lower()
        if tile not in ["wall", "#"] and direction not in rat.failed_actions:
            available_moves.append(direction)

    # Strategy 1: Follow strongest scent gradient (if not returning to burrow)
    if current_goal != "return_to_burrow" and food_scent > 0:
        gradients = rat.detect_scent_gradient(env, "F")
        if gradients:
            best_scent_direction = max(gradients, key=gradients.get)
            if best_scent_direction in available_moves:
                rat.log_memory(f"🔍 Stuck strategy 1: Following scent gradient {best_scent_direction}")
                return best_scent_direction

    # Strategy 2: Head towards burrow if trying to return
    if current_goal == "return_to_burrow" and hasattr(rat, 'burrow_pos'):
        burrow_direction = get_direction_to_target(rat.pos, rat.burrow_pos)
        if burrow_direction in available_moves:
            rat.log_memory(f"🏠 Stuck strategy 2: Direct path to burrow {burrow_direction}")
            return burrow_direction

    # Strategy 3: Explore unvisited areas
    if hasattr(rat, 'visited_positions'):
        for direction in available_moves:
            next_pos = get_next_position(rat.pos, direction)
            if next_pos not in rat.visited_positions:
                rat.log_memory(f"🗺️ Stuck strategy 3: Exploring unvisited area {direction}")
                return direction

    # Strategy 4: Random exploration from available moves
    if available_moves:
        import random
        random_choice = random.choice(available_moves)
        rat.log_memory(f"🎲 Stuck strategy 4: Random exploration {random_choice}")
        return random_choice

    # Strategy 5: Clear failed actions and try again
    if rat.failed_actions:
        rat.log_memory("🧹 Stuck strategy 5: Clearing failed actions for fresh start")
        rat.failed_actions.clear()
        all_directions = ["up", "down", "left", "right"]
        for direction in all_directions:
            tile = surroundings.get(direction, "Wall").lower()
            if tile not in ["wall", "#"]:
                return direction

    # Last resort: try any direction
    rat.log_memory("⚠️ Last resort: Attempting any direction")
    return "up"  # Default fallback


def get_direction_to_target(current_pos, target_pos):
    """Calculate the best direction to move towards a target position"""
    curr_x, curr_y = current_pos
    target_x, target_y = target_pos

    dx = target_x - curr_x
    dy = target_y - curr_y

    # Prioritize the axis with larger distance
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


def get_next_position(current_pos, direction):
    """Calculate the next position given a direction"""
    x, y = current_pos

    if direction == "up":
        return (x, y - 1)
    elif direction == "down":
        return (x, y + 1)
    elif direction == "left":
        return (x - 1, y)
    elif direction == "right":
        return (x + 1, y)


    return current_pos

def calculate_action_confidence(action, tile_content, hunger, curiosity, rat, fallback_used):
    """
    Calculates a confidence score for a chosen action based on various factors.

    Args:
        action (str): The action chosen by the rat (e.g., "up", "dig_left").
        tile_content (str): The content of the tile the rat is moving to or interacting with.
        hunger (float): Current hunger level of the rat.
        curiosity (float): Current curiosity level of the rat.
        rat: The rat agent object (SeedRat instance), used to access fear and energy.
        fallback_used (bool): True if a fallback action was used instead of the LLM's primary suggestion.

    Returns:
        float: A confidence score between 0.0 and 1.0.
    """
    confidence = 0.50 # Default baseline

    if fallback_used:
        confidence -= 0.10 # Slightly lower confidence if fallback was engaged

    # Specific action-based confidence boosts
    if action in ["eat", "consume_food"]:
        confidence = max(confidence, 0.95) # Very high confidence for eating
    elif action.startswith("investigate_"):
        confidence = max(confidence, 0.70) # Good confidence for investigation
    elif action.startswith(("dig_", "climb_", "squeeze_")):
        confidence = max(confidence, 0.75) # Good confidence for special actions

    # Tile-content and internal state based boosts
    if tile_content.lower() in ["f", "food"]:
        if hunger > 60: # Very hungry and moving to food
            confidence = max(confidence, 0.90)
        elif hunger > 20: # Just hungry and moving to food
            confidence = max(confidence, 0.85)
    elif tile_content.lower() in ["d", "door"]:
        if curiosity > 1.5: # Curious and moving to a door
            confidence = max(confidence, 0.70)
        elif hunger > 50: # Hungry, but still considering a door (might be food beyond)
            confidence = max(confidence, 0.60)
    elif tile_content.lower() in ["b", "burrow"]:
        if hunger < 30 and rat.energy < 50: # Returning to burrow when needed
            confidence = max(confidence, 0.90)
    elif tile_content.lower() in ["#", "wall"]:
        confidence = min(confidence, 0.10) # Very low confidence if trying to move into a wall

    # General state modifiers
    confidence -= (rat.fear / 200) # Higher fear reduces confidence, accessed from rat object
    confidence += (curiosity / 300) # Higher curiosity slightly increases confidence (for exploration)


    return max(0.01, min(1.0, confidence)) # Clamp between 0 and 1 # Clamp between 0 and 1


def enhanced_simulation_step_with_monologue(rat, env, jepa, current_goal, current_subplan_step):
    """Enhanced simulation step that fully integrates LLM monologue with intelligent door logic"""

    if not hasattr(rat, 'failed_actions'):
        rat.failed_actions = set()
    if not hasattr(rat, 'food_acquired'):
        rat.food_acquired = False
    if not hasattr(rat, 'je_pa_errors'):
        rat.je_pa_errors = []

    surroundings = get_surroundings(rat.pos, env)
    food_scent = env.get_scent_map("F", rat.pos)
    danger_scent = env.get_scent_map("N", rat.pos)

    context = rat.encode_state_enhanced(surroundings, food_scent, danger_scent,
                                        burrow_nearby=any(t == "B" for t in surroundings.values()))

    predicted_action = rat.plan_with_world_model(surroundings, env, current_goal)

    action_result = None # Initialize action_result before the try block

    try:
        reflection = rat.get_reflection()
        llm_result = enhanced_llm_monologue_with_action(
            hunger=rat.hunger, fear=rat.fear, curiosity=rat.curiosity,
            memory=[mem for mem in rat.memory[-3:] if isinstance(mem, str)],
            surroundings=surroundings, mom_memory=rat.mom_memory, goal=rat.goal,
            food_scent=food_scent, danger_scent=danger_scent,
            burrow_nearby=any(t == "B" for t in surroundings.values()),
            predicted_plan=predicted_action,
            jepa_prediction=rat.generate_hypotheses(surroundings, {"food": food_scent, "danger": danger_scent}),
            current_strategy=rat.current_strategy, plan_confidence=rat.plan_confidence,
            active_subgoal=current_subplan_step,
            key_concepts=reflection['key_concepts'],
            age=rat.age, walls_examined=len(rat.walls_examined), stuck_counter=rat.stuck_counter,
            failed_actions=rat.failed_actions,
            current_high_level_goal=current_goal,
            current_subplan_step=current_subplan_step,
            energy=rat.energy
        )

        chosen_action = validate_action(
            llm_result['action'],
            available_actions=["up", "down", "left", "right"], # Assuming these are the primary LLM-suggested moves
            fallback_action=predicted_action
        )

        # 🧠 Intelligent Door Logic (Consolidated and Primary Block)
        # Determine the tile the chosen_action *would* lead to
        # Ensure chosen_action is just a direction (e.g., "up", "left") for get_new_position
        action_direction = chosen_action.replace("move_", "")
        potential_next_pos = get_new_position(rat.pos, action_direction, env)

        target_tile_content = "Wall" # Default if out of bounds
        if 0 <= potential_next_pos[0] < env.width and 0 <= potential_next_pos[1] < env.height:
            target_tile_content = env.grid[potential_next_pos[1]][potential_next_pos[0]]

        normalized_target_tile = str(target_tile_content).lower()

        if normalized_target_tile in ["d", "door"]:
            food_gradients = rat.detect_scent_gradient(env, "F")
            strongest_scent = max(food_gradients.values(), default=0)

            rat.log_memory(f"🔍 Door decision: potential move '{chosen_action}'. Target tile: '{target_tile_content}'. Strongest food scent in gradient: {strongest_scent:.2f}. Food acquired: {rat.food_acquired}. Stuck Counter: {rat.stuck_counter}")

            # Decision logic for door passage
            # Prioritize escaping a stuck situation or going to burrow with food
            if rat.stuck_counter >= 3:
                rat.log_memory("🚪 Stuck near door. Forcing door exploration due to loop.")
                # chosen_action remains the same, proceeding to try the door
            elif rat.food_acquired:
                rat.log_memory("✅ Food acquired. Door allowed to seek burrow or explore further (e.g., for return path).")
                # chosen_action remains the same
            elif strongest_scent < 0.1: # If no strong food scent, exploring through door is good
                rat.log_memory("🧠 No strong food scent in current room. Exploring through door.")
                # chosen_action remains the same
            else:
                rat.log_memory("🚫 Door blocked: scent is strong and not stuck. Avoiding transition for now.")
                chosen_action = predicted_action  # Fallback if not ready for door
                # Need to re-evaluate potential_next_pos and target_tile_content if fallback occurs here
                action_direction = chosen_action.replace("move_", "")
                potential_next_pos = get_new_position(rat.pos, action_direction, env)
                if 0 <= potential_next_pos[0] < env.width and 0 <= potential_next_pos[1] < env.height:
                    target_tile_content = env.grid[potential_next_pos[1]][potential_next_pos[0]]
                else:
                    target_tile_content = "Wall"


        # action_result is determined ONLY ONCE here after all logic for chosen_action
        action_result = handle_enhanced_action_with_feedback(chosen_action, rat, env, llm_result)

        if llm_result['strategy_update'] != rat.current_strategy:
            rat.log_memory(f"🧠 Strategy updated: {rat.current_strategy} → {llm_result['strategy_update']}")
            rat.current_strategy = llm_result['strategy_update']

    except Exception as e:
        print(f"⚠️ LLM integration error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback if LLM fails
        llm_result = {
            "thought": "Fallback reasoning active due to LLM error. Executing world model prediction.",
            "action": predicted_action,
            "prediction": "Executing world model prediction",
            "strategy_update": rat.current_strategy,
            "confidence": 0.3
        }
        chosen_action = predicted_action
        # In case of LLM error, still determine action_result
        action_result = handle_enhanced_action_with_feedback(chosen_action, rat, env, llm_result)


    # The logic below should now rely on the single action_result obtained above.
    # Adjust stuck_counter and failed_actions based on the final action_result
    if not action_result.success:
        rat.stuck_counter += 1
        # Add to failed_actions if it's a movement action or special action that failed
        if chosen_action.startswith("move_"):
            rat.failed_actions.add(chosen_action.replace("move_", "")) # Normalize to just direction
        elif chosen_action in ["up", "down", "left", "right"]:
            rat.failed_actions.add(chosen_action)
        elif action_result.result_type.endswith("_failed"): # Catches dig_failed, climb_failed, etc.
            rat.failed_actions.add(chosen_action)
        rat.log_memory(f"❌ Action '{chosen_action}' failed. Stuck counter: {rat.stuck_counter}. Added to failed actions: {action_result.message}")
    else: # Action was successful, reset stuck counter and remove from failed list
        rat.stuck_counter = 0
        # If a previously failed action is now successful, remove it from the set
        normalized_chosen_action = chosen_action.replace("move_", "") # For moves
        if normalized_chosen_action in rat.failed_actions:
            rat.failed_actions.remove(normalized_chosen_action)
            rat.log_memory(f"✅ Removed '{normalized_chosen_action}' from failed actions - now successful.")
        elif chosen_action in rat.failed_actions: # For special actions
            rat.failed_actions.remove(chosen_action)
            rat.log_memory(f"✅ Removed '{chosen_action}' from failed actions - now successful.")


    rat.pos = action_result.new_position

    if action_result.result_type == "F" and not rat.food_acquired:
        rat.food_acquired = True
        rat.hunger = max(0, rat.hunger - 30)
        rat.energy = min(100, rat.energy + 20)
        rat.log_memory(f"🥣 Food acquired at position {rat.pos}. Hunger reduced.")
        if rat.current_strategy != "secure_and_exit":
            rat.current_strategy = "secure_and_exit"
            rat.log_memory("🔄 Strategy shift to 'secure_and_exit' after acquiring food.")

    # Room exploration boost from doors (only if the action actually resulted in entering a door tile)
    if action_result.result_type == "D":
        curiosity_boost = 0.2
        rat.curiosity += curiosity_boost
        rat.log_memory(f"🚪 Entered a door at {rat.pos}. Curiosity increased by {curiosity_boost:.1f}. New curiosity: {rat.curiosity:.2f}")
        if rat.current_strategy != "room_transition":
            rat.log_memory("🔄 Strategy update: transitioning rooms.")
            rat.current_strategy = "room_transition"

    next_context = rat.encode_state_enhanced(
        get_surroundings(rat.pos, env),
        env.get_scent_map("F", rat.pos),
        env.get_scent_map("N", rat.pos),
        any(t == "B" for t in get_surroundings(rat.pos, env).values())
    )

    tile = action_result.result_type if action_result.result_type in ["F", "N", "B", " ", "D", "C"] else " " # Include 'D' and 'C' in recognized tiles for update_emotions
    rat.update_emotions_and_state(tile, context)

    # JEPA training and prediction error calculation
    jepa.train()
    input_tensor = torch.tensor(context.detach().clone(), dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(next_context.detach().clone(), dtype=torch.float32).unsqueeze(0)
    loss = jepa.compute_loss(input_tensor, target_tensor)
    loss.backward()

    prediction_error = loss.item()
    rat.curiosity += min(prediction_error, 0.5) # Increase curiosity for high prediction error
    rat.curiosity *= 0.97 # Decay curiosity over time
    rat.je_pa_errors.append(prediction_error)

    if len(rat.je_pa_errors) % 10 == 0:
        avg_error = sum(rat.je_pa_errors[-10:]) / 10
        rat.log_memory(f"📉 Avg JEPA error (last 10): {avg_error:.4f}")

    if action_result.success and llm_result['confidence'] > 0.8:
        rat.log_memory(f"🎯 High-confidence LLM prediction succeeded! Reward: +0.1 curiosity")
        rat.curiosity += 0.1

    rat.log_memory(f"🧪 JEPA prediction error: {prediction_error:.4f} | LLM confidence: {llm_result['confidence']:.2%}")

    # Handle extreme stuck behavior (override)
    if rat.stuck_counter >= 3:
        print("🚨 Extreme stuck behavior detected. Forcing scent-based override...")
        gradients = rat.detect_scent_gradient(env, "F") # Assuming this returns a dict like {'up': 0.5, 'down': 0.1}
        if gradients:
            # Find the direction with the strongest food scent
            best_scent_direction = max(gradients, key=gradients.get)
            override_action = validate_action(best_scent_direction, ["up", "down", "left", "right"], fallback_action=best_scent_direction)

            # Check what's in that direction
            override_potential_next_pos = get_new_position(rat.pos, override_action, env)
            override_target_tile_content = "Wall"
            if 0 <= override_potential_next_pos[0] < env.width and 0 <= override_potential_next_pos[1] < env.height:
                override_target_tile_content = env.grid[override_potential_next_pos[1]][override_potential_next_pos[0]]

            if str(override_target_tile_content).lower() not in ["wall", "#"]:
                rat.log_memory(f"🛠️ Override: Attempting to move {override_action} towards strong scent ('{override_target_tile_content}')")
                override_result = handle_enhanced_action_with_feedback(override_action, rat, env, {
                    "thought": "Emergency override active. Following strongest scent gradient due to being stuck.",
                    "prediction": f"Moving {override_action} towards potential food source.",
                    "confidence": 0.9 # High confidence for emergency override
                })
                rat.pos = override_result.new_position
                rat.failed_actions.clear() # Clear failed actions on successful override
                rat.stuck_counter = 0
                chosen_action = override_action # Update chosen_action for logging
                rat.current_strategy = "emergency_override"
                rat.log_memory(f"🛠️ Emergency override successful to {rat.pos} - cleared failed actions.")
                action_result = override_result # Update action_result with override result
            else:
                rat.log_memory(f"⚠️ Could not apply scent-based override: {override_action} leads to a wall ('{override_target_tile_content}').")
        else:
            rat.log_memory("⚠️ No scent gradients detected for stuck override. Still stuck.")

    if hasattr(jepa, 'optimizer'):
        jepa.optimizer.step()
        jepa.optimizer.zero_grad()
    else:
        print("⚠️ JEPA optimizer not found. Skipping optimization step.")

    # ✅ Then Add This Right Before return {...} Block at the Bottom
    # 🔁 Final action confidence estimation (after override logic completes)
    # Ensure env has get_tile or pass env.grid to it
    final_tile_content_at_pos = env.grid[rat.pos[1]][rat.pos[0]] # Get the tile content at the rat's final position
    # Determine if a fallback was truly used. This is more nuanced.
    # A simple way is to check if stuck_counter is high, or if LLM's initial choice was overridden.
    fallback_used_for_confidence = (rat.stuck_counter > 0) or \
                                   (llm_result['action'].replace('move_', '') != chosen_action.replace('move_', ''))

    final_confidence = calculate_action_confidence(
        action=chosen_action, # The action that was ultimately taken
        tile_content=final_tile_content_at_pos, # The tile the rat ended up on
        hunger=rat.hunger,
        curiosity=rat.curiosity,
        rat=rat, # <--- **FIXED: Pass the rat object here**
        fallback_used=fallback_used_for_confidence
    )
    rat.log_memory(f"📈 Final action confidence rating: {final_confidence:.2f} on tile '{final_tile_content_at_pos}'")


    # So the final return becomes:
    return {
        "action": chosen_action,
        "result": action_result.result_type,
        "position": rat.pos,
        "llm_thought": llm_result['thought'],
        "llm_prediction": llm_result['prediction'],
        "llm_confidence": llm_result['confidence'],
        "action_success": action_result.success,
        "hypotheses": rat.generate_hypotheses(surroundings, {"food": food_scent, "danger": danger_scent}),
        "reflection": rat.get_reflection(),
        "surroundings": surroundings,
        "scents": {"food": food_scent, "danger": danger_scent},
        "strategy": rat.current_strategy,
        "scent_strength": food_scent, # This seems redundant with 'scents' but keeping as per your original
        "prediction_error": prediction_error,
        "final_action_confidence": final_confidence # ✅ Added this
    }
def fix_json_logging_error(dashboard_log):
    """
    Fix the JSON serialization error by converting tuple keys to strings
    """
    def convert_keys(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                # Convert tuple keys to string representation
                if isinstance(key, tuple):
                    new_key = str(key)
                else:
                    new_key = key
                new_dict[new_key] = convert_keys(value)
            return new_dict
        elif isinstance(obj, list):
            return [convert_keys(item) for item in obj]
        else:
            return obj

    return convert_keys(dashboard_log)
def render_env(env_map, rat_pos, predator_positions=None):
    SYMBOLS = {
        'wall': '#',
        'floor': '.',
        'crack': 'C',
        'food': 'F',
        'burrow': 'B',
        'predator': 'P',
        'rat': '🐭',  # or 'R' for terminal-safe
        'door': 'D'   # ADDED: Symbol for Door
    }

    rendered = ""
    for y, row in enumerate(env_map):
        for x, tile in enumerate(row):
            pos = (y, x)
            if pos == rat_pos:
                rendered += SYMBOLS['rat']
            elif predator_positions and pos in predator_positions:
                rendered += SYMBOLS['predator']
            elif tile == '#':
                rendered += SYMBOLS['wall']
            elif tile == '.':
                rendered += SYMBOLS['floor']
            elif tile == 'F':
                rendered += SYMBOLS['food']
            elif tile == 'B':
                rendered += SYMBOLS['burrow']
            elif tile == 'C':
                rendered += SYMBOLS['crack']
            elif tile == 'D': # ADDED: Condition for Door
                rendered += SYMBOLS['door']
            else:
                rendered += '?' # Fallback for any truly unhandled tile types
        rendered += "\n"
    print(rendered)



def load_environments(map_path="env/maps/hierarchical_labyrinth.json"):
    """Load environments with error handling"""
    try:
        print(f"🔁 Loading environment from {map_path}...")
        with open(map_path) as f:
            data = json.load(f)

        unified_grid = []
        for room in data["rooms"]:
            if unified_grid:
                unified_grid.append(["Wall"] * len(room["grid"][0]))
            for row in room["grid"]:
                unified_grid.append(row)

        print(f"✅ Loaded environment map: {os.path.basename(map_path)}")
        return [Environment.from_grid(unified_grid)]

    except FileNotFoundError:
        print(f"⚠️ Map file not found: {map_path}")
        print("Creating simple test environment...")

        simple_grid = [
            ["Wall", "Wall", "Wall", "Wall", "Wall"],
            ["Wall", " ", "F", " ", "Wall"],
            ["Wall", " ", "Wall", " ", "Wall"],
            ["Wall", " ", " ", "B", "Wall"],
            ["Wall", "Wall", "Wall", "Wall", "Wall"]
        ]
        return [Environment.from_grid(simple_grid)]

    except Exception as e:
        print(f"⚠️ Error loading environment: {e}")
        return []

# Here's the updated `run_enhanced_simulation_with_integrated_monologue` function
# with a Pygame GUI embedded inside it.


def run_enhanced_simulation_with_integrated_monologue_with_gui(
        steps=200, map_path="env/maps/hierarchical_labyrinth.json"
):
    TILE_SIZE = 32
    COLORS = {
        'Wall': (100, 100, 100),
        'Floor': (230, 230, 230),
        'F': (255, 50, 50),
        'B': (50, 200, 50),
        'C': (120, 70, 20),
        'P': (0, 0, 0),
        'Rat': (50, 50, 255),
        'ConsumedFood': (150, 100, 100)  # Darker red for consumed food locations
    }

    def initialize_pygame(grid_width, grid_height):
        pygame.init()
        screen = pygame.display.set_mode((grid_width * TILE_SIZE, grid_height * TILE_SIZE))
        pygame.display.set_caption("SeedRat AGI Simulation")
        return screen

    def draw_environment(screen, env, rat_pos, consumed_food_locations):
        for y, row in enumerate(env.grid):
            for x, tile in enumerate(row):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if (x, y) == rat_pos:
                    color = COLORS['Rat']
                elif (x, y) in consumed_food_locations:
                    color = COLORS['ConsumedFood']
                else:
                    color = COLORS.get(tile, COLORS['Floor'])
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)
        pygame.display.flip()

    def consume_food_at_position(env, pos, consumed_food_locations, rat):
        """Handle food consumption when rat reaches food tile"""
        x, y = pos
        if env.grid[y][x] == 'F':
            # Mark food as consumed
            consumed_food_locations.add(pos)
            # Remove food from environment (replace with floor)
            env.grid[y][x] = 'Floor'

            # Update rat's state
            rat.hunger = max(0, rat.hunger - 30)  # Reduce hunger significantly
            rat.energy = min(100, rat.energy + 20)  # Increase energy
            rat.food_found += 1  # Track food found

            # Add to rat's memory of food locations (ArbitrationLayer also tracks this)
            if not hasattr(rat, 'known_food_locations'):
                rat.known_food_locations = set()
            rat.known_food_locations.add(pos)

            rat.log_memory(f"🍎 FOOD CONSUMED at {pos}! Hunger: {rat.hunger:.1f}, Energy: {rat.energy:.1f}")
            print(f"🍎 FOOD CONSUMED at {pos}! Hunger: {rat.hunger:.1f}, Energy: {rat.energy:.1f}")
            return True
        return False

    def get_nearby_consumed_food_info(rat_pos, consumed_food_locations, radius=3):
        """Get information about consumed food locations near the rat"""
        x, y = rat_pos
        nearby_consumed = []

        for fx, fy in consumed_food_locations:
            distance = abs(x - fx) + abs(y - fy)  # Manhattan distance
            if distance <= radius:
                nearby_consumed.append({
                    'position': (fx, fy),
                    'distance': distance,
                    'direction': (fx - x, fy - y)
                })

        return nearby_consumed

    print("🧬 Starting Enhanced AGI Simulation with Integrated LLM Reasoning...")

    environments = load_environments(map_path)
    if not environments:
        print("❌ No environments loaded. Simulation cannot continue.")
        return

    rat = SeedRat(goal="Develop general intelligence through survival and exploration...")
    # The 'meta_agent' variable is unused in your original code.
    # If you have plans for it, integrate it. Otherwise, you can remove it.
    # meta_agent = MetaAgent() # Removed as it was unused and causing a warning

    # Instantiate the GoalArbitrationLayer
    arbitration_layer = GoalArbitrationLayer()

    # Initialize food tracking on rat
    rat.food_found = 0
    # Note: arbitration_layer manages its own `last_known_food_locations`
    # rat.known_food_locations is for the rat's internal memory, not directly used by arbitration_layer now
    # Ensure rat has a log_memory method, e.g.:
    # def log_memory(self, message): self.memory.append(message)


    example_state = rat.encode_state_enhanced(
        {"up": "Wall", "down": "Wall", "left": "Wall", "right": "Wall"},
        food_scent=0.0, danger_scent=0.0, burrow_nearby=False
    )
    jepa = JEPA(input_dim=example_state.shape[0])

    os.makedirs("models", exist_ok=True)
    if os.path.exists("models/jepa.pth"):
        jepa.load_state_dict(torch.load("models/jepa.pth"))
        print("✅ JEPA model loaded.")

    dashboard_log = []
    reflection_log = []

    print("\n🧠 Inherited Knowledge Base:")
    for rule in rat.mom_memory[:5]:
        print("•", rule)

    print(f"\n🎯 Goal: {rat.goal}")
    print(f"📋 Subgoals: {rat.subgoals}") # This should probably be controlled by arbitration_layer or LLM now

    # Variable to hold prediction error from the *previous* step for the current step's arbitration
    last_jepa_prediction_error = 0.5 # Default initial value (e.g., moderate uncertainty)


    for env_index, env in enumerate(environments):
        print(f"\n🌍 Environment #{env_index} - Size: {env.width}x{env.height}")

        screen = initialize_pygame(env.width, env.height)
        consumed_food_locations = set()  # Track consumed food for this environment

        # Count initial food in environment
        initial_food_count = sum(row.count('F') for row in env.grid)
        print(f"🍎 Initial food available: {initial_food_count}")

        # Find a valid starting position and burrow
        for _ in range(100): # Try up to 100 times to find a valid spot
            start_x = random.randint(1, env.width - 2)
            start_y = random.randint(1, env.height - 2)
            if env.grid[start_y][start_x] != "Wall":
                rat.pos = (start_x, start_y)
                break
        else: # If loop completes without finding a spot
            print("❌ Could not find a valid starting position for the rat.")
            pygame.quit()
            return

        # Set burrow position on the environment object (used by arbitration layer)
        burrow_found_in_map = False
        for y_b, row_b in enumerate(env.grid):
            for x_b, tile_b in enumerate(row_b):
                if tile_b == 'B':
                    env.burrow_pos = (x_b, y_b)
                    burrow_found_in_map = True
                    break
            if burrow_found_in_map:
                break
        if not burrow_found_in_map:
            # Default burrow for a placeholder if not in map
            env.burrow_pos = (env.width // 2, env.height // 2)
            print(f"⚠️ No 'B' (burrow) found in map, defaulting env.burrow_pos to {env.burrow_pos}")
        # Ensure rat also knows its burrow_pos (used in arbitration layer for comparison)
        rat.burrow_pos = env.burrow_pos

        print(f"🐭 Rat starting at position: {rat.pos}")
        print(f"🏠 Burrow position: {env.burrow_pos}")


        for step in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            print(f"\n--- Step {step + 1}/{steps} (Age: {rat.age}) ---")

            try:
                # --- SENSING (BEFORE ARBITRATION AND ACTION) ---
                # Get current sensor readings for the arbitration layer to use
                current_food_scent = env.get_scent_map("F", rat.pos) # Assuming env.get_scent_map is available
                # current_danger_scent = env.get_scent_map("D", rat.pos) # If you have danger scent

                # --- GOAL ARBITRATION ---
                # The arbitration layer decides the high-level goal and the immediate subplan step
                # based on current internal state (rat) and sensed environmental cues.
                current_high_level_goal, current_subplan_step = arbitration_layer.arbitrate_goal(
                    rat, env, current_food_scent, last_jepa_prediction_error # Pass the error from previous step
                )

                # Update rat's perceived goal (optional, but good for internal consistency)
                rat.goal = current_high_level_goal
                print(f"➡️ High-level Goal (Arbitration Layer): {current_high_level_goal}")
                print(f"👣 Current Subplan Step: {current_subplan_step}")

                # --- LLM-DRIVEN SIMULATION STEP (ACTION SELECTION & EXECUTION) ---
                # This is the single call to your core simulation logic that involves the LLM.
                temp_step_result = enhanced_simulation_step_with_monologue(
                    rat, env, jepa,
                    current_goal=current_high_level_goal,
                    current_subplan_step=current_subplan_step
                )

                # --- POST-ACTION UPDATES AND LOGGING ---
                # 1. Handle food consumption at the rat's *new* position after the action
                #    This needs to happen after the rat moves in enhanced_simulation_step_with_monologue
                food_consumed_this_step = consume_food_at_position(env, rat.pos, consumed_food_locations, rat)

                # 2. Update nearby consumed food info based on the new position
                nearby_consumed_food = get_nearby_consumed_food_info(rat.pos, consumed_food_locations)

                # 3. Update the arbitration layer's internal map based on the *outcome* of this step
                #    `temp_step_result['position']` is the rat's new position.
                #    `temp_step_result['scents']['food']` is the food scent at the new position.
                arbitration_layer.update_tile_values(
                    temp_step_result['position'],
                    temp_step_result['scents']['food'],
                    found_food=food_consumed_this_step  # Pass True if food was just consumed
                )
                # Decay values for the next step's decision making
                arbitration_layer.decay_all_values()

                # 4. Store the JEPA prediction error from *this* step for the *next* arbitration cycle
                last_jepa_prediction_error = temp_step_result.get('prediction_error', 0.0)


                # --- PRINTING AND VISUALIZATION ---
                temp_step_result['food_consumed'] = food_consumed_this_step
                temp_step_result['total_food_found'] = rat.food_found
                temp_step_result['nearby_consumed_food'] = nearby_consumed_food
                temp_step_result['remaining_food'] = sum(row.count('F') for row in env.grid) # Recalculate after consumption

                print(f"📍 Position: {temp_step_result['position']}")
                print(f"💭 Emotions: H:{rat.hunger:.1f}, F:{rat.fear:.1f}, C:{rat.curiosity:.1f}, E:{rat.energy:.1f}")
                print(f"🧠 LLM Thought: {temp_step_result['llm_thought']}")
                print(f"🎯 LLM Prediction: {temp_step_result['llm_prediction']}")
                print(f"📊 Action Confidence: {temp_step_result['llm_confidence']:.1%}")
                print(f"📋 Action: {temp_step_result['action']} → Result: {temp_step_result['result']}")
                print(f"✅ Success: {temp_step_result['action_success']}")
                print(f"🔄 Strategy: {temp_step_result['strategy']}")
                print(f"🍎 Food Status: Found {rat.food_found}, Remaining {temp_step_result['remaining_food']}")

                if nearby_consumed_food:
                    print(f"📍 Nearby consumed food locations: {len(nearby_consumed_food)}")

                draw_environment(screen, env, rat.pos, consumed_food_locations)
                time.sleep(0.2)

                # --- DASHBOARD LOGGING ---
                dashboard_log.append({
                    'step': step + 1,
                    'environment': env_index,
                    'position': rat.pos,
                    'emotions': {
                        'hunger': rat.hunger,
                        'fear': rat.fear,
                        'curiosity': rat.curiosity,
                        'energy': rat.energy
                    },
                    'high_level_goal': current_high_level_goal,
                    'subplan_step': current_subplan_step,
                    'action': temp_step_result['action'],
                    'result': temp_step_result['result'],
                    'llm_thought': temp_step_result['llm_thought'],
                    'llm_prediction': temp_step_result['llm_prediction'],
                    'llm_confidence': temp_step_result['llm_confidence'],
                    'action_success': temp_step_result['action_success'],
                    'strategy': temp_step_result['strategy'],
                    'food_consumed': food_consumed_this_step,
                    'total_food_found': rat.food_found,
                    'remaining_food': temp_step_result['remaining_food'],
                    'consumed_food_locations': list(consumed_food_locations),
                    'nearby_consumed_food_count': len(nearby_consumed_food),
                    'tile_values_snapshot': arbitration_layer.tile_values.copy()
                })

                # You'll need to define how reflections are generated and what they contain
                # For example, calling a reflection function after N steps or specific events.
                # reflection_log.append(some_reflection_data)

            except Exception as e:
                print(f"❌ Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                # Consider adding a break here if an error should halt the current environment
                continue

        # Environment summary after loop for one environment
        final_food_count = sum(row.count('F') for row in env.grid)
        print(f"\n🍎 Environment #{env_index} Summary:")
        print(f"   • Food consumed: {initial_food_count - final_food_count}")
        print(f"   • Food remaining: {final_food_count}")
        print(f"   • Consumed food locations: {len(consumed_food_locations)}")
        print(f"   • Last known food locations (Arbitration Layer): {arbitration_layer.last_known_food_locations}")

        torch.save(jepa.state_dict(), "models/jepa.pth")
        print("💾 JEPA model saved.")

    pygame.quit() # Quit Pygame once all environments are simulated

    # Final logging and analysis (after all environments)
    try:
        os.makedirs("dashboard", exist_ok=True)

        with open("dashboard/enhanced_simulation_log.json", "w") as f:
            json.dump(dashboard_log, f, indent=2, default=str)

        with open("dashboard/reflection_log.jsonl", "w") as f:
            for entry in reflection_log:
                f.write(json.dumps(entry, default=str) + "\n")

        performance_analysis = analyze_simulation_performance(dashboard_log, reflection_log)

        total_food_consumed = sum(1 for log in dashboard_log if log.get('food_consumed', False))
        unique_food_locations = len(rat.known_food_locations) if hasattr(rat, 'known_food_locations') else 0

        performance_analysis['food_metrics'] = {
            'total_food_consumed': total_food_consumed,
            'unique_food_locations_found': unique_food_locations,
            'food_efficiency': total_food_consumed / len(dashboard_log) if dashboard_log else 0,
            'known_food_locations': list(arbitration_layer.last_known_food_locations)
        }

        with open("dashboard/performance_analysis.json", "w") as f:
            json.dump(performance_analysis, f, indent=2)

        print("✅ All logs and analysis saved.")
        print(f"📊 Performance Summary:")
        print(f"   • Total steps: {len(dashboard_log)}")
        print(f"   • Successful actions: {performance_analysis['successful_actions']}")
        print(f"   • LLM accuracy: {performance_analysis['llm_accuracy']:.1%}")
        print(f"   • Food consumed: {total_food_consumed}")
        print(f"   • Unique food locations: {unique_food_locations}")
        print(f"   • Food efficiency: {performance_analysis['food_metrics']['food_efficiency']:.2%}")
        print(f"   • Average confidence: {performance_analysis['avg_confidence']:.1%}")

    except Exception as e:
        print(f"⚠️ Error saving logs: {e}")
        import traceback
        traceback.print_exc()
    plot_learning_curve(rat)

# This function should be defined somewhere accessible to run_enhanced_simulation_with_integrated_monologue_with_gui
# For example, at the top of your run.py file, or in a separate utility file that you import.

def analyze_simulation_performance(dashboard_log, reflection_log):
    """
    Analyzes the simulation logs to provide performance metrics.

    Args:
        dashboard_log (list): List of dictionaries, each representing a simulation step.
        reflection_log (list): List of dictionaries, each representing a reflection entry.

    Returns:
        dict: A dictionary containing various performance metrics.
    """
    total_steps = len(dashboard_log)
    if total_steps == 0:
        return {
            'successful_actions': 0,
            'llm_accuracy': 0.0,
            'avg_confidence': 0.0,
            'total_food_consumed_from_log': 0 # Added for clarity
        }

    successful_actions_count = 0
    llm_correct_predictions = 0
    total_llm_predictions = 0
    total_llm_confidence = 0.0

    for entry in dashboard_log:
        if entry.get('action_success', False):
            successful_actions_count += 1

        # Assuming 'llm_prediction' is the predicted action and 'action' is the actual executed action
        if 'llm_prediction' in entry and 'action' in entry:
            total_llm_predictions += 1
            # You need a way to determine if the LLM's prediction was "correct"
            # This is often tricky and depends on your definition of correctness.
            # For a simple example, let's say it's correct if the predicted action
            # matches the actual action taken, and the action was successful.
            if entry['llm_prediction'] == entry['action'] and entry.get('action_success', False):
                llm_correct_predictions += 1
            # Or simply if the predicted action matches the executed action, regardless of success:
            # if entry['llm_prediction'] == entry['action']:
            #    llm_correct_predictions += 1

        if 'llm_confidence' in entry:
            total_llm_confidence += entry['llm_confidence']

    llm_accuracy = llm_correct_predictions / total_llm_predictions if total_llm_predictions > 0 else 0.0
    avg_confidence = total_llm_confidence / total_steps if total_steps > 0 else 0.0

    # Example of how to get food consumption from the dashboard_log itself
    total_food_consumed_from_log = sum(1 for log_entry in dashboard_log if log_entry.get('food_consumed', False))


    return {
        'total_steps': total_steps,
        'successful_actions': successful_actions_count,
        'llm_accuracy': llm_accuracy,
        'avg_confidence': avg_confidence,
        'total_food_consumed_from_log': total_food_consumed_from_log,
        # Add any other metrics you want to track
        # 'average_hunger': sum(e['emotions']['hunger'] for e in dashboard_log) / total_steps if total_steps > 0 else 0,
        # 'max_fear': max(e['emotions']['fear'] for e in dashboard_log) if dashboard_log else 0
    }

def create_monologue_action_summary(rat, step_results):
    """Create a summary of how LLM monologue influenced actions"""
    summary = {
        "total_llm_decisions": 0,
        "llm_overrides": 0,
        "correct_predictions": 0,
        "strategy_changes": 0,
        "confidence_trends": [],
        "reasoning_quality": {
            "detailed_analysis": 0,
            "strategic_thinking": 0,
            "environmental_awareness": 0,
            "goal_alignment": 0
        }
    }

    for result in step_results:
        if 'llm_thought' in result and result['llm_thought']:
            summary["total_llm_decisions"] += 1

            # Check for override indicators
            if "override" in result['llm_thought'].lower() or "change" in result['llm_thought'].lower():
                summary["llm_overrides"] += 1

            # Analyze reasoning quality
            thought = result['llm_thought'].lower()
            if len(thought) > 100:  # Detailed analysis
                summary["reasoning_quality"]["detailed_analysis"] += 1
            if any(word in thought for word in ["strategy", "plan", "goal", "objective"]):
                summary["reasoning_quality"]["strategic_thinking"] += 1
            if any(word in thought for word in ["environment", "surroundings", "scent", "wall"]):
                summary["reasoning_quality"]["environmental_awareness"] += 1
            if any(word in thought for word in ["survival", "food", "intelligence", "learning"]):
                summary["reasoning_quality"]["goal_alignment"] += 1

            # Track confidence
            if 'llm_confidence' in result:
                summary["confidence_trends"].append(result['llm_confidence'])

            # Check prediction accuracy
            if result.get('action_success') and result.get('llm_confidence', 0) > 0.6:
                summary["correct_predictions"] += 1

    return summary

# Additional utility functions for enhanced action-monologue integration

def validate_llm_response(response_text, available_actions):
    """Validate and clean LLM response to ensure proper action extraction"""
    lines = response_text.strip().split('\n')

    # Extract different components
    components = {
        'thought': '',
        'action': '',
        'prediction': '',
        'strategy': ''
    }

    for line in lines:
        line = line.strip()
        if line.startswith('Thought:'):
            components['thought'] = line.replace('Thought:', '').strip()
        elif line.startswith('Action:'):
            components['action'] = line.replace('Action:', '').strip()
        elif line.startswith('Prediction:'):
            components['prediction'] = line.replace('Prediction:', '').strip()
        elif line.startswith('Strategy:'):
            components['strategy'] = line.replace('Strategy:', '').strip()

    # Validate action
    if components['action'] not in available_actions:
        # Try to find a valid action in the text
        for action in available_actions:
            if action in response_text.lower():
                components['action'] = action
                break

    return components

def create_action_context_prompt(rat, env, surroundings):
    """Create rich context for LLM action decisions"""
    context = {
        "position": rat.pos,
        "surroundings": surroundings,
        "emotions": {
            "hunger": rat.hunger,
            "fear": rat.fear,
            "curiosity": rat.curiosity,
            "energy": rat.energy
        },
        "recent_memory": [mem for mem in rat.memory[-5:] if isinstance(mem, str)],
        "failed_actions": list(getattr(rat, 'failed_actions', set())),
        "current_strategy": getattr(rat, 'current_strategy', 'exploration'),
        "age": rat.age,
        "goal": rat.goal
    }

    # Add environmental analysis
    food_positions = []
    danger_positions = []

    for y in range(env.height):
        for x in range(env.width):
            if env.grid[y][x] == 'F':
                food_positions.append((x, y))
            elif env.grid[y][x] == 'N':
                danger_positions.append((x, y))

    context["environmental_intel"] = {
        "food_locations": food_positions,
        "danger_locations": danger_positions,
        "map_size": (env.width, env.height)
    }

    return context

if __name__ == "__main__":
    # Run the enhanced simulation with integrated LLM monologue
    run_enhanced_simulation_with_integrated_monologue_with_gui(steps=2000)

