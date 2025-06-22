import math
from collections import deque


class Environment:
    def __init__(self, grid):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])

        self.normalize_grid_tiles()

        self.food_positions = self._find_tile_positions('F')
        self.noise_positions = self._find_tile_positions('N')
        self.burrow_positions = self._find_tile_positions('B')
        self.door_positions = self._find_tile_positions('D') 

    def normalize_grid_tiles(self):
        """Normalize tile names: capitalize, fix 'door' → 'D' etc."""
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if isinstance(tile, str):
                    normalized = tile.strip().capitalize()
                    if normalized == "Door":
                        normalized = "D"
                    self.grid[y][x] = normalized

    def _find_tile_positions(self, tile_type):
        positions = []
        for r_idx, row in enumerate(self.grid):
            for c_idx, cell in enumerate(row):
                if cell == tile_type:
                    positions.append((c_idx, r_idx))
        return positions

    def get_tile_type(self, pos):
        x, y = pos
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.grid[y][x]
        return "Wall"

    def get_surroundings(self, pos):
        surroundings = {}
        directions = {
            "move_up": (0, -1),
            "move_down": (0, 1),
            "move_left": (-1, 0),
            "move_right": (1, 0),
        }
        for action, (dx, dy) in directions.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                surroundings[action] = self.grid[ny][nx]
            else:
                surroundings[action] = "Wall"
        return surroundings



    def update_grid(self, pos, new_tile_type):
        x, y = pos
        if 0 <= y < self.height and 0 <= x < self.width:
            normalized = new_tile_type.strip().capitalize()
            if normalized == "Door":
                normalized = "D"

            self.grid[y][x] = normalized

            if normalized == 'F':
                if pos not in self.food_positions:
                    self.food_positions.append(pos)
            elif pos in self.food_positions and normalized != 'F':
                self.food_positions.remove(pos)

            if normalized == 'N':
                if pos not in self.noise_positions:
                    self.noise_positions.append(pos)
            elif pos in self.noise_positions and normalized != 'N':
                self.noise_positions.remove(pos)

            if normalized == 'B':
                if pos not in self.burrow_positions:
                    self.burrow_positions.append(pos)
            elif pos in self.burrow_positions and normalized != 'B':
                self.burrow_positions.remove(pos)

            if normalized == 'D':
                if pos not in self.door_positions:
                    self.door_positions.append(pos)
            elif pos in self.door_positions and normalized != 'D':
                self.door_positions.remove(pos)
        else:
            print(f"⚠️ Warning: Tried to update out-of-bounds tile at {pos}")

    def get_scent_map(self, target_tile, start_pos):
        scent_map = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        visited = set()
        queue = deque()

        # Collect all sources of scent (e.g., all 'F' tiles)
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == target_tile:
                    queue.append(((x, y), 1.0))  # start with full scent strength

        walkable_tiles = {".", "F", "B", "D", "|"}  # Add doors and passages

        while queue:
            (x, y), strength = queue.popleft()
            if (x, y) in visited or strength < 0.01:
                continue
            visited.add((x, y))

            scent_map[y][x] = max(scent_map[y][x], strength)

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if (
                        0 <= nx < self.width and 0 <= ny < self.height and
                        self.grid[ny][nx] in walkable_tiles and
                        (nx, ny) not in visited
                ):
                    queue.append(((nx, ny), strength * 0.85))  # decay factor

        # Return the scent strength at the requested position
        return round(min(scent_map[start_pos[1]][start_pos[0]], 1.0), 3)



    @staticmethod
    def from_grid(grid):
        env = Environment(grid)
        env.normalize_grid_tiles()
        return env
