import numpy as np
from gym.envs.toy_text import discrete
from collections import defaultdict
import time
import pickle
import os
from gym.envs.classic_control import rendering

# Constants
CELL_PIXEL_SIZE = 100
MARGIN = 10

# Function to get grid coordinate for the specified cell
def get_coords(row, col, loc='center'):
    xc = (col + 1.5) * CELL_PIXEL_SIZE
    yc = (row + 1.5) * CELL_PIXEL_SIZE
    if loc == 'center':
        return xc, yc
    elif loc == 'interior_corners':
        half_size = CELL_PIXEL_SIZE//2 - MARGIN
        xl, xr = xc - half_size, xc + half_size
        yt, yb = xc - half_size, xc + half_size
        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
    elif loc == 'interior_triangle':
        x1, y1 = xc, yc + CELL_PIXEL_SIZE//3
        x2, y2 = xc + CELL_PIXEL_SIZE//3, yc - CELL_PIXEL_SIZE//3
        x3, y3 = xc - CELL_PIXEL_SIZE//3, yc - CELL_PIXEL_SIZE//3
        return [(x1, y1), (x2, y2), (x3, y3)]

# Function to draw an object given coordinates of points
def draw_object(coords_list):
    if len(coords_list) == 1:  # -> circle
        obj = rendering.make_circle(int(0.45*CELL_PIXEL_SIZE))
        obj_transform = rendering.Transform()
        obj.add_attr(obj_transform)
        obj_transform.set_translation(*coords_list[0])
        obj.set_color(0.2, 0.2, 0.2)  # -> black
    elif len(coords_list) == 3:  # -> triangle
        obj = rendering.FilledPolygon(coords_list)
        obj.set_color(0.9, 0.6, 0.2)  # -> yellow
    elif len(coords_list) > 3:  # -> polygon
        obj = rendering.FilledPolygon(coords_list)
        obj.set_color(0.4, 0.4, 0.8)  # -> blue
    return obj


# Class for grid world environment
# row\column  0   1   2   3   4   5
#     0      24  25  26  27  28  29
#     1      18  19  20  21  22  23
#     2      12  13  14  15  16  17
#     3       6   7   8   9  10  11
#     4       0   1   2   3   4   5
class GridWorldEnv(discrete.DiscreteEnv): # inherit the environment base class 
    def __init__(self, num_rows=5, num_cols=6, delay_for_each_render=0.05):
        # Initialize parameters
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.delay_for_each_render_in_seconds = delay_for_each_render

        # Define actions
        move_row_up = lambda row, col: (max(row - 1, 0), col)
        move_row_down = lambda row, col: (min(row + 1, num_rows - 1), col)
        move_column_down = lambda row, col: (row, max(col - 1, 0))
        move_column_up = lambda row, col: (row, min(col + 1, num_cols - 1))
        self.action_defs = {0: move_row_up, 1: move_column_up,
                            2: move_row_down, 3: move_column_down}

        # Number of states/actions
        nS = num_cols * num_rows
        nA = len(self.action_defs)

        # Conversions between the grid number and the state
        self.grid2state_dict = {(s // num_cols, s % num_cols): s for s in range(nS)}
        self.state2grid_dict = {s: (s // num_cols, s % num_cols) for s in range(nS)}

        # Gold state
        gold_cell = (num_rows // 2, num_cols - 2)
        gold_state = self.grid2state_dict[gold_cell]

        # Trap states
        trap_cells = [((gold_cell[0] + 1), gold_cell[1]),
                      (gold_cell[0], gold_cell[1] - 1),
                      ((gold_cell[0] - 1), gold_cell[1])]
        trap_states = [self.grid2state_dict[(r, c)]
                       for (r, c) in trap_cells]


        # Terminal states = Gold state + Trap states
        self.terminal_states = [gold_state] + trap_states
        print('Terminal states: ', self.terminal_states)

        # Build the transition probability
        P = defaultdict(dict)
        for s in range(nS): # for each grid state
            row, col = self.state2grid_dict[s]
            P[s] = defaultdict(list)
            for a in range(nA): # for each action
                action = self.action_defs[a]
                next_s = self.grid2state_dict[action(row, col)]

                # Compute the reward
                if self.is_terminal(next_s):
                    r = (1.0 if next_s == self.terminal_states[0]
                         else -1.0)
                else:
                    r = 0.0

                # Mark if done
                if self.is_terminal(next_s):
                    done = True
                else:
                    done = False

                # Update the transition probability
                # P[state][action] = [(probability, next state, reword, done?), ...]
                P[s][a] = [(1.0, next_s, r, done)]

        # Initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0 # always start at the grid state 0

        # Call the initialization function for the parent class
        super(GridWorldEnv, self).__init__(nS, nA, P, isd)

        # Build the display
        self.viewer = None
        self._build_display(gold_cell, trap_cells)

    def is_terminal(self, state):
        return state in self.terminal_states

    def _build_display(self, gold_cell, trap_cells):
        # Initialization
        all_drawing_objects = []

        # Compute the screen pixel size
        screen_width = (self.num_cols + 2) * CELL_PIXEL_SIZE
        screen_height = (self.num_rows + 2) * CELL_PIXEL_SIZE

        # Instantiate the viewer class
        self.viewer = rendering.Viewer(screen_width, screen_height)

        # Add border
        border_point_list = [
            (CELL_PIXEL_SIZE - MARGIN, CELL_PIXEL_SIZE - MARGIN),
            (screen_width - CELL_PIXEL_SIZE + MARGIN, CELL_PIXEL_SIZE - MARGIN),
            (screen_width - CELL_PIXEL_SIZE + MARGIN, screen_height - CELL_PIXEL_SIZE + MARGIN),
            (CELL_PIXEL_SIZE - MARGIN, screen_height - CELL_PIXEL_SIZE + MARGIN)
        ]
        border = rendering.PolyLine(border_point_list, True)
        border.set_linewidth(5)        
        all_drawing_objects.append(border)

        # Add vertical lines
        for col in range(self.num_cols + 1):
            x1, y1 = (col + 1) * CELL_PIXEL_SIZE, CELL_PIXEL_SIZE
            x2, y2 = (col + 1) * CELL_PIXEL_SIZE, \
                     (self.num_rows + 1) * CELL_PIXEL_SIZE
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_drawing_objects.append(line)

        # Add horizontal lines
        for row in range(self.num_rows + 1):
            x1, y1 = CELL_PIXEL_SIZE, (row + 1) * CELL_PIXEL_SIZE
            x2, y2 = (self.num_cols + 1) * CELL_PIXEL_SIZE, \
                     (row + 1) * CELL_PIXEL_SIZE
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_drawing_objects.append(line)

        # Add traps --> circles
        for cell in trap_cells:
            trap_coords = get_coords(*cell, loc='center')
            all_drawing_objects.append(draw_object([trap_coords]))

        # Add gold --> triangle
        gold_coords = get_coords(*gold_cell, loc='interior_triangle')
        all_drawing_objects.append(draw_object(gold_coords))

        # Add agent --> square
        agent_coords = get_coords(0, 0, loc='interior_corners')
        agent = draw_object(agent_coords)
        self.agent_trans = rendering.Transform()
        agent.add_attr(self.agent_trans)
        all_drawing_objects.append(agent)

        # Draw objects
        for obj in all_drawing_objects:
            self.viewer.add_geom(obj)

    def render(self, mode='human', done=False):
        if done:
            sleep_time = 1
        else:
            sleep_time = self.delay_for_each_render_in_seconds
        x_coord = self.s % self.num_cols
        y_coord = self.s // self.num_cols
        x_coord = (x_coord + 0) * CELL_PIXEL_SIZE
        y_coord = (y_coord + 0) * CELL_PIXEL_SIZE
        self.agent_trans.set_translation(x_coord, y_coord)
        rend = self.viewer.render(
            return_rgb_array=(mode == 'rgb_array'))
        time.sleep(sleep_time)
        return rend

    # Function to close the display
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':
    # Build the 5 x 6 grid world environment
    env = GridWorldEnv(5, 6)

    # Reset the environment
    s = env.reset()
    # Display the environment
    env.render()
    while True:
        # Choose an action randomly
        action = np.random.choice(env.nA)
        current_state = env.s

        # Take the action
        res = env.step(action) # response: (new state, reward, done?, info)

        # Print summary of the action
        print('Current state: ', current_state, ', Action: ', action, ' -> ', '(new state, reward, done?, info) ', res)

        # Update the display
        env.render(done=res[2])

        # If finished, stop the loop
        if res[2]:
            break

    env.close()
