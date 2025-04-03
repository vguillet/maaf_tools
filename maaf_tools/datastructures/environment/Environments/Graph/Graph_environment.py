
################################################################################################################
"""
Environment based on graph structures.
"""

# Built-in/Generic Imports
from copy import deepcopy
import random
import os

# Libs
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from networkx import *

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.best_first import BestFirst

# Own modules
from MAF.MAF_environment import MAF_environment
from Environments.Tools.tile_type import *
from Environments.Tools.action import Action
from Environments.Graph.Map_generator import *
from Tools.Animation_tools import *

from Environments.Tools.Grid_tools import convert_coordinates

# from src.Visualiser.Visualiser import Visualiser
# from src.Visualiser.Visualiser_tools import Visualiser_tools
# from src.Visualiser.Visualiser_tools import Visualiser_tools

# from MAF.to_delete.CAF_environment import CAF_environment

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '31/01/2020'

################################################################################################################

SUPPORTED_ACTIONS = [
    Action.EMPTY,
    Action.WAIT,
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
]

# Environment layout
RANDOM = 0
GRID = 1
STAR = 2

ENVIRONMENT_LAYOUTS = [
    RANDOM,
    GRID,
    STAR
]

ENVIRONMENT_LAYOUTS_NAMES = [
    "Random",
    "Grid",
    "Star"
]


class Environment(MAF_environment):
    def __init__(self,
                 namespace: str,         # Name of the environment instance
                 layout: int = GRID,
                 nodes_count: int = 20,
                 connectivity_percent: float = 0.8,  # Ignored for star
                 branch_count: int = 8,  # Only relevant for the star layout
                 start_node=None,
                 ):
        """
        Graph environment class, used to generate RS environments
        """

        environment_type = "Graph"

        # ---- Store all inputs (! Important for instance cloning)
        self.namespace = namespace
        self.layout = layout
        self.nodes_count = nodes_count
        self.connectivity_percent = connectivity_percent
        self.branch_count = branch_count
        self.start_node = start_node

        # -> Initialise MAF environment
        super().__init__(
            namespace=namespace,
            environment_type=environment_type,
            map_reference=f"{ENVIRONMENT_LAYOUTS_NAMES[layout]}_{nodes_count}_{connectivity_percent}_{branch_count}"
        )

        # -> Initialise environment graph
        if layout == RANDOM:
            self.environment_graph, self.nodes_pos = generate_random_layout(
                num_nodes=nodes_count,
                connectivity_percent=connectivity_percent
            )

        elif layout == GRID:
            self.environment_graph, self.nodes_pos = generate_grid_layout(
                num_nodes=nodes_count,
                connectivity_percent=connectivity_percent
            )

        elif layout == STAR:
            self.environment_graph, self.nodes_pos = generate_star_layout(
                num_nodes=nodes_count,
                num_branches=branch_count
            )

        else:
            raise f"ERROR: Invalid layout setting: {layout}"

        # ----- Setup Agent tracking
        # Agent start loc
        if start_node is not None:
            # -> Get start state in sim reference frame
            self.start_state = start_node

        else:
            # -> Get random start state
            self.start_state = random.choice(self.valid_start_positions)

        # Agent goal
        # if goal_world is not None or goal_sim is not None:
        #     self.goal = convert_coordinates(
        #         simulation_origin=self.origin,
        #         simulation_shape=self.shape,
        #         world_pos=goal_world,
        #         simulation_pos=goal_sim
        #     )[-1]
        # else:
        #     # -> Get random goal
        #     self.goal = random.choice(self.valid_start_positions)

        # Agent state
        self.state = self.start_state

    # ===================================== Properties
    @property
    def valid_start_positions(self) -> list:
        """
        :return: List of valid start positions
        """
        return list(self.nodes_pos.values())

    # ===================================== Prints
    @staticmethod
    def _render_state(fig,
                      ax,
                      paths: list = [],
                      positions: list = [],
                      POIs: list = [],
                      show: bool = True,
                      flat: bool = True
                      ):

        """
        Render the current state of the environment

        :param fig:
        :param ax:
        :param paths: List of paths to render
        :param positions: List of positions to render
        :param POIs: List of POIs to render
        :param show: Show the plot
        :param flat: Render in 2D or 3D
        """

        # -> Create matplotlib plot
        # ----- 2D plot
        if flat is True:
            # > Colors
            positions_colors = cm.rainbow(np.linspace(0, 1, len(positions)))

            # > Paths
            if paths:
                for i, path in enumerate(paths):
                    ax.plot(*zip(*path), color=positions_colors[i])

            # > POIs
            if POIs:
                ax.scatter(*zip(*POIs), color="orange", s=15, marker="D")

            # > Positions
            if positions:
                ax.scatter(*zip(*positions), color=positions_colors, s=30)

            # -> Display plot
            if show:
                plt.show()

        # ----- 3D plot
        else:
            # TODO: Implement 3D plot
            raise NotImplementedError

        return fig, ax

    def render_terrain(self,
                       paths: list = [],
                       positions: list = [],
                       POIs: list = [],
                       show: bool = True,
                       *args,
                       **kwargs
                       ):
        """
        Render the current terrain of the environment

        :param paths: List of paths to render
        :param positions: List of positions to render
        :param POIs: List of POIs to render
        :param show: Show the plot
        """

        # -> Set up the figure and axis
        fig, ax = plt.subplots()

        # -> Set min and max of axis
        min_x = 0
        max_x = 0

        min_y = 0
        max_y = 0

        for x, y in self.nodes_pos:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        buffer = 0.5

        ax.set_xlim(min_x - buffer, max_x + buffer)
        ax.set_ylim(min_y - buffer, max_y + buffer)

        ax.set_axis_off()

        # -> Lock the aspect ratio to 1:1
        ax.set_aspect('equal')

        # -> Get edge weights
        weights = nx.get_edge_attributes(self.environment_graph, 'weight')

        # -> Draw the graph as a background
        # > Draw nodes
        nx.draw_networkx(
            G=self.environment_graph,
            pos=self.nodes_pos,
            with_labels=False,
            ax=ax,
            # node_color="white",
            alpha=0.3)

        # Draw the node positions as text underneath the nodes
        offset = 0.2  # Offset for the text position
        for node, (x, y) in self.nodes_pos.items():
            plt.text(
                x,
                y - offset,
                f"({x},{y})",
                ha='center',
                fontsize=7,
                color="grey"
            )

        # > Draw edges
        nx.draw_networkx_edge_labels(
            G=self.environment_graph,
            pos=self.nodes_pos,
            edge_labels=weights,
            ax=ax)

        # ----- Create matplotlib plot
        fig, ax = self._render_state(
            fig=fig,
            ax=ax,
            paths=paths,
            positions=positions,
            POIs=POIs,
            show=show,
            flat=True
        )

        return fig, ax

    def render_comms(self,
                     paths: list = [],
                     positions: list = [],
                     POIs: list = [],
                     show: bool = True,
                     flat: bool = True,
                     *args,
                     **kwargs
                     ):
        # TODO: Implement
        pass

    def animate(self,
                background: str = "terrain",  # -> "terrain" or "comms"
                paths: list = [],
                plot_paths: bool = False,
                goals: list = [],
                POIs: list = [],
                duration: int = 200,
                save_path: str = None,
                *args,
                **kwargs
                ):

        interpolation_rate = 10

        print("-------------------------->>> Creating animation")

        # -> Create base env image
        if background == "terrain":
            fig, ax = self.render_terrain(show=False, flat=True)
        elif background == "comms":
            fig, ax = self.render_comms(show=False, flat=True)
        else:
            raise ValueError(f"Invalid background type: {background}, must be 'terrain' or 'comms'")

        # -> Lock the aspect ratio to 1:1
        ax.set_aspect('equal')

        # -> Create agent colors
        agent_colors, _ = self.gen_rainbow_color_list(length=len(paths))

        # -> Get max path length
        max_path_length = 0

        for path in paths:
            if len(path) > max_path_length:
                max_path_length = len(path)

        # > Create empty goals if none are given
        if not goals:
            for _ in range(len(paths)):
                goals.append([[None] for _ in range(max_path_length)])

        # ========================================================================== Construct frames
        # ----- Interpolate frames to create smooth animation
        frame_count = max_path_length * interpolation_rate

        interpolated_paths = []

        for i, path in enumerate(paths):
            # Extract x and y values from the original list
            x = [point[0] for point in path]
            y = [point[1] for point in path]

            # Calculate the indices for interpolation
            indices = np.linspace(0, len(path) - 1, frame_count)

            # Interpolate x and y values separately
            interp_x = interp1d(range(len(path)), x)(indices)
            interp_y = interp1d(range(len(path)), y)(indices)

            # Create the interpolated list of tuples
            interpolated_path = [(interp_x[i], interp_y[i]) for i in range(frame_count)]

            interpolated_paths.append(interpolated_path)

        # -> Set interpolated paths as the new paths
        paths = interpolated_paths

        # -> Pad goals and POIs lists
        # > Goals
        for i, goal in enumerate(goals):
            goals[i] = adjust_list_length(lst=goal, new_length=frame_count)

        # > POIs
        POIs = adjust_list_length(lst=POIs, new_length=frame_count)

        # ----- Create frames
        frames = []

        # -> Create frames, each containing all the data for the given frame
        for i in range(frame_count):
            frame = {
                "index": i,
                "paths": [],
                "goals": [],
                "POIs": POIs[i],
            }

            # > Add paths and goals
            for path in paths:
                frame["paths"].append(path[:i+1])

            for goal in goals:
                frame["goals"].append(goal[i])

            frames.append(frame)

        # ========================================================================== Create animation
        # -> Create agent plots
        agent_positions = []
        goal_rays = []

        # ... for every agent
        for agent in range(len(paths)):
            agent_positions.append(ax.scatter([], [], color=agent_colors[agent], marker="D", zorder=1))
            goal_rays.append(ax.plot([], [], color=agent_colors[agent], linestyle="--", zorder=1)[0])

        # -> Create POI plots
        # > Find max POI count at a single timestep
        max_POIs_count = 0

        for POIs_snapshot in POIs:
            if len(POIs_snapshot) > max_POIs_count:
                max_POIs_count = len(POIs_snapshot)

        tasks_positions = []

        # ... for every task
        for task in range(max_POIs_count):
            tasks_positions.append(
                ax.scatter(
                    [], [],
                    color="orange",
                    marker="x",
                    zorder=10)
            )

        # self.blink_toggle = False

        # -> Create step counter
        step_counter = ax.text(
            0.5,
            0.02,
            f"Frame: 1/{len(frames)/interpolation_rate}",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=10,
            color="black"
        )

        def update(frame):
            # -> Update step counter
            step_counter.set_text(f"Frame: {int(((frame['index']+1) - ((frame['index']+1) % interpolation_rate))/interpolation_rate)}/{int(len(frames)/interpolation_rate)}")

            # ... for every agent
            for agent in range(len(frame["paths"])):
                agent_positions[agent].set_offsets(frame["paths"][agent][-1])   # Update agent position

                # > Construct goal ray
                if frame["goals"][agent] is None:
                    goal_rays[agent].set_data([], [])                        # Clear goal ray
                else:
                    x = [frame["paths"][agent][-1][0], frame["goals"][agent][0]]
                    y = [frame["paths"][agent][-1][1], frame["goals"][agent][1]]

                    goal_rays[agent].set_data(x, y)                              # Update goal ray

            # -> Cleanup prev POIs
            for task in range(len(tasks_positions)):
                tasks_positions[task].set_alpha(0)

            # [x, y , ID]
            for POI in range(len(frame["POIs"])):
                if not frame["POIs"][POI]:
                    continue

                tasks_positions[POI].set_offsets(frame["POIs"][POI][:2])    # Update POI position
                tasks_positions[POI].set_alpha(1)                           # Set POI alpha

                # if frame["index"] % int(max_path_length/(duration/1000)) == 0:
                #     # > Toggle blink toggle
                #     self.blink_toggle = not self.blink_toggle
                #
                # if self.blink_toggle:
                #     tasks_positions[POI].set_edgecolor('white')
                # else:
                #     tasks_positions[POI].set_edgecolor('orange')

            return [agent_positions[i] for i in range(len(agent_positions))] \
                + [tasks_positions[i] for i in range(len(tasks_positions))] \
                + [goal_rays[i] for i in range(len(goal_rays))] \
                + [step_counter]

        # -> Create animation
        frame_count = len(frames)
        duration = duration * 1000      # Convert to ms
        playback_rate = 30

        anim = FuncAnimation(fig, update, frames=frames, interval=duration/frame_count, blit=True)

        # > Get root
        if save_path is None:
            root = str(os.getcwd()) + "/Results"

            # > Create folder if not exists
            if not os.path.exists(root):
                os.makedirs(root)

            # > Create save path
            save_path = f"{root}/{self.name}_{self.environment_type}_{self.map_reference}.gif"

        # -> Save
        anim.save(filename=save_path, writer='pillow')

        # -> Show the animation
        plt.show()

        return anim

    def render_path(self,
                    path,
                    positions: list = [],
                    show: bool = True
                    ):
        """
        Renders a given path
        """

        # TODO: Implement

    # ===================================== Interfaces
    def step(self, action):
        """
        Performs a step in the environment
        """
        raise NotImplementedError

    # def step(self, action):
    #     """
    #     Performs a step in the environment
    #     """
    #
    #     if self.state is None:
    #         raise "ERROR: Environment state was never set"
    #     if self.goal is None:
    #         raise "ERROR: Goal was never set"
    #
    #     assert (action in SUPPORTED_ACTIONS)
    #
    #     # ----- Perform action
    #     if action == Action.EMPTY:
    #         pass
    #     elif action == Action.UP:
    #         self.state[1] += 1
    #
    #     elif action == Action.DOWN:
    #         self.state[1] -= 1
    #
    #     elif action == Action.LEFT:
    #         self.state[0] -= 1
    #
    #     elif action == Action.RIGHT:
    #         self.state[0] += 1
    #
    #     # ----- Get step results
    #     # -> Get obs
    #     obs = self.state  # New state
    #
    #     # -> Get step reward
    #     reward = -1 if self.state == self.goal else 0  # Value of new tile
    #
    #     # -> Get done state
    #     # > Check terminal fail
    #     if "terminal_fail" in self.grids_dict:
    #         TERMINAL_FAIL = self.grids_dict["terminal_fail"][self.state] == 1
    #     else:
    #         TERMINAL_FAIL = 0
    #
    #     # > Check goal reached
    #     REACHED_GOAL = self.state == self.goal
    #
    #     # > Set done
    #     done = TERMINAL_FAIL * REACHED_GOAL  # Goal reached or dead
    #
    #     # -> Get step info
    #     infos = {}  # Info
    #
    #     return obs, reward, done, infos

    def compute_path_from_to(self, start: tuple, end: tuple, weighted: bool = False, *args, **kwargs):
        """
        Computes the shortest path from start to end using the environment graph
        """

        # -> Determine shortest path to goal from current loc
        if weighted:
            path = astar_path(
                G=self.environment_graph,
                source=tuple(start),
                target=tuple(end),
                weight='weight'
            )
        else:
            path = shortest_path(
                G=self.environment_graph,
                source=tuple(start),
                target=tuple(end)
            )

        return path

    # ------------------------------------------------------------ OLD

    # ===================================== Plots
    # def plot_environment(self):

    # def plot_state(self):
    #     pass
    #
    # def animate_sim_snapshots(self, frame_count: int, snapshots, task_schedule):
    #     # Animation config
    #     playback_rate = 0.1
    #
    #     # -------------------------------------- Prepare data
    #     # -> Reformat snapshots to correct lists
    #     # TODO
    #
    #     # -> Gather all agent-task pairings for each epoch
    #     task_consensus_timeline = []
    #
    #     for fleet_snapshot in snapshots["fleet_state"]:
    #         epoch_consensus_state = []
    #
    #         for agent in fleet_snapshot.agent_list:
    #             agent_beliefs = []
    #
    #             for task_id in agent.local_tasks_dict.keys():
    #                 if agent.local_tasks_dict[task_id]["status"] == "done":
    #                     continue
    #
    #                 winning_bid = agent._get_local_task_winning_bid(task_id=task_id)
    #                 local_winner_id = winning_bid["agent_id"]
    #
    #                 task_loc = agent.local_tasks_dict[task_id]["instructions"][-1][-1]
    #
    #                 # -> Store (winning agent, goto loc)
    #                 agent_beliefs.append({
    #                     "local_winner_id": local_winner_id,
    #                     "task_loc": task_loc})
    #
    #             epoch_consensus_state.append(agent_beliefs)
    #
    #         for _ in range(int(frame_count / len(snapshots["fleet_state"]))):
    #             task_consensus_timeline.append(epoch_consensus_state)
    #
    #             # -> Create the interpolator
    #     interpolator = interp1d(np.arange(len(snapshots["agents_pos"])),
    #                             snapshots["agents_pos"], axis=0)
    #
    #     # -> Generate new frames by interpolation
    #     new_frames = []
    #     for i in np.linspace(0, len(snapshots["agents_pos"]) - 1, frame_count):
    #         interpolated_frame = interpolator(i)
    #         new_frames.append(interpolated_frame, )
    #
    #     print(len(new_frames), len(task_consensus_timeline))
    #
    #     # print(new_frames)
    #     merged_frames = []
    #
    #     for i in range(frame_count):
    #         merged_frames.append({
    #             "agents_pos": new_frames[i],
    #             "agents_beliefs": task_consensus_timeline[i]})
    #
    #     new_frames = merged_frames
    #
    #     # -------------------------------------- Prepare plot
    #     # -> Set up the figure and axis
    #     fig, ax = plt.subplots()
    #
    #     # -> Set min and max of axis
    #     min_x = 0
    #     max_x = 0
    #
    #     min_y = 0
    #     max_y = 0
    #
    #     for x, y in self.nodes_pos:
    #         min_x = min(min_x, x)
    #         max_x = max(max_x, x)
    #         min_y = min(min_y, y)
    #         max_y = max(max_y, y)
    #
    #     buffer = 0.5
    #
    #     ax.set_xlim(min_x - buffer, max_x + buffer)
    #     ax.set_ylim(min_y - buffer, max_y + buffer)
    #
    #     ax.set_axis_off()
    #
    #     # -> Lock the aspect ratio to 1:1
    #     ax.set_aspect('equal')
    #
    #     # -> Get edge weights
    #     weights = nx.get_edge_attributes(self.environment_graph, 'weight')
    #
    #     # -> Draw the graph as a background
    #     nx.draw_networkx(
    #         G=self.environment_graph,
    #         pos=self.nodes_pos,
    #         with_labels=False,
    #         ax=ax,
    #         # node_color="white",
    #         alpha=0.3)
    #
    #     nx.draw_networkx_edge_labels(
    #         G=self.environment_graph,
    #         pos=self.nodes_pos,
    #         edge_labels=weights,
    #         ax=ax)
    #
    #     # -> Generate a color per agent
    #     agents_colors = cm.rainbow(np.linspace(0, 1, len(self.agents)))
    #
    #     # -> Create a scatter plot for each agent loc
    #     agent_positions = []
    #
    #     # ... for every agent
    #     for agent in range(len(self.get_agent_group(group_name="base_agent"))):
    #         agent_positions.append(ax.scatter([], [], color=agents_colors[agent], marker="D"))
    #
    #     # -> Create belief line plots
    #     beliefs_lines = []
    #
    #     # ... for every agent and every tasks
    #     for _ in range(self.agent_count * len(task_schedule)):
    #         beliefs_lines.append(ax.plot([], []))
    #
    #     # -> Create a scatter plot for each task loc
    #     tasks_positions = []
    #
    #     # ... for every tasks
    #     for task in range(len(task_schedule)):
    #         tasks_positions.append(ax.scatter([], [], color="green"))
    #
    #     # -> Define the animation function
    #     def update(frame):
    #         # ... for every agent
    #         for agent in range(len(frame["agents_pos"])):
    #             agent_positions[agent].set_offsets(frame["agents_pos"][agent][0:-1])  # Update agent position
    #
    #         # -> Cleanup prev lines
    #         for i in range(len(beliefs_lines)):
    #             beliefs_lines[i][0].set_data([], [])
    #
    #         # -> Cleanup prev tasks positions
    #         for task in range(len(tasks_positions)):
    #             tasks_positions[task].set_alpha(0)
    #
    #         offset_increment = 0.02
    #         offset_tracker = {}
    #
    #         # ... for every agent
    #         for agent in range(len(frame["agents_beliefs"])):
    #             # ... for every tasks
    #             for task in range(len(frame["agents_beliefs"][agent])):
    #                 # if frame["agents_beliefs"][agent][task]:
    #
    #                 # -> Set task position
    #                 tasks_positions[task].set_offsets((frame["agents_beliefs"][agent][task]["task_loc"][0],
    #                                                    frame["agents_beliefs"][agent][task]["task_loc"][1]))
    #                 tasks_positions[task].set_alpha(1)
    #
    #                 # -> Find current agent loc
    #                 for agent_i in range(len(frame["agents_pos"])):
    #                     if frame["agents_pos"][agent_i][-1] == frame["agents_beliefs"][agent][task]["local_winner_id"]:
    #                         line_x = [frame["agents_pos"][agent_i][0],
    #                                   frame["agents_beliefs"][agent][task]["task_loc"][0]]
    #                         line_y = [frame["agents_pos"][agent_i][1],
    #                                   frame["agents_beliefs"][agent][task]["task_loc"][1]]
    #
    #                         # -> If overlap, update and apply offset
    #                 if str(frame["agents_beliefs"][agent][task]) in offset_tracker.keys():
    #                     offset_tracker[str(frame["agents_beliefs"][agent][task])] += offset_increment
    #
    #                 else:
    #                     offset_tracker[str(frame["agents_beliefs"][agent][task])] = 0
    #
    #                 # -> Calculate the angle of the line segment
    #                 x0, y0 = line_x[0], line_y[0]
    #                 x1, y1 = line_x[-1], line_y[-1]
    #                 angle = math.atan2(y1 - y0, x1 - x0)
    #
    #                 # -> Ponderate offset
    #                 # angle_diff = abs(angle - math.pi/4)
    #                 # offset_scale = max(1 - angle_diff/(math.pi/4), 0)
    #                 # offset = offset_scale * offset_increment
    #                 # x_coordinates = [x + offset for x in frame["agents_beliefs"][agent][task][0]]
    #                 # y_coordinates = [y + offset for y in frame["agents_beliefs"][agent][task][1]]
    #
    #                 # -> Apply the offset to the x and y coordinates based on the angle
    #                 offset = offset_tracker[str(frame["agents_beliefs"][agent][task])]
    #                 x_offset = math.sin(angle) * offset
    #                 y_offset = math.cos(angle) * offset
    #                 x_coordinates = [x + x_offset for x in line_x]
    #                 y_coordinates = [y + y_offset for y in line_y]
    #
    #                 # Apply the offset to the x and y coordinates
    #                 # offset = offset_tracker[str(frame["agents_beliefs"][agent][task])]
    #                 # x_coordinates = [x + offset for x in frame["agents_beliefs"][agent][task][0]]
    #                 # y_coordinates = [y + offset for y in frame["agents_beliefs"][agent][task][1]]
    #
    #                 # -------------------------------------------------
    #                 beliefs_lines[agent * len(task_schedule) + task][0].set_data(x_coordinates,
    #                                                                                   y_coordinates)  # update x and y data of the plot object
    #                 beliefs_lines[agent * len(task_schedule) + task][0].set_color(
    #                     agents_colors[agent])  # set color
    #
    #         return [agent_positions[i] for i in range(len(agent_positions))] + [beliefs_lines[i][0] for i in
    #                                                                             range(len(beliefs_lines))] + [
    #                    tasks_positions[i] for i in range(len(tasks_positions))]
    #
    #     # -> Create the animation
    #     anim = FuncAnimation(fig, update, frames=new_frames, interval=(1000 / frame_count) / playback_rate, blit=True)
    #
    #     # -> Save
    #     anim.save(filename=f"Sim.gif", writer='pillow')
    #
    #     # -> Show the animation
    #     plt.show()
    #
    #     return anim
