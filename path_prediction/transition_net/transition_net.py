import math
import time
from typing import Dict
import itertools

import numpy as np
from numpy import ndarray

from path_prediction.transition_net.grid_point import GridPoint
from path_prediction.transition_net.transition_data import TransitionData


class TransitionNet:
    dim_x: int
    dim_y: int
    dim_cam: int
    state_length: int
    state_scaler: float
    state_to_index: {}
    index_to_state: {}
    transition_data: TransitionData
    state_grid_size: dict[int, (int, int)]
    absolute_net: ndarray
    probability_net: ndarray

    def __init__(self,
                 transition_data: TransitionData,
                 grid_dimensions: (int, int),
                 state_length: int,
                 state_scaler: float):

        self.dim_x = grid_dimensions[0]  # How many grid points in x direction
        self.dim_y = grid_dimensions[1]  # How many grid points in y direction
        self.dim_cam = len(transition_data.unique_cam_ids)  # How many cameras
        self.state_length = state_length  # How many states
        self.state_scaler = state_scaler  # How much to scale the grid for each state
        self.state_to_index = {}  # Dictionary of single point to index
        self.index_to_state = {}  # Dictionary of index to single state
        self.transition_data = transition_data  # Transition data
        self.state_grid_size = {}  # Dictionary of state to grid size
        self.transitions: ([int], [int]) = []  # List of transitions
        self.transitions_to_index: dict[[int], int] = {}  # Dictionary of transitions to index
        self.index_to_transitions: dict[int, [int]] = {}  # Dictionary of index to transitions

        # create state_to_index dictionary
        grid_index: int = 0
        state_scaler_omega = 1
        # temp test
        self.test = []
        test = []
        for i_state in range(self.state_length):

            # First state should have original dimensions, only consecutive states should be scaled
            if i_state > 0:
                state_scaler_omega *= state_scaler

            self.state_grid_size[i_state] = (math.ceil(self.dim_x * state_scaler_omega),
                                             math.ceil(self.dim_y * state_scaler_omega))

            for i_cam in range(int(self.dim_cam + 1)):
                for i_grid_x in range(self.state_grid_size[i_state][0]):
                    for i_grid_y in range(self.state_grid_size[i_state][1]):

                        # if i_cam == 0 and i_grid_x > 0 or i_grid_y > 0 skip
                        if i_cam == 0 and (i_grid_x > 0 or i_grid_y > 0):
                            continue

                        self.state_to_index[(i_state, i_cam, i_grid_x, i_grid_y)] = grid_index
                        test.append(grid_index)
                        grid_index += 1

            self.test.append(test)
            test = []

        # invert state_to_index into index_to_state
        self.index_to_state = {v: k for k, v in self.state_to_index.items()}

        # compute transitions
        self.compute_transitions()

        # compute absolute net
        self.compute_absolute_net()

        # compute relative net
        self.compute_probability_net()

    def compute_transitions(self) -> [int, int]:

        transitions: [int, int] = []
        for instance_id in self.transition_data.get_unique_instance_ids():

            # get all grid points for this instance
            grid_points: dict[int, GridPoint] = self.transition_data.get_instance_grid_points(instance_id=instance_id)

            # iterate over each key in the dictionary (each known observation)
            for key in grid_points.keys():

                key_origin = key

                origin_list: [int] = []
                destination_list: [int] = []

                for offset in range(0, self.state_length):

                    adjusted_key_origin = key_origin - (self.state_length - 1) + offset
                    adjusted_key_destination = adjusted_key_origin + 1

                    # check if the adjusted key is in the dictionary
                    if adjusted_key_origin in grid_points.keys():
                        origin_point = grid_points[adjusted_key_origin]
                    else:
                        origin_point = GridPoint(is_blank=True)
                    origin_point.reassign_grid(grid_dim=self.state_grid_size[offset])

                    origin_index = self.state_to_index[(offset,
                                                        origin_point.cam_id,
                                                        origin_point.grid_x,
                                                        origin_point.grid_y)]

                    origin_list.append(origin_index)

                    if adjusted_key_destination in grid_points.keys():
                        destination_point = grid_points[adjusted_key_destination]
                    else:
                        destination_point = GridPoint(is_blank=True)
                    destination_point.reassign_grid(grid_dim=self.state_grid_size[offset])

                    destination_index = self.state_to_index[(offset,
                                                             destination_point.cam_id,
                                                             destination_point.grid_x,
                                                             destination_point.grid_y)]

                    destination_list.append(destination_index)

                # if state is new, add to dictionary
                if tuple(origin_list) not in self.transitions_to_index.keys():
                    self.transitions_to_index[tuple(origin_list)] = len(self.transitions_to_index)

                if tuple(destination_list) not in self.transitions_to_index.keys():
                    self.transitions_to_index[tuple(destination_list)] = len(self.transitions_to_index)

                transitions.append((origin_list, destination_list))

        self.transitions = transitions
        self.index_to_transitions = {v: k for k, v in self.transitions_to_index.items()}
        return transitions

    def compute_absolute_net(self) -> np.ndarray:
        absolute_net = np.zeros((len(self.transitions_to_index), len(self.transitions_to_index)))
        for transition in self.transitions:
            absolute_net[
                self.transitions_to_index[tuple(transition[0])],
                self.transitions_to_index[tuple(transition[1])]
            ] += 1

        # convert to int
        absolute_net = absolute_net.astype(int)
        self.absolute_net = absolute_net
        return absolute_net

    def compute_probability_net(self) -> np.ndarray:
        self.probability_net = self.absolute_net / (self.absolute_net.sum(axis=1, keepdims=True) + 0.0000001)
        return self.probability_net

    def predict_transition(self, starting_vector: [int], round_to_int: bool = True):

        destination_vector = np.dot(starting_vector, self.probability_net)

        if round_to_int:
            destination_vector = np.rint(destination_vector)

        return destination_vector

    def predict_n_steps(self, starting_index: int, n_steps: int, n_samples: int = 1,
                        prune_to_top=False, prune_to_top_n=3) -> (np.ndarray, [np.ndarray]):

        starting_vector = np.zeros(len(self.probability_net[0]))
        starting_vector[starting_index] = n_samples

        destination_vector = starting_vector
        intermediate_vectors = [starting_vector]
        for _ in range(n_steps):
            destination_vector = self.predict_transition(starting_vector=destination_vector)

            if prune_to_top:
                # keep only the top n values and set all others to 0
                top_n = np.argpartition(destination_vector, -prune_to_top_n)[-prune_to_top_n:]
                destination_vector = np.where(np.isin(np.arange(len(destination_vector)), top_n), destination_vector, 0)

            intermediate_vectors.append(destination_vector)
        return destination_vector, intermediate_vectors

    def create_starting_index(self, path: [float, float, int]) -> int:
        """
        Create a starting index for the path
        :param path: a list of waypoints (x, y, cam_id)
        :return: an index
        """

        transition = []
        path_length = len(path)

        for distance_from_present in range(self.state_length):
            waypoint = path[path_length - distance_from_present - 1]
            transition.append(self.convert_waypoint_to_state(waypoint=waypoint,
                                                             distance_from_present=distance_from_present))

        # convert transition to index (if not in dictionary, print warning)
        if tuple(transition) not in self.transitions_to_index.keys():
            print('Warning: transition not known')
            return 0

        return self.transitions_to_index[tuple(transition)]

    def convert_waypoint_to_state(self, waypoint: [float, float, int], distance_from_present) -> [int]:
        x, y, cam_id = waypoint

        grid_x_size, grid_y_size = self.state_grid_size[distance_from_present]
        grid_x, grid_y = int(x * grid_x_size), int(y * grid_y_size)

        return self.state_to_index[(distance_from_present, cam_id, grid_x, grid_y)]


if __name__ == '__main__':
    td = TransitionData()
    td.load_floorplan_folder(folder_path='inference_data/transitions/',
                             source_resolution=(960, 720),
                             cam_id=4)

    tn = TransitionNet(transition_data=td,
                       grid_dimensions=(4, 4),
                       state_length=2,
                       state_scaler=1)

    starting_index = tn.create_starting_index(path=[(0, 0.5, 1), (0, 0.51, 1)])

    start_time = time.time()
    d, i = tn.predict_n_steps(starting_index=starting_index, n_steps=5, n_samples=1000000,
                              prune_to_top=True, prune_to_top_n=2)
    end_time = time.time()
    # print time in mikro seconds
    print((end_time - start_time) * 1000000)
