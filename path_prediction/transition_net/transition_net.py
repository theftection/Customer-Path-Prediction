import math
from typing import Dict
import itertools

import numpy as np

from path_prediction.transition_net.grid_point import GridPoint
from transition_data import TransitionData


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

    # compute the cartesian product of (0, 1, 2, 3) and (0, 1, 2, 3)

    def compute_transitions(self) -> [int, int]:

        transitions: [int, int] = []
        for instance_id in self.transition_data.get_unique_instance_ids():

            print('####')
            print('#### instance_id: ', instance_id, '####')
            print('####')

            # get all grid points for this instance
            grid_points: dict[int, GridPoint] = self.transition_data.get_instance_grid_points(instance_id=instance_id)

            # iterate over each key in the dictionary (each known observation)
            for key in grid_points.keys():

                print('#> key: ', key)

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

                print('origin_list: ', origin_list, 'destination_list: ', destination_list)

                # if state is new, add to dictionary
                if tuple(origin_list) not in self.transitions_to_index.keys():
                    self.transitions_to_index[tuple(origin_list)] = len(self.transitions_to_index)

                if tuple(destination_list) not in self.transitions_to_index.keys():
                    self.transitions_to_index[tuple(destination_list)] = len(self.transitions_to_index)

                transitions.append((origin_list, destination_list))

        self.transitions = transitions
        self.index_to_transitions = {v: k for k, v in self.transitions_to_index.items()}
        return transitions

    def compute_absolute_net(self):
        absolute_net = np.zeros((len(self.transitions_to_index), len(self.transitions_to_index)))
        for transition in self.transitions:
            absolute_net[
                self.transitions_to_index[tuple(transition[0])],
                self.transitions_to_index[tuple(transition[1])]
            ] += 1

        # convert to int
        absolute_net = absolute_net.astype(int)
        return absolute_net


td = TransitionData()
td.add_txt_data(path_to_data='inference_data/transitions/Ch4_cam11_1.txt',
                cam_id=11)

t = td.get_instance_grid_points(instance_id=1)
t2 = t[0].reassign_grid(grid_dim=(10, 10))

tn = TransitionNet(transition_data=td,
                   grid_dimensions=(2, 2),
                   state_length=1,
                   state_scaler=1)

od = tn.compute_transitions()
an = tn.compute_absolute_net()


def product(ar_list):
    if not ar_list:
        yield ()
    else:
        for a in ar_list[0]:
            for prod in product(ar_list[1:]):
                print('a: ', a, 'prod: ', prod)
                yield (a,) + prod

# p = list(product(tn.test))
