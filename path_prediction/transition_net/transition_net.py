import math
from typing import Dict

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

        self.dim_x = grid_dimensions[0]
        self.dim_y = grid_dimensions[1]
        self.dim_cam = len(transition_data.unique_cam_ids)
        self.state_length = state_length
        self.state_scaler = state_scaler
        self.state_to_index = {}
        self.index_to_state = {}
        self.transition_data = transition_data
        self.state_grid_size = {}

        # create state_to_index dictionary
        grid_index: int = 0
        state_scaler_omega = 1
        for i_state in range(self.state_length):

            # First state should have original dimensions, only consecutive states should be scaled
            if i_state > 0:
                state_scaler_omega *= state_scaler

            self.state_grid_size[i_state] = (math.ceil(self.dim_x * state_scaler_omega),
                                             math.ceil(self.dim_y * state_scaler_omega))

            for i_cam in range(self.dim_cam):
                for i_grid_x in range(self.state_grid_size[i_state][0]):
                    for i_grid_y in range(self.state_grid_size[i_state][1]):

                        # if i_cam == 0 and i_grid_x > 0 or i_grid_y > 0 skip
                        if i_cam == 0 and (i_grid_x > 0 or i_grid_y > 0):
                            continue

                        self.state_to_index[(i_state, i_cam, i_grid_x, i_grid_y)] = grid_index
                        grid_index += 1

        # invert state_to_index into index_to_state
        self.index_to_state = {v: k for k, v in self.state_to_index.items()}

    def convert_data_to_index_pair(self) -> (int, int):

        for instance_id in self.transition_data.get_unique_instance_ids():

            # get all grid points for this instance
            grid_points: dict[int, GridPoint] = self.transition_data.get_instance_grid_points(instance_id=instance_id)

            # iterate over each key in the dictionary
            for key in grid_points.keys():

                for offset in range(0, self.state_length):
                    pass





        pass


td = TransitionData()
td.add_txt_data(path_to_data='inference_data/transitions/Ch4_cam11_1.txt',
                cam_id=11)

td.add_txt_data(path_to_data='inference_data/transitions/Ch4_cam11_1.txt',
                cam_id=2)

t = td.get_instance_grid_points(instance_id=1)
t2 = t[0].reassign_grid(grid_dim=(10, 10))

tn = TransitionNet(transition_data=td,
                   grid_dimensions=(40, 40),
                   state_length=20,
                   state_scaler=0.8)
