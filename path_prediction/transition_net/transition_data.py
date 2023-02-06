from typing import Dict, Any

import pandas
import pandas as pd
from pandas import DataFrame
from enum import Enum
from grid_point import GridPoint


def data_aggregate_framerate(df, frame_rate):
    df['Frame'] = df['Frame'] // frame_rate
    df = df.groupby('Frame').mean().round(0)
    return df


class BoundingBoxAdjustment(Enum):
    CENTER = "CENTER"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_CENTER = "BOTTOM_CENTER"
    TOP_LEFT = "TOP_LEFT"


# ToDo: Prevent cam 0 from being used
# ToDo: Add quick-access for unique cam-ids
class TransitionData:
    raw_data: DataFrame
    highest_id: int
    unique_cam_ids: {}

    def __init__(self) -> None:
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.highest_id = -1
        self.unique_cam_ids = {}

    def add_txt_data(self, path_to_data: str, source_resolution: (int, int) = (960, 720),
                     invert_x: bool = False, invert_y: bool = False,
                     bounding_box_adjustment: BoundingBoxAdjustment = BoundingBoxAdjustment.CENTER,
                     cam_id: int = 1, frame_rate: int = 24) -> None:

        # if cam id is 0 throw an exception
        if cam_id == 0:
            raise ValueError("Cam id 0 is not allowed")

        # check if it already exists in unique_cam_ids
        if cam_id not in self.unique_cam_ids.keys():
            self.unique_cam_ids[cam_id] = len(self.unique_cam_ids.keys()) + 1

        # adjust cam_id to be increasing
        cam_id = self.unique_cam_ids[cam_id]

        new_data: DataFrame = pd.read_csv(path_to_data, sep=' ', index_col=False)

        # invert data if necessary
        if invert_x:
            new_data['x'] = source_resolution[0] - new_data['x']
        if invert_y:
            new_data['y'] = source_resolution[1] - new_data['y']

        # adjust bounding box
        # (x,y) is the top left corner of the bounding box
        # (x+w,y+h) is the bottom right corner of the bounding box
        # x increases to the right
        # y increases to the bottom
        if bounding_box_adjustment == BoundingBoxAdjustment.TOP_LEFT:
            pass
        elif bounding_box_adjustment == BoundingBoxAdjustment.BOTTOM_LEFT:
            new_data['y'] = new_data['y'] + new_data['h']
        elif bounding_box_adjustment == BoundingBoxAdjustment.BOTTOM_CENTER:
            new_data['y'] = new_data['y'] + new_data['h']
            new_data['x'] = new_data['x'] + new_data['w'] / 2
        elif bounding_box_adjustment == BoundingBoxAdjustment.CENTER:
            new_data['y'] = new_data['y'] + new_data['h'] / 2
            new_data['x'] = new_data['x'] + new_data['w'] / 2

        # aggregate frame rate per instance id
        grouped = new_data.groupby(['id'])
        grouped = grouped.apply(data_aggregate_framerate, frame_rate=frame_rate)
        new_data = grouped.reset_index(level=1)

        # convert to relative coordinates
        new_data['x'] = new_data['x'] / source_resolution[0]
        new_data['y'] = new_data['y'] / source_resolution[1]

        # drop unnecessary columns and rename columns
        new_data.drop(['ign', 'ign2', 'ign3', 'class', 'w', 'h'], axis=1, inplace=True)
        new_data.rename(columns={'Frame': 'frame_id', 'id': 'instance_id', 'x': 'x_rel', 'y': 'y_rel'}, inplace=True)

        # add source_id column
        new_data['cam_id'] = cam_id

        # correct instance id if necessary
        lowest_id = new_data['instance_id'].min()
        if lowest_id <= self.highest_id:
            new_data['instance_id'] = new_data['instance_id'] + self.highest_id

        # add it to the raw data
        self.raw_data = pandas.concat([self.raw_data, new_data], ignore_index=True)
        self.highest_id = self.raw_data['instance_id'].max()

    def get_unique_instance_ids(self) -> list:
        return self.raw_data['instance_id'].unique().tolist()

    def get_instance_grid_points(self, instance_id: int) -> dict[int, GridPoint]:
        """
        Returns the grid points of a specific instance
        :param instance_id: the id of the instance
        :return: a dictionary with the frame id as key and the grid point as value
        """
        instance_data = self.raw_data[self.raw_data['instance_id'] == instance_id]
        instance_data = instance_data.sort_values(by=['frame_id'])
        
        grid_point_dict: dict[int, GridPoint] = {}
        
        # loop through all rows
        for i, row in instance_data.iterrows():

            current_frame_id: int = int(row['frame_id'])

            new_grid_point = GridPoint(rel_x=row['x_rel'],
                                       rel_y=row['y_rel'],
                                       cam_id=row['cam_id'])
            grid_point_dict[current_frame_id] = new_grid_point

        return grid_point_dict


td = TransitionData()
td.add_txt_data(path_to_data='inference_data/transitions/Ch4_cam11_1.txt')
