import numpy as np
import pandas as pd

from gridpoint import GridPoint

'''
    TransitionNet class
    this class reads in a txt file with ids and coordinates of the points
    and creates a transition net with probabilities and absolutes
'''
class TransitionNet:

    def __init__(self, data, grid, resolution, frames_per_step, count_standing=False, count_huge_movement=False):
        
        self.grid = grid
        self.resolution = resolution

        #read in data 
        data = pd.read_csv('data.txt', sep=' ')
        data.drop(['ign', 'ign2', 'ign3', 'class'], axis=1, inplace=True)

        #get from grid_cell to index
        self.grid_to_index = {}
        index = 0
        for i in range(grid[0]):
            for j in range(grid[1]):
                self.grid_to_index[(i,j)] = index
                index += 1
        self.index_to_grid = {v: k for k, v in self.grid_to_index.items()}

        #create transition net with probailities and absolutes
        self.absolute_net = np.ones((grid[0] * grid[1], grid[0] * grid[1]))

        grouped = data.groupby('id')
        for index, data_per_id in grouped:
            data_per_id['Frame'] = data_per_id['Frame'] // frames_per_step
            data_per_id = data_per_id.groupby('Frame').mean().round(0)
            data_per_id['center_coordinates'] = list(zip(data_per_id.x + data_per_id.w / 2, data_per_id.y + data_per_id.h / 2))
            data_per_id['grid_point'] = data_per_id.center_coordinates.apply(lambda x: GridPoint(x[0], x[1], grid, resolution).get_grid_cell())
            data_per_id = data_per_id.reset_index(level=0)

            # iterate through consecutive columns and check if the "Frame" is consecutive. If so, add 1 to the absolute net
            for index in data_per_id.index[:-1]:
                if data_per_id.loc[index, 'Frame'] + 1 == data_per_id.loc[index + 1, 'Frame']:
                    from_grid_point = data_per_id.loc[index, 'grid_point']
                    to_grid_point = data_per_id.loc[index + 1, 'grid_point']

                    # if the movement is too big, ignore it
                    if not count_huge_movement and (from_grid_point[0] - to_grid_point[0] > 1 or to_grid_point[0] - from_grid_point[0] > 1):
                        continue

                    self.absolute_net[self.grid_to_index[from_grid_point], self.grid_to_index[to_grid_point]] += 1


        if not count_standing:
            # remove standing transistions
            self.absolute_net = self.absolute_net - np.diag(np.diag(self.absolute_net))
            
        #normalize absolute transition net to probabilities
        self.probability_net = self.absolute_net / self.absolute_net.sum(axis=1, keepdims=True)

        print(self.absolute_net)
        print(self.probability_net)

    def transform_coordiante_to_point_datastructure(self, x, y):
        return GridPoint(x, y, self.grid, self.resolution)

    def sample_transition(self, point):
        point_index = self.grid_to_index[point.get_grid_cell()]
        point_vector = np.zeros(self.absolute_net.shape[0])
        point_vector[point_index] = 1
        transition_vector = np.dot(point_vector, self.probability_net)
        transition_index = np.random.choice(range(len(transition_vector)), p=transition_vector)
        return GridPoint.from_grid_cell(self.index_to_grid[transition_index], self.grid, self.resolution)
    
    def sample_path(self, point, length):
        path = []
        for i in range(length):
            path.append(point)
            point = self.sample_transition(point)
        return path


if __name__ == '__main__':
    tn = TransitionNet('data.txt', (2,2), (960,720), 100)
    point = tn.transform_coordiante_to_point_datastructure(100,100)
    print(point.get_grid_cell())
    sampled_point = tn.sample_transition(point)
    print(sampled_point.get_grid_cell())
    print(sampled_point.get_grid_cell_middle_coordinates())