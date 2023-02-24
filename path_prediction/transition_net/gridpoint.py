'''
This class is used to convert a point in the frame to a grid cell and vice versa.

Note: coordinates are from top, left corner and the grid starts at 0,0
'''
class GridPoint:

    def __init__(self, x, y, grid, resolution):
        self.x = int(x) # meaning col
        self.y = int(y) # meaning row
        self.grid = grid # e.g. (10, 10)
        self.resolution = resolution # e.g (960,720)
        self.grid_width = resolution[0] / grid[0]
        self.grid_height = resolution[1] / grid[1]

        #check if point is in frame
        if x < 0 or x > self.resolution[0] or y < 0 or y > self.resolution[1]:
            raise ValueError(f'Point {x,y} is not in frame')

        # calculate grid cell
        self.grid_cell = (int(x / self.grid_width), int(y / self.grid_height))

        # calculate grid cell coordinates
        self.grid_cell_middle_coordinates = (self.grid_cell[0] * self.grid_width + self.grid_width / 2,
                                        self.grid_cell[1] * self.grid_height + self.grid_height / 2)

    @classmethod
    def from_grid_cell(cls, grid_cell, grid, resolution):
        grid_width = resolution[0] / grid[0]
        grid_height = resolution[1] / grid[1]
        x = grid_cell[0] * grid_width + grid_width / 2
        y = grid_cell[1] * grid_height + grid_height / 2
        return cls(x, y, grid, resolution)

    def get_grid_cell(self):
        return self.grid_cell
    
    def get_original_coordinates(self):
        return (self.x, self.y)

    def get_grid_cell_middle_coordinates(self):
        return self.grid_cell_middle_coordinates

if __name__ == "__main__":
    grid = (30, 20)
    resolution = (960, 720)
    point = GridPoint(750, 340, grid, resolution)
    print(point.get_grid_cell())
    print(point.get_original_coordinates())
    print(point.get_grid_cell_middle_coordinates())
        
        
    
    
