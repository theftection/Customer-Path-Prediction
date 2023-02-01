import math


class GridPoint:
    rel_x: float
    rel_y: float
    cam_id: int

    grid_x: int
    grid_y: int
    grid_dim: (int, int)

    is_blank: bool

    def __init__(self, rel_x: float = 0, rel_y: float = 0,
                 grid_dim: (int, int) = (10, 10), cam_id: int = 0,
                 is_blank: bool = False):

        self.is_blank = is_blank
        self.grid_dim = grid_dim

        # enable blank positions
        if is_blank:
            self.rel_x = 0
            self.rel_y = 0
            self.cam_id = 0
            self.grid_x = 0
            self.grid_y = 0

        else:
            self.rel_x = rel_x
            self.rel_y = rel_y
            self.cam_id = cam_id
            self.grid_x = int(rel_x * grid_dim[0])
            self.grid_y = int(rel_y * grid_dim[1])

    def __eq__(self, other):
        # if either position is blank they are equal
        if self.is_blank or other.is_blank:
            return True

        return (self.grid_x, self.grid_y, self.cam_id) == (other.grid_x, other.grid_y, other.cam_id)

    def convert_to_pixel_position(self, image_size: (int, int), return_center: bool = True) -> (int, int):
        """
        Converts the relative position to an absolute pixel position.
        :param return_center:
        :param image_size: The dimensions of the image in pixels.
        :return: The absolute pixel position of the grid points.
        """

        grid_width = image_size[0] / self.grid_dim[0]
        grid_height = image_size[1] / self.grid_dim[1]

        if return_center:
            return int(self.grid_x * grid_width + grid_width / 2), int(self.grid_y * grid_height + grid_height / 2)

        return int(self.grid_x * grid_width), int(self.grid_y * grid_height)

    def get_distance_to_position(self, other, absolute_precision: bool = False) -> float:
        """
        Gets the distance between this position and another position.
        :param absolute_precision: If true, the distance will be calculated using the absolute pixel position.
        :param other: The other position.
        :return: The distance between the two positions.
        """
        # if either position is blank return 0
        if self.is_blank or other.is_blank:
            return 0

        # if we want absolute precision, return the euclidean distance
        if absolute_precision:
            x1 = self.rel_x
            y1 = self.rel_y
            x2 = other.rel_x
            y2 = other.rel_y
        else:  # otherwise return the grid distance
            x1 = self.grid_x / self.grid_dim[0]
            y1 = self.grid_y / self.grid_dim[1]
            x2 = other.grid_x / other.grid_dim[0]
            y2 = other.grid_y / other.grid_dim[1]

        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
