class TransitionNet:

    def __init__(self, data, grid, frames_per_step):
        ...
        #create transition net with probailities and absolutes

    def transform_coordiante_to_point_datastructure():
        ...
        return point

    def sample_transition(point): # takes previously transformed point from coordinate
        ...
        return transition

    def sample_path(start_point, steps):
        for i in steps:
            ...
        return [transition, transition, ...]



# usage of transition net in track_predict.py:
# net = TransitionNet(data, grid, frames_per_step)
# point = net.transform_coordiante_to_point_datastructure(x_on_picture,y_on_picture)
# path = net.sample_path(point, steps)

