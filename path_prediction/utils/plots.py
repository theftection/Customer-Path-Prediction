import cv2


def draw_predicted_path(im0, path, color, line_thickness):
    for i in range(len(path) - 1):
        org = path[i].get_original_coordinates()
        dest = path[i + 1].get_original_coordinates()
        cv2.arrowedLine(im0, org, dest, color, line_thickness)


def draw_transition_net(im0, transition_net, color):
    # draw the grid on the image
    for i in range(transition_net.shape[0]):
        ...

# Idea: make a 5x5 heatmap for every cell where a person is standing on
