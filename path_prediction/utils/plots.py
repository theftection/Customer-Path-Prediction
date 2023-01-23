import cv2

def draw_predicted_path(im0, path, color, line_thickness):
    color = (0, 255, 0)
    thickness = line_thickness
    for i in range(len(path)-1):
        cv2.arrowedLine(im0, path[i], path[i+1],
                color, thickness)