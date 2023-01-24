import cv2

def draw_predicted_path(im0, path, color, line_thickness):
    for i in range(len(path)-1):
        org = path[i].get_original_coordinates()
        dest = path[i+1].get_original_coordinates()
        cv2.arrowedLine(im0, org, dest, color, line_thickness)