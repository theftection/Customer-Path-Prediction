import cv2
import numpy as np
from pathlib import Path


def project_2D_to_3D(P_inv, camera_origin, point_2D, height):
    """Project 2D point to 3D space.
    Args:
        P_inv (np.ndarray): Inverse of projection matrix.
        camera_origin (np.ndarray): Camera origin in 3D space.
        point_2D (np.ndarray): 2D point to project.
        height (float): Height of projected point.
    Returns:
        np.ndarray: 3D points.
    """
    point_2D_h = np.concatenate((point_2D, np.ones((point_2D.shape[0],1))), axis=1)
    point_3D_h = P_inv @ point_2D_h.T
    point_3D = (point_3D_h / point_3D_h[3])[:3].T
    ray_3D = point_3D - camera_origin
    t = (height - camera_origin[2]) / ray_3D[:, 2]
    world_point = camera_origin + ray_3D * t
    return world_point.astype(int)

def top_center(bbox):
    """Get top center of bounding box.
    Args:
        bbox (list): Bounding box.
    Returns:
        np.ndarray: Top center of bounding box.
    """
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, y1])

def bottom_center(bbox):
    """Get bottom center of bounding box.
    Args:
        bbox (list): Bounding box.
    Returns:
        np.ndarray: Bottom center of bounding box.
    """
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, y2])

def check_in_zones(point, zones):
    for zone in zones:
        if point[0] > zone[0] and point[0] < zone[2] and point[1] > zone[1] and point[1] < zone[3]:
            return True, zone
    return False, None

def correct_for_redzone(standing_position, ray_3D, redzones):
        floor_position = standing_position[0]
        in_redzone, zone = check_in_zones(floor_position, redzones)
        if in_redzone:
            t_x = (zone[2] - floor_position[0]) / ray_3D[:, 0]
            t_y = (zone[3] - floor_position[1]) / ray_3D[:, 1]
            t_x_neg = (zone[0] - floor_position[0]) / ray_3D[:, 0]
            t_y_neg = (zone[1] - floor_position[1]) / ray_3D[:, 1]
            # take the t which has the smallest absolute value but keep the sign
            t = [t_x, t_y, t_x_neg, t_y_neg][np.argmin(np.abs([t_x, t_y, t_x_neg, t_y_neg]))]
            floor_position = floor_position + ray_3D[0] * t
            print("Corrected point from: ", standing_position[0], " -> ", floor_position)
        return np.array([floor_position])

def estimate_floor_position(P_inv, camera_origin, bbox, redzones, greenzones, avg_heigth=100):
    """Estimate floor position of person by checking how tall this
    person would be if the bbox would show the full person. If this
    is not the case take the head position as the estimate.
    """
    bottom_center_2D = np.array([bottom_center(bbox)])
    bottom_center_2D_h = np.concatenate((bottom_center_2D, np.ones((bottom_center_2D.shape[0],1))), axis=1)
    bottom_center_3D_h = P_inv @ bottom_center_2D_h.T
    bottom_center_3D = (bottom_center_3D_h / bottom_center_3D_h[3])[:3].T
    bottom_ray_3D = bottom_center_3D - camera_origin
    bottom_t = (0 - camera_origin[2]) / bottom_ray_3D[:, 2]
    theo_standing_position = camera_origin + bottom_ray_3D * bottom_t
    depth_standing_position = theo_standing_position[0, 1]

    top_center_2D = np.array([top_center(bbox)])
    top_center_2D_h = np.concatenate((top_center_2D, np.ones((top_center_2D.shape[0],1))), axis=1)
    top_center_3D_h = P_inv @ top_center_2D_h.T
    top_center_3D = (top_center_3D_h / top_center_3D_h[3])[:3].T
    top_ray_3D = top_center_3D - camera_origin
    top_t = (depth_standing_position - camera_origin[1]) / top_ray_3D[:, 1]
    top_center_world = camera_origin + top_ray_3D * top_t

    # check if theoretical height is large enough
    # to assume the whole person is in the frame
    if  False or check_in_zones(bottom_center_2D[0], greenzones)[0]: # first condition 'top_center_world[0, 2] > 100'
        corrected_floor_position = correct_for_redzone(theo_standing_position, bottom_ray_3D, redzones)
        return corrected_floor_position[0, :2].astype(int)
    else:
        head_t = (avg_heigth - camera_origin[2]) / top_ray_3D[:, 2]
        head_standing_position = camera_origin + top_ray_3D * head_t
        corrected_floor_position = correct_for_redzone(head_standing_position, bottom_ray_3D, redzones)
        return corrected_floor_position[0, :2].astype(int)
    
def load_zones(project: Path):
    with open(str(project / "floor_redzones.txt"), "r") as f:
        floor_redzones = [list(map(float, line.split())) for line in f]
    with open(str(project / "image_greenzones.txt"), "r") as f:
        image_greenzones = [list(map(float, line.split())) for line in f]
    return floor_redzones, image_greenzones

def load_projection_matrix(project: Path):
    P = np.load(str(project / "projection_matrix.npy"))
    P_inv = np.load(str(project / "projection_matrix_inv.npy"))
    camera_origin = np.load(str(project / "camera_origin.npy"))
    return P, P_inv, camera_origin


if __name__ == "__main__":
    # load points
    project = "Ch4_demo_960"
    P, P_inv, camera_origin = load_projection_matrix(project)
    floor_redzones = np.array([[20, 495, 90, 615]])
    image_greenzones = np.array([[0 , 500, 961, 721], 
                                    [420, 100, 520, 540]])
    bbox = [80, 120, 215, 200]
    print(estimate_floor_position(P_inv, camera_origin, bbox, floor_redzones, image_greenzones))





