import os
import cv2
import numpy as np

def collect_3D_points(floor_plan):

    points = []
    print("Select h == 0 to exit the program")

    while True:
        h = int(input("Enter height of point: "))
        if h < 0:
            assert False, "Height must be positive"
        elif h == 0:
            break
            
        def select_points_plan(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y, 0])
                points.append([x, y, h])
                print("Floor plan point added:", [x, y, 0])
                print("Floor plan point added:", [x, y, h])

        cv2.namedWindow("floor_plan")
        cv2.setMouseCallback("floor_plan", select_points_plan)

        while True:
            cv2.imshow("floor_plan", floor_plan)
            for point in points:
                cv2.circle(floor_plan, tuple(point[:2]), 4, (0,0,255), -1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

    return np.array(points, dtype=np.float32)


def collect_2D_points(image):
    
        points = []
    
        def select_points_image(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                print("Image point added:", [x, y])
    
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", select_points_image)
    
        while True:
            cv2.imshow("image", image)
            for point in points:
                cv2.circle(image, tuple(point), 4, (0,0,255), -1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
    
        return np.array(points, dtype=np.float32)



if __name__ == "__main__":
    project = input("Enter project/camera name: ")
    floor_plan = cv2.imread(f"inference_data/projection_matrix/{project}/images/Ch4_floor_plan.png")
    image = cv2.imread(f"inference_data/projection_matrix/{project}/images/Ch4_image_undistorted_960.png")

    if floor_plan is None or image is None:
        assert False, "Could not load images"

    points_3D = collect_3D_points(floor_plan)
    points_2D = collect_2D_points(image)

    # save points
    os.makedirs(f"inference_data/projection_matrix/{project}/points", exist_ok=True)
    np.save(f"inference_data/projection_matrix/{project}/points/points_3D.npy", points_3D)
    np.save(f"inference_data/projection_matrix/{project}/points/points_2D.npy", points_2D)