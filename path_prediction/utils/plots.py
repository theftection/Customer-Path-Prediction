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

def draw_icon(img, icon, position):

    x, y = position

    # check if x,y is within the image
    if x < 0 or y < 0 or x > img.shape[1] or y > img.shape[0]:
        return img

    if icon.shape[2] == 4:
        alpha = icon[:,:,3]
        mask = cv2.merge((alpha,alpha,alpha))
        icon = icon[:,:,0:3]

    icon_size = icon.shape[:2]
    icon = cv2.resize(icon, icon_size)

    x_offset = max(0, x - int(icon_size[0]/2))
    y_offset = max(0, y - int(icon_size[1]/2))

    roi = img[y_offset:y_offset+icon_size[1], x_offset:x_offset+icon_size[0]]

    icon_gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(icon_gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    icon_fg = cv2.bitwise_and(icon, icon, mask=mask)
    # Combine the icon and ROI to get the final img
    dst = cv2.add(roi_bg, icon_fg)
    img[y_offset:y_offset+icon_size[1], x_offset:x_offset+icon_size[0]] = dst

    return img