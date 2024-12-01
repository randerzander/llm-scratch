import imageio
import cv2
import numpy as np

# Load GIF
gif = imageio.mimread('calvin.gif')
frames = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGR) for frame in gif]

def is_panel(box, w_thresh=200, h_thresh=200):
    x, y, w, h = box
    return w > w_thresh and h > h_thresh

def hidden_panel(box1, box2, dist_thresh=100):
    if edge_distance(box1, box2) > dist_thresh:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        right_edge = x1+w1
        left_edge = x2
        x3 = right_edge
        y3 = y1
        w3 = left_edge - right_edge
        h3 = h1
        return (x3, y3, w3, h3)
    else:
        return


def edge_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    right_edge = x1+w1
    left_edge = x2
    return left_edge - right_edge

green = (0, 255, 0) 
red = (0, 0, 255)
yellow = (0, 255, 255)
black = (0, 0, 0)


# green by default
def draw_box(box, frame, color=(0, 255, 0)):
    x, y, w, h = box
    print(box)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def extract_image(idx, panel, frame):
    x, y, w, h = panel
    extracted = frame[y:y+h, x:x+w]
    cv2.imwrite(f'panel_{idx}.png', extracted)

for idx, frame in enumerate(frames):
    # find edged panels
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # many false positive panels.. apply threshold
    panels = [box for box in bounding_boxes if is_panel(box)]

    # outline opencv found panels in green
    # [draw_box(box, frame, green) for box in panels]

    # between each panel, check if distance is about panel sized.. i.e. there's an unboredered panel between
    hp = hidden_panel(panels[0], panels[1])
    if hp:
        panels.append(hp)
        #draw_box(hp, frame, yellow)

    hp = hidden_panel(panels[1], panels[2])
    if hp:
        panels.append(hp)
        #draw_box(hp, frame, yellow)

    # save comic w/ panels highlighted, for visual sanity check
    # cv2.imwrite(f'highlighted_panels.png', frame)

    # sort by x position
    panels = sorted(panels, key=lambda x: x[0])

    # save panels as separate images
    [extract_image(idx, panel, frame) for idx, panel in enumerate(panels)]

print("Bounding boxes saved in output frames.")
