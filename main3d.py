import numpy as np
import cv2
import random
import os
import time
import datetime


NO_FRAMES = 30
FPS = 30

###
# tweakbles

NO_ADJ_BIRDS = 3
X_FORCE_COEF = 0.006 / NO_ADJ_BIRDS
V_FORCE_COEF = 0.49 / NO_ADJ_BIRDS
DRAG_COEF = 0.4
NO_BIRDS = 1000
RESOLUTION_VAR = 2

# dimensions of sky

DEPTH = 1200
HEIGHT = [300, 600, 720, 1080, 1440][RESOLUTION_VAR]
WIDTH = [400, 800, 1280, 1920, 2560][RESOLUTION_VAR]

###

V_INIT_MAX = 5

SINGLE_BIRD = np.array([255, 255, 255], np.uint8)
SINGLE_SKY = np.array([0, 0, 0], np.uint8)

PROJECTION = np.full((HEIGHT, WIDTH, 3), SINGLE_SKY)

# z, y, x, vz, vy, vx, fz, fy, fx
STARLINGS = None


last_frame_positions = np.zeros((2, NO_BIRDS), dtype=int)


def bird_maker():
    # create all the birds and gives them their positions, velocities and forces

    global STARLINGS

    bird_mask_indices = random.sample(range(DEPTH * HEIGHT * WIDTH), NO_BIRDS)
    HEIGHTxWIDTH = HEIGHT * WIDTH
    STARLINGS = np.array([[i // HEIGHTxWIDTH for i in bird_mask_indices], [i % HEIGHTxWIDTH // WIDTH for i in bird_mask_indices], [i % HEIGHTxWIDTH % WIDTH for i in bird_mask_indices], random.choices(range(-V_INIT_MAX, V_INIT_MAX + 1),
                         None, k=NO_BIRDS), random.choices(range(-V_INIT_MAX, V_INIT_MAX + 1), None, k=NO_BIRDS), random.choices(range(-V_INIT_MAX, V_INIT_MAX + 1), None, k=NO_BIRDS), np.zeros(NO_BIRDS), np.zeros(NO_BIRDS), np.zeros(NO_BIRDS)], dtype=int)


bird_maker()


def projector():
    # projects 3D sky onto a 2D frame

    PROJECTION[last_frame_positions[0], last_frame_positions[1]] = SINGLE_SKY
    y_x_coord_list = STARLINGS[1:3]
    PROJECTION[y_x_coord_list[0],
               y_x_coord_list[1]] = SINGLE_BIRD


projector()


def forcer():
    # calculates what force acts on each bird and changes their velocity accordingly

    # calculates force on each bird (excluding drag)
    for bird in range(NO_BIRDS):
        dist_arr = np.sum(np.delete(np.square(
            STARLINGS[0:3, bird, np.newaxis]-STARLINGS[0:3]), bird, axis=1), axis=0)

        adj_bird_arr = (np.argpartition(
            dist_arr, NO_ADJ_BIRDS)[:NO_ADJ_BIRDS])
        STARLINGS[6:9, bird] = (X_FORCE_COEF * np.sum(STARLINGS[:3, adj_bird_arr] - STARLINGS[:3, bird,
                                np.newaxis], axis=1) + V_FORCE_COEF * np.sum(STARLINGS[3:6, adj_bird_arr], axis=1)).astype(int)

    # applies above claculated force + drag force
    STARLINGS[3:6] += (STARLINGS[6:9] - DRAG_COEF *
                       STARLINGS[3:6]).astype(int)


def mover():
    global last_frame_positions
    # changes birds' positions based on their velocity

    # coarse positions changes by just adding their velocities to positions

    last_frame_positions = STARLINGS[1:3].copy()

    STARLINGS[:3] += STARLINGS[3:6]

    # reverses speed at rebound
    rebound_bool_arr = np.logical_or(STARLINGS[:3] >= np.array(
        [[DEPTH], [HEIGHT], [WIDTH]]), STARLINGS[:3] < np.array([[0], [0], [0]])).astype(int)

    STARLINGS[3:6] = np.where(
        rebound_bool_arr, -STARLINGS[3:6], STARLINGS[3:6])

    # rebounds position
    rebound_bool_arr = STARLINGS[:3] >= np.array([[DEPTH], [HEIGHT], [WIDTH]])
    STARLINGS[:3] = np.where(
        rebound_bool_arr, STARLINGS[:3]-(STARLINGS[:3] % np.array([[DEPTH], [HEIGHT], [WIDTH]]) + np.array([[1], [1], [1]])), STARLINGS[:3])

    rebound_bool_arr = STARLINGS[:3] < np.array([[0], [0], [0]])
    STARLINGS[:3] = np.where(
        rebound_bool_arr, -(STARLINGS[:3] + np.array([[1], [1], [1]])), STARLINGS[:3])


def main():
    # runs all all functions for each frame and creates the video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    vid_no_var = 0
    while 1:
        file_name = f'starlings_3d_{vid_no_var}.mp4'
        if not os.path.exists(file_name):
            out = cv2.VideoWriter(file_name, fourcc, FPS, (WIDTH, HEIGHT))
            break
        vid_no_var += 1

    print(f"\nGenerating video starlings_3d_{vid_no_var}.mp4")

    start_time = time.time()
    no_decimals_perc = 1

    for frame in range(NO_FRAMES):
        print(f'\r{(frame+1)*10**(no_decimals_perc + 2)//NO_FRAMES/10**no_decimals_perc}%\t|    Time remaining: {datetime.timedelta(seconds=(NO_FRAMES-frame)//((frame+1)/(time.time() - start_time + 1)))}', end='\t\t')

        forcer()
        mover()
        projector()

        out.write(PROJECTION)

main()


'''
- Optimize mover funtion
'''
