# -*- coding: utf-8 -*-
# T-TFIG
# @Nathaphong Chinyooyong
# more stable and destructure
from pass_value import For_Pass_value
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import random
import math


'''  Set up  '''
# Choice 1
Screen_X = 640
Screen_Y = 480
Frame_depth = 30
Frame_color = 30

# Choice 2
#Screen_X = 848
#Screen_Y = 480
#Frame_depth = 90
#Frame_color = 30

# Choice 3
#Screen_X = 1280
#Screen_Y = 720
#Frame_depth = 6
#Frame_color = 15

# Special Choice
#Screen_X_depth = 840
#Screen_Y_depth = 480
#Screen_X_color = 1920
#Screen_Y_color = 1080
#Frame_depth = 6
#Frame_color = 8

# for check to pass through AI_Depth


model = torch.hub.load('ultralytics/yolov5', 'custom', 'static/best.pt')
pipeline = rs.pipeline()  # Define the process pipeline, Create a pipe
config = rs.config()  # Define configuration config


config.enable_stream(rs.stream.depth, 1280, 720,
                     rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720,
                     rs.format.bgr8, 15)

pipe_profile = pipeline.start(config)

#  Create aligned objects with color Flow alignment
# align_to  Is the stream type for which the depth frame is scheduled to be aligned
align_to = rs.stream.color
# rs.align  Align the depth frame with other frames
align = rs.align(align_to)


'''  Get the alignment image frame and camera parameters  '''


def get_aligned_images():

    # Wait for image frame , Get the frameset of color and depth
    frames = pipeline.wait_for_frames()
    # Get alignment frame , Align the depth box with the color box
    aligned_frames = align.process(frames)

    # Gets the in the alignment frame depth frame
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # Gets the in the alignment frame color frame
    aligned_color_frame = aligned_frames.get_color_frame()

    ####  Get camera parameters  ####
    # Get the depth parameter （ Pixel coordinate system to camera coordinate system will use ）
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = aligned_color_frame.profile.as_video_stream_profile(
    ).intrinsics  # Get camera internal parameters

    ####  take images To numpy arrays ####
    img_color = np.asanyarray(aligned_color_frame.get_data())       # RGB chart
    # Depth map （ Default 16 position ）
    img_depth = np.asanyarray(aligned_depth_frame.get_data())

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame


'''  Obtain the 3D coordinates of random points  '''


def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    if x + 1 <= 1:
        x = 1
    elif x >= Screen_X - 1:
        x = Screen_X - 1
    elif y + 1 <= 1:
        y = 1
    while y > 719:
        y = 718

    dis = aligned_depth_frame.get_distance(x, y)
    camera_coordinate = rs.rs2_deproject_pixel_to_point(
        depth_intrin, depth_pixel, dis)
    return dis, camera_coordinate


def checkbox(inter, check):
    inter = [int(inter[0]), int(inter[1]), int(inter[2]), int(inter[3])]
    check = [int(check[0]), int(check[1]), int(check[2]), int(check[3])]
    x_inter = int(inter[2]) - int(inter[0])
    y_inter = int(inter[3]) - int(inter[1])
    inter[0], inter[1], inter[2], inter[3] = inter[0] - \
        (x_inter/3), inter[1]-(y_inter/3), inter[2] + \
        (x_inter/3), inter[3]-(y_inter/2)
    x_check = ((check[0] > inter[0]) and (check[2] < inter[2]))
    y_check = ((check[1] > inter[1]) and (check[3] < inter[3]))
    return (x_check and y_check)


def resize_box(boxbox):
    len_size_x = int(boxbox[2]) - int(boxbox[0])  # calibate x
    len_size_y = int(boxbox[3]) - int(boxbox[1])  # calibate y
    box_size_x = int((len_size_x * 0.2)/2)
    box_size_y = int((len_size_y * 0.2)/2)
    return box_size_x, box_size_y


def get_stem_labels(model):
    pred = model(img_color)
    stem = []
    purple = []
    labels = pred.xyxy[0]
    for i in labels:
        if int(i[5]) == 3 and float(i[4]) > 0.50:
            stem.append([int(i[0]), int(i[1]), int(i[2]), int(i[3])])
        if int(i[5]) == 2 and float(i[4]) > 0.75:
            purple.append([int(i[0]), int(i[1]), int(i[2]), int(i[3])])
    focus = []
    for i in stem:
        for j in purple:
            if checkbox(j, i):
                focus.append(i)
                break
    # print(pred.pandas().xyxy[0])  # for check
    return focus, purple, labels


# def zero_filter(Rist):
#    Ans = []
#    for i in range(0, len(Rist)):
#        for j in range(0, 3):
#            Ans.append(round(Rist[i][j], 2))
#    return Ans


def first_filter(array):
    get_each_graph = []
    Ans = []
    for i in array:
        for j in i:
            get_each_graph.append(round(j, 2))
        Ans.append(get_each_graph)
        get_each_graph = []
    return second_filter(Ans)


def second_filter(interested_array):
    i_list = []
    j_list = []
    for i in interested_array:
        for j in i:
            if (str(j)[-1]) in ["1", '2', "3"]:
                j_list.append(float(str(j)[:3] + str(2)))
            elif (str(j)[-1]) in ['4', "5", '6']:
                j_list.append(float(str(j)[:3] + str(5)))
            elif (str(j)[-1]) in ["7", '8', "9"]:
                j_list.append(float(str(j)[:3] + str(8)))
            else:
                j_list.append(j)
        i_list.append(j_list)
        j_list = []
    Ans = i_list  # ไม่มีประโยชน์แต่สวยดี
    return Ans


def let_it_go(check_pass):
    while check_pass == False:
        video_capture_1 = cv2.VideoCapture(3)
        ret1, frame1 = video_capture_1.read()
        if (ret1):
            #cv2.imshow("Cam 1", frame1)
            pred = model(frame1)
            labels = pred.xyxy[0]
            print(pred.pandas().xyxy[0])
            print("loop nahee")
            for i in labels:
                print("kuy krub")
                if i[5] == 2 and float(i[4]) > 0.7:
                    check_pass = True
                    # video_capture_1.release()
                    # cv2.destroyAllWindows()
                    break
    return check_pass


def get_zero_out(list):  # [[x,y,z]]
    ans = []
    for i in list:
        sta = 1
        for j in i:
            if abs(float(j)) == 0:
                sta = 0
                break
        if sta == 1:
            ans.append(i)
    return ans


def convert(List):  # [[x,y,z],[x,y,z]]  #cm CM Cm cM
    List = get_zero_out(List)
    # print(List)
    ans = []
    for i in List:  # [x,y,z]
        ans.append([round(((i[0]**2)+(i[1]**2))**(1/2), 2),
                   round(math.atan(i[0]/i[1])*57.29, 2), i[2]])
        ans.sort()
    return ans


if __name__ == "__main__":
    while True:
        check_stem = False
        keep_value = []
        Random_list = []
        keep_random_list = []
        point_list = []
        Pass_through_value = []
        count = 0
        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images(
        )

        for j in get_stem_labels(model):
            #pred = get_stem_labels(model)
            # print(pred.pandas().xyxy[0])
            for i in j:
                for check in get_stem_labels(model)[2]:
                    if check[5] == 2:
                        check_stem = True  # เช็คว่าเจอก้านไหม
                center_x = int((i[0]+i[2])/2)  # mid_x
                center_y = int((i[1]+i[3])/2)  # mid_y
                box_re = resize_box(i)

                center_point = [center_x, center_y]
                top_left = [int(i[0]), int(i[1])]
                bottom_right = [int(i[2]), int(i[3])]
                top_middle_x = [center_x, int(i[1])]
                bottom_middle_x = [center_x, int(i[3])]

                keep_depth = [center_point, top_left,
                              bottom_right, top_middle_x, bottom_middle_x]

                #keep_depth = [top_middle_x]
                # Random dot but non essential
                for time in range(50):
                    X_1 = random.randint(
                        center_x - box_re[0], center_x + box_re[1])
                    Y_1 = random.randint(
                        center_y - box_re[0], center_y + box_re[1])
                    Random_list.append([X_1, Y_1])
                # End of dot
                Random_list.sort()
                for i in range(int(len(Random_list)*0.1)):
                    Random_list.pop(-i)

                # Change_coordinate_zone
                    # Essential dot
                for i in keep_depth:
                    depth_pixel = i
                    dis, camera_coordinate = get_3d_camera_coordinate(
                        depth_pixel, aligned_depth_frame, depth_intrin)
                    keep_value.append(camera_coordinate)

                # for i in keep_value:
                #    for j in i:
                #        j = round(j, 2)
                # print(keep_value)

                # Random dot
                for random_dot in Random_list:
                    depth_pixel = random_dot
                    dis, camera_coordinate = get_3d_camera_coordinate(
                        depth_pixel, aligned_depth_frame, depth_intrin)
                    keep_random_list.append(camera_coordinate)

                # Display_Zone 2

                for pixel_list in Random_list:
                    ran_x = pixel_list[0]
                    ran_y = pixel_list[1]
                    cv2.circle(img_color, (ran_x, ran_y), 0, (255, 0, 0), 1)
                # End of display Zone 2
                keep_value = first_filter(keep_value)
                keep_value = second_filter(keep_value)
                # Calculate for clener depth
                # find radius
                real_radius = abs(keep_value[1][0] - keep_value[2][0])/2

                # find depth average
                depth_value = 0
                for i in range(len(keep_random_list)):
                    depth_value += keep_random_list[i][2]
                depth_value /= len(keep_random_list)

                # ค่าที่ส่งออก (r,delta,depth)
                after_filter = convert(keep_value)
                print(after_filter)
                # print(after_filter)
                keep_value = []
                # end process of middle depth
            cv2.imshow('RealSence', img_color)
            key = cv2.waitKey(1)


# output 253 และ 318
