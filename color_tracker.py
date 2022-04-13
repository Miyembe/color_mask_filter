import cv2
import numpy as np
import argparse
import datetime
import time
import csv
import os
import _thread
import pandas as pd
from functools import partial
from collections import deque

color_tracker_window = "Video Capture"
filtered_window = "Filtered Image"
foreground_window = "Background Substitution"
subtracted_window = "Background Substitution Applied"
track_bar = "track_bar"

def doNothing(x):
    pass
    
def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):#
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    
    return cv2.resize(image, dim, interpolation=inter)

def getFrame(frame_no, video):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
def setSpeed(val, speed):
    speed = max(val, 1)

#########
## Constant Values
#########
max_value = 255
low_H = 0
low_S = 0
low_V = 0
high_H = 180
high_S = max_value
high_V = max_value
##########
class AntsHoleDetermination:
    def __init__(self, num_holes):
        self.num_obj_inner = [[] for _ in range(num_holes)]
        self.num_obj_outer = [[] for _ in range(num_holes)]

class ColorTracker:
    def __init__(self, args):
        #cv2.namedWindow(color_tracker_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(track_bar, cv2.WINDOW_NORMAL)
    
        self.video_name = args.video_loc
        self.window_width = None
        self.window_height = 900 
        
        cv2.createTrackbar('min_h', track_bar, 0, high_H, doNothing)
        cv2.createTrackbar('min_s', track_bar, 0, high_S, doNothing)
        cv2.createTrackbar('min_v', track_bar, 0, high_V, doNothing)
        cv2.createTrackbar('max_h', track_bar, high_H, high_H, doNothing)
        cv2.createTrackbar('max_s', track_bar, high_S, high_S, doNothing)
        cv2.createTrackbar('max_v', track_bar, high_V, high_V, doNothing)
        if args.input == 'camera':
            self.capture = cv2.VideoCapture(0)
        elif args.input == 'video':
            try:
                self.capture = cv2.VideoCapture(f"{args.video_loc}")
            except:
                print("The video address is not correctly given!")
        else:
            raise Exception("Neither camera nor video is provided as input.")

        self.ret, self.frame = self.capture.read()
        self.frame = resizeWithAspectRatio(self.frame, height=self.window_height)
        cv2.imshow(color_tracker_window, self.frame)
        self.no_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.play_speed = 50

        cv2.createTrackbar('Frame', color_tracker_window, 0, self.no_frames, partial(getFrame,video = self.capture))
        #cv2.createTrackbar('Speed', color_tracker_window, self.play_speed, 100, partial(setSpeed, speed=self.play_speed))

        self.min_h = low_H
        self.min_s = low_S
        self.min_v = low_V

        self.max_h = high_H
        self.max_s = high_S
        self.max_v = high_V

        # Background Subtraction Algorithm
        self.back_sub = cv2.createBackgroundSubtractorMOG2()

        # HSV range file generation
        self.hsv_range_file_name = "hsv_range.csv"
        self.hsv_range_file_header = ['time', 'min_h', 'min_s', 'min_v', 'max_h', 'max_s', 'max_v', 'colors']
        if self.hsv_range_file_name not in os.listdir():
            with open(self.hsv_range_file_name, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.hsv_range_file_header)

        # FPS counter
        self.cur_time = 0
        self.prev_time = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Segmentation related
        self.top_left_corner = None
        self.bottom_right_corner = None
        self.mouse_flag = 0
        if args.num_holes is not None:
            self.num_holes = int(args.num_holes)
            self.list_holes = [[0,0] for i in range(self.num_holes)]
            self.list_holes_big = [[0,0] for i in range(self.num_holes)]
        else: self.num_holes = args.num_holes
        self.list_index = 0
        self.isHoleReady = False
        self.justReady = False

        # rescaling
        self.rescaling_factor = [2160/506, 3840/900]

        # Determination whether ants went into hole related
        self.num_object_color = deque(maxlen=2) # storing things 

    

        # HSV value loading
        self.hsv_trackbar = args.hsv_trackbar
        print(f"hsv_trackbar: {args.hsv_trackbar}")
        if args.hsv_trackbar == 1:
            print("hsv_trackbar is being used")
            self.is_hsv_trackbar = True
            # Color Extraction by click
            self.click_flag = False
            self.extracted_hsv = []
            self.extracted_hsv_name = "extracted_hsv.csv"
            self.extracted_hsv_file_header = ['video_name', 'num_frame', 'colors', 'x', 'y', 'h', 's', 'v', 'comment']
            if self.extracted_hsv_name not in os.listdir():
                with open(self.extracted_hsv_name, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.extracted_hsv_file_header)
        else:
            print("hsv_trackbar is not being used")
            self.is_hsv_trackbar = False
            #self.hsv_ranges = [[0,0] for _ in range(self.num_colors)] 
            #self.name_colors = [blue, red, green, yellow] #etc
            hsv_range_csv = pd.read_csv('hsv_range.csv')
            hsv_range_color = hsv_range_csv['colors']
            self.list_colors = hsv_range_color.tolist()
            self.num_colors = len(self.list_colors)
            self.hsv_ranges = [[0,0] for _ in range(self.num_colors)] 
            for i, color in enumerate(self.list_colors):
                hsv_range_row = hsv_range_csv.loc[hsv_range_csv['colors']==color]
                self.hsv_ranges[i][0] = [hsv_range_row['min_h'].values[0], hsv_range_row['min_s'].values[0], hsv_range_row['min_v'].values[0]]
                self.hsv_ranges[i][1] = [hsv_range_row['max_h'].values[0], hsv_range_row['max_s'].values[0], hsv_range_row['max_v'].values[0]]

            # Save dict_count and dict_count_log
            self.dict_count_file_name = f"{self.video_name}_color_count.csv"
            self.dict_count_log_file_name = f"{self.video_name}_color_count_log.csv"
            self.dict_count_file_header = ['box_id'] + self.list_colors
            self.dict_count_log_file_header = ['frame_no', 'box_id'] + self.list_colors
            
            if self.dict_count_file_name not in os.listdir():
                with open(self.dict_count_file_name, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.dict_count_file_header)
            if self.dict_count_log_file_name not in os.listdir():
                with open(self.dict_count_log_file_name, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.dict_count_log_file_header)
            # Determine whether ants go into the holes
        
            self.dict_count = dict.fromkeys([f'Box_ID_{i}' for i in range(self.num_holes)])
            for key in self.dict_count:
                self.dict_count[key] = dict.fromkeys(self.list_colors)
                for color in self.dict_count[key]:
                    self.dict_count[key][color] = 0
            self.cur_dict_num_objects = None
            self.prev_dict_num_objects = None

            if int(args.load_box) == 1:
                self.list_holes, self.list_holes_big = self.loadBoxLoc('GH020284.MP4') # Non-rescaling version
                #self.list_holes, self.list_holes_big = self.loadBoxLoc('GH020284.MP4', self.rescaling_factor)
                print(f"self.list_holes: {type(self.list_holes)}, self.list_holes_big: {self.list_holes_big}")

            

    def saveRectangle(self, action, x, y, flags, padding):
        #print("Instance called")
        #print("Hello")
        # TODO - need to put exception when the padded coordinates is out of the window. 
        if action == cv2.EVENT_LBUTTONDOWN:
            print("Top Left coordinate is saved.")
            self.list_holes[self.list_index][0] = [(x,y)]   
            self.list_holes_big[self.list_index][0] = [(x-padding,y-padding)]
        elif action == cv2.EVENT_LBUTTONUP:
            self.list_holes[self.list_index][1] = [(x,y)]
            self.list_holes_big[self.list_index][1] = [(x+padding,y+padding)]
            self.list_index += 1
            print(f"Top Left and Bottom Right Coordinates are now set !!!")


    def drawRectangle(self, window, frame):
        if self.top_left_corner != None and self.bottom_right_corner != None:
            cv2.rectangle(frame, self.top_left_corner[0], self.bottom_right_corner[0], (0, 255, 0), 2, 8)
            cv2.imshow(window, frame)
        else: pass

    def drawAllRectangle(self, window, frame, coordinates):
        for i, coordinate in enumerate(coordinates):
            cv2.rectangle(frame, tuple(coordinate[0][0]), tuple(coordinate[1][0]), (0, 255, 0), 2, 8)
            cv2.imshow(window, frame)

    def saveBoxLoc(self, list_holes, list_holes_big):
        list_holes_npy = np.array(list_holes)
        list_holes_big_npy = np.array(list_holes_big)
        
        with open(f"{self.video_name}_box_loc.npy", 'wb') as f:
            np.save(f, list_holes_npy)
            np.save(f, list_holes_big_npy)
        
        print(f"Box locations for {self.video_name} is successfully saved.")
    
    def loadBoxLoc(self, video_name, rescaling_factor = None):
        # resize_factor = (row, height)
        with open(f"{video_name}_box_loc.npy", 'rb') as f:
            list_holes = np.load(f, allow_pickle = True)
            list_holes_big = np.load(f, allow_pickle = True)
        
        print(f"Smaller Box: {list_holes}, Larger Box: {list_holes_big}")
        print(f"rescaling_factor: {rescaling_factor}")
        if rescaling_factor is not None:
            for hole in list_holes:
                for corner in hole:
                    #print(f"corner: {corner}, rescaling_factor: {rescaling_factor}")
                    corner[0][0] = int(corner[0][0]*rescaling_factor[0])
                    corner[0][1] = int(corner[0][1]*rescaling_factor[1])
            for hole in list_holes_big:
                for corner in hole:
                    corner[0][0] = int(corner[0][0]*rescaling_factor[0])
                    corner[0][1] = int(corner[0][1]*rescaling_factor[1])
                    
        self.isHoleReady = True
        #print(f"Box Locations are successfully loaded from {video_name}")
        #print(f"Smaller Box: {list_holes}, Larger Box: {list_holes_big}")
        #print(f"Box Shape: {np.array(list_holes).shape}")
        return list_holes.tolist(), list_holes_big.tolist()
    
    def chooseHoles(self, window, frame, num_holes = 0, padding = 50):
        # Need to generate empty list with the len of num_holes
        self.isHoleReady = False
        if self.mouse_flag == 1:
            if self.list_index < self.num_holes:
                cv2.setMouseCallback(window, self.saveRectangle, padding) 
                #print(f"self.list_index: {self.list_index}")
                #print(f"self.mouse_flag: {self.mouse_flag}")
            else:
                self.list_index = 0
                self.mouse_flag = 0
                cv2.setMouseCallback(window, lambda x, y, flags, *userdata: None) # Disable MouseCallback function
                self.isHoleReady = True
                self.justReady = True
                print("Hole Segmentation is done")

    def extractColor(self, action, x, y, flags, userdata):
        hsv_frame = userdata[0]
        width = userdata[1]
        height = userdata[2]
        if action == cv2.EVENT_RBUTTONDOWN:
            # OpenCV window, x and y (row and column are swapped)
            print(f"Color is extracted from pixel position x: {y}, and y: {x}")
            color_name = input("Enter the name of the color you extracted: ")
            comment = input("Enter a comment you want to add: ")
            no_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            # Get hsv value for the surrounding pixels
            for i in range(width):
                for j in range(height):
                    self.extracted_hsv = hsv_frame[y+i-int(width-1/2),x+j-int(height-1/2)]
                    extracted_hsv_data = [f'{self.video_name}', f'{no_frame}', f'{color_name}', f'{y+i-int(width/2)}',
                                    f'{x+j-int(height/2)}', f'{self.extracted_hsv[0]}', f'{self.extracted_hsv[1]}', f'{self.extracted_hsv[2]}', f'{comment}']
                    with open(self.extracted_hsv_name, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(extracted_hsv_data)
            self.click_flag = 0

    def checkPixelHSVFrame(self, window, frame, hsv_range, min_num = 1, padding = 50):
        '''
        Check if the pixels fit within the range in the frame.
        '''
        shape_frame = np.array(frame).shape # (row, column, channel)
        list_outer_in_range = []
        list_inner_in_range = []

        for i in range(shape_frame[0]):
            for j in range(shape_frame[1]):
                isInRange = True
                for val, min_range, max_range in zip(frame[i,j], hsv_range[0], hsv_range[1]):
                    if val < min_range or val > max_range:
                        isInRange = False
                        break
                    else: continue
                if isInRange == True:
                    list_outer_in_range.append([i, j])
                    #print("Colored Pixels in Outer Box")
                    if (i >= padding and i < shape_frame[0] - padding) and (j >= padding and j < shape_frame[1]):
                        list_inner_in_range.append([i, j])
                        #print("Colored Pixels in Inner Box")
        if len(list_inner_in_range) < min_num:
            for i in list_inner_in_range:
                if i is not None:
                    list_outer_in_range.remove(i)
            list_inner_in_range = []
        if len(list_outer_in_range) < min_num:
            list_outer_in_range = []



        return list_outer_in_range, list_inner_in_range
    def manhattan_distance(self, p1, p2):
        distance = 0
        for x1, x2 in zip(p1, p2):
            abs_difference = abs(x2-x1)
            distance += abs_difference
        return distance

    def common_elements(self, list1, list2):
        return [element for element in list1 if element in list2]

    def groupingPixels(self, list_pixels, min_group_dist):

        if len(list_pixels) != 0:
            list_group = []
            #print(f"list_pixels: {list_pixels}")
            list_group.append([list_pixels[0]])
            # Sorting the pixels into a group
            for i, px in enumerate(list_pixels):
                found_group = False
                for j, lg in enumerate(list_group):
                    for mem in lg:
                        if self.manhattan_distance(px, mem) <= min_group_dist:
                            list_group[j].append(px)
                            found_group = True
                            break
                    if found_group == True:
                        break
                if found_group == False:
                    list_group.append([list_pixels[i]])
            
            # Merging groups
            list_sim_idx = []
            for i, gp1 in enumerate(list_group):
                for j, gp2 in enumerate(list_group):
                    if i != j:
                        list_inter = self.common_elements(gp1, gp2)     
                        if list_inter is not list():
                            list_sim_idx.append([i,j])
            
            list_sim_idx = np.unique(np.array(list_sim_idx).reshape(1,-1).squeeze())
            list_sep_idx = np.arange(0, len(list_group))
            for i in list_sim_idx:
                list_sep_idx = np.delete(list_sep_idx, np.where(list_sep_idx == i))

            #print(f"list_sim_idx: {list_sim_idx}, list_sep_idx: {list_sep_idx}")
            if len(list_sim_idx) > 1: 
                list_group_final = [np.concatenate([list_group[i] for i in list_sim_idx]).tolist()]
            else: list_group_final = []

            for i in list_sep_idx:
                list_group_final.append(list_group[i])
        
            return list_group_final, len(list_group_final)
        
        else: return list(), 0

    def compareDictNumObj(self, cur_dict, prev_dict):
        # cur_dict and prev_dict -- second order nested dictionaries.
        for box in cur_dict:
            for i, color in enumerate(cur_dict[box]):
                diff_num_obj = np.subtract(np.array(prev_dict[box][color]),np.array(cur_dict[box][color])) 
                if diff_num_obj[0] == 1 and diff_num_obj[1] == 1:
                    self.dict_count[box][color] += 1
                    print(f"Frame: {self.capture.get(cv2.CAP_PROP_POS_FRAMES)}, Box: {box}, Color: {color}, ant just went into hole!")
                    with open(self.dict_count_log_file_name, 'a') as f:
                        writer = csv.writer(f)
                        list_mask_color_count = [0 for _ in range(self.num_colors)]
                        list_mask_color_count[i] = 1
                        color_count_log_data = [int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)), box] + list_mask_color_count
                        writer.writerow(color_count_log_data)
    def run(self):
        video_start_time = time.time()
        while True:#
        # Image Collection & Processing
            ret, frame = self.capture.read()
            #frame = cv2.GaussianBlur(frame,(11,11), cv2.BORDER_DEFAULT)
            if ret:

                # Calculation FPS
                self.cur_time = time.time()
                t_d = self.cur_time - self.prev_time
                fps = 1/t_d
                self.prev_time = self.cur_time
                fps = str(int(fps))
                fps = f'FPS: {fps}, time: {t_d:.2f}'
                # Read from Track bar
                self.min_h = cv2.getTrackbarPos('min_h', track_bar)
                self.min_s = cv2.getTrackbarPos('min_s', track_bar)
                self.min_v = cv2.getTrackbarPos('min_v', track_bar)
                self.max_h = cv2.getTrackbarPos('max_h', track_bar)
                self.max_s = cv2.getTrackbarPos('max_s', track_bar)
                self.max_v = cv2.getTrackbarPos('max_v', track_bar)
                
                
                
                frame = resizeWithAspectRatio(frame, height=self.window_height)
                cv2.imshow(color_tracker_window, frame)
                if self.mouse_flag == 1:
                    self.chooseHoles(color_tracker_window, frame, num_holes = self.num_holes) # Choose holes with desired number with mouse drag.
                elif self.isHoleReady:
                    cv2.setMouseCallback(color_tracker_window, lambda x, y, flags, *userdata: None) # Stop choosing holes and visualise all the chosen rectangles
                    self.drawAllRectangle(color_tracker_window, frame, self.list_holes)
                    self.drawAllRectangle(color_tracker_window, frame, self.list_holes_big)                    
                    segmented_frame = [[] for _ in range(self.num_holes)]
                    segmented_sub_frame = [[] for _ in range(self.num_holes)]
                    for i in range(self.num_holes):
                        segmented_frame[i] = frame[self.list_holes_big[i][0][0][1]:self.list_holes_big[i][1][0][1], self.list_holes_big[i][0][0][0]:self.list_holes_big[i][1][0][0]]
                        cv2.imshow(f"segmented_image_{i}", segmented_frame[i])
                #cv2.putText(frame, fps,(7, 70), self.font, 3, (100, 255, 0), 3, cv2.LINE_AA)

                # Set the frame number from video
                #cv2.setTrackbarPos("Frame", color_tracker_window, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)))

                # HSV conversion and background substitution
                
                #cv2.imshow(foreground_window, fg_frame_viz)
                
                
                #cv2.putText(filtered_rgb_frame, fps,(7, 70), self.font, 3, (100, 255, 0), 3, cv2.LINE_AA)
                

                if self.is_hsv_trackbar:
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    if self.click_flag == 1:
                        cv2.setMouseCallback(color_tracker_window, self.extractColor, [hsv_frame, 3, 3])
                    else:
                        cv2.setMouseCallback(color_tracker_window, lambda x, y, flags, *userdata: None)
                    fg_frame = self.back_sub.apply(frame) 
                    fg_frame_viz = resizeWithAspectRatio(fg_frame, height=self.window_height)
                    min_values = np.array([self.min_h, self.min_s, self.min_v], dtype = "uint8")
                    max_values = np.array([self.max_h, self.max_s, self.max_v], dtype = "uint8")
                    masked_frame = cv2.inRange(hsv_frame, min_values, max_values)
                    filtered_hsv_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask = masked_frame)
                    filtered_rgb_frame = cv2.cvtColor(filtered_hsv_frame, cv2.COLOR_HSV2BGR)
                    subtracted_rgb_frame = cv2.bitwise_and(filtered_rgb_frame, filtered_rgb_frame, mask = fg_frame)
                    filtered_rgb_frame = resizeWithAspectRatio(filtered_rgb_frame, height = self.window_height)
                    #cv2.imshow(filtered_window, filtered_rgb_frame)
                    subtracted_rgb_frame = resizeWithAspectRatio(subtracted_rgb_frame, height = self.window_height)
                    cv2.putText(subtracted_rgb_frame, fps,(7, 70), self.font, 3, (100, 255, 0), 3, cv2.LINE_AA)
                    cv2.imshow(subtracted_window, subtracted_rgb_frame)
                    if self.isHoleReady == True:
                        for i in range(self.num_holes):
                            segmented_sub_frame[i] = subtracted_rgb_frame[self.list_holes[i][0][0][1]:self.list_holes[i][1][0][1], self.list_holes[i][0][0][0]:self.list_holes[i][1][0][0]]
                            cv2.imshow(f"segmented_filtered_image_{i}", segmented_sub_frame[i])
                            #print(f"x coordinates: {self.list_holes[i][0][0][0]}, {self.list_holes[i][1][0][0]}, y coordinates: {self.list_holes[i][0][0][1]}, {self.list_holes[i][1][0][1]} ")

                elif self.isHoleReady: 
                    # mask segmented frames and apply directly to the segmented frame.
                    if self.justReady:
                        self.justReady = False
                    else:
                        # I need to get the num_obj of (((outer, inner) * num_colors) * num_holes) for one frame - should I use nested dictionary?
                        # Like {hole_ID {blue: ..., pink: ..., etc}} I think it is really goo method. Every loop, I can make a nested dict, 
                        # Dictionary intialisation to store num_objects for each color in each hole.
                        # TODO Put the num_objects in this dict over the loop!!!
                        # TODO and complete determine function!!!
                        self.prev_dict_num_objects = self.cur_dict_num_objects
                        dict_num_objects = dict.fromkeys([f'Box_ID_{i}' for i in range(self.num_holes)])
                        for key in dict_num_objects:
                            dict_num_objects[key] = dict.fromkeys(self.list_colors)
                        for i, hsv_range in enumerate(self.hsv_ranges):
                            for j in range(self.num_holes):
                                hsv_frame = cv2.cvtColor(segmented_frame[j], cv2.COLOR_BGR2HSV)
                                fg_frame = self.back_sub.apply(segmented_frame[j])
                                subtracted_hsv_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask = fg_frame) 
                                masked_frame = cv2.inRange(subtracted_hsv_frame, np.array(hsv_range[0], dtype="uint8"), np.array(hsv_range[1], dtype="uint8"))
                                filtered_hsv_frame = cv2.bitwise_and(subtracted_hsv_frame, subtracted_hsv_frame, mask = masked_frame)
                                filtered_rgb_frame = cv2.cvtColor(filtered_hsv_frame, cv2.COLOR_HSV2BGR)
                                #subtracted_rgb_frame = cv2.bitwise_and(filtered_rgb_frame, filtered_rgb_frame, mask = fg_frame)
                                cv2.imshow(f"segmented_filtered_image_{j}_{self.list_colors[i]}", filtered_rgb_frame)
                                list_outer_in_range, list_inner_in_range = self.checkPixelHSVFrame(f"segmented_filtered_image_{j}_{self.list_colors[i]}", filtered_hsv_frame, hsv_range, min_num = 4)

                                #print(f"list_outer_in_range: {list_outer_in_range}, list_inner_in_range: {list_inner_in_range}") 
                                _, num_outer_group = self.groupingPixels(list_outer_in_range, 5)
                                _, num_inner_group = self.groupingPixels(list_inner_in_range, 5)
                                dict_num_objects[f'Box_ID_{j}'][self.list_colors[i]] = [num_outer_group, num_inner_group]
                                #print(f"list_outer_range: {list_outer_in_range}, list_inner_range: {list_inner_in_range}")
                                #print(f"dict_num_objects: {dict_num_objects}")
                                #print(f"list_outer_group: {list_outer_group}, num_group: {len(list_outer_group)}")
                                #print(f"Image_{j}, Outer box pixels: {list_outer_in_range}, Inner box pixels; {list_inner_in_range}")
                        self.cur_dict_num_objects = dict_num_objects
                        if self.cur_dict_num_objects is not None and self.prev_dict_num_objects is not None:
                            self.compareDictNumObj(self.cur_dict_num_objects, self.prev_dict_num_objects)



                # Keyboard inputs
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif cv2.waitKey(1) & 0xFF == ord('s'):
                    color_name = input("Enter the color name for this hsv range: ")
                    hsv_range_data = [f'{datetime.datetime.utcnow()}', f'{self.min_h}', f'{self.min_s}', f'{self.min_v}',
                                      f'{self.max_h}', f'{self.max_s}', f'{self.max_v}', f'{color_name}']
                    with open(self.hsv_range_file_name, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(hsv_range_data)
                elif cv2.waitKey(1) & 0xFF == ord('h'): # Hole segmentation
                    if self.mouse_flag == 0:
                        print("Mouse based segmentation is activated")
                        self.mouse_flag = 1
                    else:
                        print("Mouse based segmentation is deactivated")
                        self.mouse_flag = 0
                elif cv2.waitKey(1) & 0xFF == ord('c'): # Color Extraction by click
                    if self.click_flag == False:
                        print("Color Extraction by click is activated")
                        self.click_flag = True
                    else:
                        print("Color Extraction by click is deactivated")
                        self.click_flag = False
                elif cv2.waitKey(1) & 0xFF == ord('b'):
                    self.saveBoxLoc(self.list_holes, self.list_holes_big)
                elif cv2.waitKey(1) & 0xFF == ord('l'):
                    self.list_holes, self.list_holes_big = self.loadBoxLoc('GH020284.MP4') # non-rescaling version
                    #self.list_holes, self.list_holes_big = self.loadBoxLoc('GH020284.MP4', self.rescaling_factor)
                
                # elif cv2.waitKey(1) & 0xFF == ord('p'):
                #     print(f"Hue: {self.h}, Saturation: {self.s}, Value: {self.v}")
            elif ret is False and int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) >= int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                print(f"The video is ended ({int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))} / {int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))})")
                print(f"Time taken: {time.time() - video_start_time:2f}")
                break
        #print(f"dict_count: {self.dict_count}")
        if self.hsv_trackbar == 0:
            with open(self.dict_count_file_name, 'a') as f:
                writer = csv.writer(f)
                for i, box in enumerate(self.dict_count):
                    color_count_data = [i]
                    for color in self.dict_count[box]:
                        color_count_data.append(self.dict_count[box][color])        
                    writer.writerow(color_count_data)
        self.capture.release()
        cv2.destroyAllWindows()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Type of input source. Option: [camera, video]")
    parser.add_argument("--video_loc", help="Location of video only required when the input source is video.")
    parser.add_argument("--num_holes", help="Specifing number of holes needed to be segmented")
    parser.add_argument("--num_colors", help="Specifing number of colors used for experiments")
    parser.add_argument("--hsv_trackbar", help="Specifing if using trackbar for determining hsv ranges or using loaded hsv ranges from hsv_range.csv", default=1, type=int)
    parser.add_argument("--load_box", help="Load Boxes from saved npy file")
    args = parser.parse_args()

    color_tracker = ColorTracker(args)
    color_tracker.run()

