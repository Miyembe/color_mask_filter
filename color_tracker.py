import cv2
import numpy as np
import argparse
import datetime
import time
import csv
import os
import _thread
from functools import partial

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

class ColorTracker:
    def __init__(self, args):
        #cv2.namedWindow(color_tracker_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(track_bar, cv2.WINDOW_NORMAL)
    

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
        self.hsv_range_file_header = ['time', 'min_h', 'min_s', 'min_v', 'max_h', 'max_s', 'max_v', 'comments']
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
        self.num_holes = int(args.num_holes)
        self.list_holes = [[0,0] for i in range(self.num_holes)]
        self.list_index = 0
        self.isHoleReady = False

        
    def saveRectangle(self, action, x, y, flags, *userdata):
        #print("Instance called")
        #print("Hello")
        if action == cv2.EVENT_LBUTTONDOWN:
            print("Top Left coordinate is saved.")
            self.list_holes[self.list_index][0] = [(x,y)]
        elif action == cv2.EVENT_LBUTTONUP:
            self.list_holes[self.list_index][1] = [(x,y)]
            self.list_index += 1
            print(f"Top Left and Bottom Right Coordinates are now set !!!")


    def drawRectangle(self, window, frame):
        if self.top_left_corner != None and self.bottom_right_corner != None:
            cv2.rectangle(frame, self.top_left_corner[0], self.bottom_right_corner[0], (0, 255, 0), 2, 8)
            cv2.imshow(window, frame)
        else: pass

    def drawAllRectangle(self, window, frame, coordinates):
        for i, coordinate in enumerate(coordinates):
            cv2.rectangle(frame, coordinate[0][0], coordinate[1][0], (0, 255, 0), 2, 8)
            cv2.imshow(window, frame)


    def chooseHoles(self, window, frame, num_holes = 0):
        # Need to generate empty list with the len of num_holes
        self.isHoleReady = False
        if self.mouse_flag == 1:
            if self.list_index < self.num_holes:
                cv2.setMouseCallback(window, self.saveRectangle) 
                #print(f"self.list_index: {self.list_index}")
                #print(f"self.mouse_flag: {self.mouse_flag}")
            else:
                self.list_index = 0
                self.mouse_flag = 0
                cv2.setMouseCallback(window, lambda x, y, flags, *userdata: None) # Disable MouseCallback function
                self.isHoleReady = True
                print("Hole Segmentation is done")
        



    def run(self):
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
                
                fg_frame = self.back_sub.apply(frame)
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                frame = resizeWithAspectRatio(frame, height=self.window_height)
                #cv2.putText(frame, fps,(7, 70), self.font, 3, (100, 255, 0), 3, cv2.LINE_AA)

                cv2.imshow(color_tracker_window, frame)
                if self.mouse_flag == 1:
                    self.chooseHoles(color_tracker_window, frame, num_holes = self.num_holes)
                else: cv2.setMouseCallback(color_tracker_window, lambda x, y, flags, *userdata: None)

                # if self.mouse_flag == 1:
                #     if self.list_index < self.num_holes + 1:
                #         cv2.setMouseCallback(color_tracker_window, self.saveRectangle) 
                #         print(f"self.mouse_flag: {self.mouse_flag}")
                #     else:
                #         self.list_index = 0
                #         self.moust_flag = 0
                #         print("Hole Segmentation is done")
                # else: cv2.setMouseCallback(color_tracker_window, lambda x, y, flags, *userdata: None) # Disable MouseCallback function
                if self.isHoleReady == True:
                    self.drawAllRectangle(color_tracker_window, frame, self.list_holes)
                #self.drawRectangle(color_tracker_window, frame)

                # Set the frame number from video
                #cv2.setTrackbarPos("Frame", color_tracker_window, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)))

                
                #fg_frame_viz = resizeWithAspectRatio(fg_frame, height=self.window_height)
                #cv2.imshow(foreground_window, fg_frame_viz)
                #min_values = np.array([self.min_h, self.min_s, self.min_v], dtype = "uint8")
                #max_values = np.array([self.max_h, self.max_s, self.max_v], dtype = "uint8")

                #masked_frame = cv2.inRange(hsv_frame, min_values, max_values)
                #filtered_hsv_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask = masked_frame)
                #filtered_rgb_frame = cv2.cvtColor(filtered_hsv_frame, cv2.COLOR_HSV2BGR)
                #subtracted_rgb_frame = cv2.bitwise_and(filtered_rgb_frame, filtered_rgb_frame, mask = fg_frame)
                
                #cv2.putText(filtered_rgb_frame, fps,(7, 70), self.font, 3, (100, 255, 0), 3, cv2.LINE_AA)
                #filtered_rgb_frame = resizeWithAspectRatio(filtered_rgb_frame, height = self.window_height)
                #cv2.imshow(filtered_window, filtered_rgb_frame)
                
                #subtracted_rgb_frame = resizeWithAspectRatio(subtracted_rgb_frame, height = self.window_height)
                #cv2.putText(subtracted_rgb_frame, fps,(7, 70), self.font, 3, (100, 255, 0), 3, cv2.LINE_AA)
                #cv2.imshow(subtracted_window, subtracted_rgb_frame)


                # Keyboard inputs
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif cv2.waitKey(1) & 0xFF == ord('s'):
                    hsv_comment = input("Enter the comment for this hsv range: ")
                    hsv_range_data = [f'{datetime.datetime.utcnow()}', f'{self.min_h}', f'{self.min_s}', f'{self.min_v}',
                                      f'{self.max_h}', f'{self.max_s}', f'{self.max_v}', f'{hsv_comment}']
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

                # elif cv2.waitKey(1) & 0xFF == ord('p'):
                #     print(f"Hue: {self.h}, Saturation: {self.s}, Value: {self.v}")

        self.capture.release()
        cv2.destroyAllWindows()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Type of input source. Option: [camera, video]")
    parser.add_argument("--video_loc", help="Location of video only required when the input source is video.")
    parser.add_argument("--num_holes", help="Specifing number of holes needed to be segmented")
    args = parser.parse_args()

    color_tracker = ColorTracker(args)
    color_tracker.run()

