import cv2
import numpy as np
import argparse

color_tracker_window = "Video Capture"
filtered_window = "Filtered Image"
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

        self.min_h = low_H
        self.min_s = low_S
        self.min_v = low_V

        self.max_h = high_H
        self.max_s = high_S
        self.max_v = high_V

    def run(self):
        while True:#
        # Image Collection & Processing
            
            ret, frame = self.capture.read()
            #frame = cv2.GaussianBlur(frame,(11,11), cv2.BORDER_DEFAULT)
            if ret:
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Read from Track bar
                self.min_h = cv2.getTrackbarPos('min_h', track_bar)
                self.min_s = cv2.getTrackbarPos('min_s', track_bar)
                self.min_v = cv2.getTrackbarPos('min_v', track_bar)
                self.max_h = cv2.getTrackbarPos('max_h', track_bar)
                self.max_s = cv2.getTrackbarPos('max_s', track_bar)
                self.max_v = cv2.getTrackbarPos('max_v', track_bar)
                
                frame = resizeWithAspectRatio(frame, height=self.window_height)
                cv2.imshow(color_tracker_window, frame)

                min_values = np.array([self.min_h, self.min_s, self.min_v], dtype = "uint8")
                max_values = np.array([self.max_h, self.max_s, self.max_v], dtype = "uint8")

                masked_frame = cv2.inRange(hsv_frame, min_values, max_values)
                filtered_hsv_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask = masked_frame)
                filtered_rgb_frame = cv2.cvtColor(filtered_hsv_frame, cv2.COLOR_HSV2BGR)
                filtered_rgb_frame = resizeWithAspectRatio(filtered_rgb_frame, height = self.window_height)
                
                cv2.imshow(filtered_window, filtered_rgb_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # if cv2.waitKey(1) & 0xFF == ord('p'):
                #     print(f"Hue: {self.h}, Saturation: {self.s}, Value: {self.v}")

        self.capture.release()
        cv2.destroyAllWindows()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Type of input source. Option: [camera, video]")
    parser.add_argument("--video_loc", help="Location of video only required when the input source is video.")
    args = parser.parse_args()

    color_tracker = ColorTracker(args)
    color_tracker.run()

