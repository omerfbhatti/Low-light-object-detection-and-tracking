import numpy as np
import torch
import cv2
import time
#import os
import sys
import argparse

# ByteTracking Utilities
from byte_tracker_utils import *

import warnings
warnings.filterwarnings("ignore")

#SOURCE_VIDEO = "./road_traffic_video.mp4"
#SOURCE_VIDEO = "./video_20220908_222541.mp4"
SOURCE_VIDEO = "./busan_driving_at_night.mp4"
#SOURCE_VIDEO = "./dubai_night_drive.mp4"
#SOURCE_VIDEO = "./singapore_morning_drive.mp4"

SCALE_FACTOR = 1

parser = argparse.ArgumentParser(prog = 'Vehicle Detection/Tracking in Low-Light conditions',
                    description = 'Vehicle Detection/Tracking in Low-Light conditions')

parser.add_argument('-e', '--enhancement', action='store', choices=['eganjt','zdce','mbllen','original', 'iat','sci'], 
                    default='eganjt', help="Choose the low-light image enhancement model")
parser.add_argument('-o', '--output', action='store', help="Provide the output video name")

def resize(img, downscale_factor=2):
    if downscale_factor!=1:
        width = int(img.shape[1]/downscale_factor)
        height = int(img.shape[0]/downscale_factor)
        img = cv2.resize(img, (width, height))
    return img

if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()

    # Enhancement + Detector model
    if args.enhancement=='eganjt':
        from egan.tracker_model import egan_detector
        model = egan_detector()
    elif args.enhancement=='zdce':
        from zdce.tracker_model import zdce_detector
        model = zdce_detector()
    elif args.enhancement=='mbllen':
        from mbllen.tracker_model import mbllen_detector
        model = mbllen_detector()
    elif args.enhancement=='original':
        #from base_model import base_detector
        from base_model_deepSORT import base_detector
        model = base_detector()
    elif args.enhancement=='iat':
        from iat.tracker_model import iat_detector
        model = iat_detector()
    elif args.enhancement=='sci':
        from SCI.tracker_model import sci_detector
        model = sci_detector()
    
    model = model.to(device)

    # Open video stream
    cap = cv2.VideoCapture(SOURCE_VIDEO)

    # Output Video
    if args.output is not None:
        OUTPUT_VIDEO = "./output_videos/"+args.output
        frame_width = int(cap.get(3)/SCALE_FACTOR)
        frame_height = int(cap.get(4)/SCALE_FACTOR)
        size = (frame_width, frame_height)
        vid_writer = cv2.VideoWriter(OUTPUT_VIDEO, 
                            cv2.VideoWriter_fourcc('m','p','4','v'),
                            25, size)
        print("Output video: ", OUTPUT_VIDEO)

    # Annotator for output image
    classes_to_annotate = ['Car','People','truck','Bus']
    annotator = BaseAnnotator(colors=COLORS, thickness=2, classes=classes_to_annotate)

    avg_fps = []
    frame = 1

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Can't open video file...")
            sys.exit()

        start = time.perf_counter()     # for FPS calculations

        #print("Starting Inference...")
        img = resize(img, downscale_factor=SCALE_FACTOR)       # for higher resolution videos

        result = model(img.copy(), enhance=True, return_image=True)
        detections = result['detections']
        enhanced_image = result['enhanced_image']

        if detections is not None:
            enhanced_img = annotator.annotate(image=enhanced_image,
                                detections=detections)

        
        end = time.perf_counter()
        totaltime = end-start
        fps = 1/totaltime
        avg_fps.append(fps)

        cv2.putText(enhanced_image, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.putText(enhanced_image, args.enhancement, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        output = np.concatenate((img, enhanced_image), axis=1)

        cv2.imshow("img", output)

        # Write to output video file
        if args.output is not None and (frame < 2251):
            vid_writer.write(enhanced_image)

        if (cv2.waitKey(1) & 0xFF == 27) or (frame > 2250):
            print("Average FPS: ", np.mean(avg_fps))
            break
        
        frame += 1

    cap.release()
    if args.output is not None:
        vid_writer.release()
    cv2.destroyAllWindows()