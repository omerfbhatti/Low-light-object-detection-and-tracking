import cv2
import sys
import numpy as np

SOURCE_VIDEO1 = "./output_videos/busan_original_output.mp4"
SOURCE_VIDEO2 = "./output_videos/busan_eganjt_output.mp4"
SOURCE_VIDEO3 = "./output_videos/busan_zdce_output.mp4"
SOURCE_VIDEO4 = "./output_videos/busan_iat_output.mp4"

if __name__=="__main__":
    cap1 = cv2.VideoCapture(SOURCE_VIDEO1)
    cap2 = cv2.VideoCapture(SOURCE_VIDEO2)
    cap3 = cv2.VideoCapture(SOURCE_VIDEO3)
    cap4 = cv2.VideoCapture(SOURCE_VIDEO4)

    OUTPUT_VIDEO = "./output_videos/"+"combined_output.mp4"
    frame_width = int(cap1.get(3)*2)
    frame_height = int(cap1.get(4)*2)
    size = (frame_width, frame_height)
    vid_writer = cv2.VideoWriter(OUTPUT_VIDEO, 
                        cv2.VideoWriter_fourcc('m','p','4','v'),
                        25, size)

    while cap1.isOpened():
        success, img1 = cap1.read()
        if not success:
            print("Can't open video file...")
            sys.exit()
        success, img2 = cap2.read()
        success, img3 = cap3.read()
        success, img4 = cap4.read()

        output1, output2 = np.concatenate((img1,img2), axis=1), np.concatenate((img3,img4), axis=1)
        output = np.concatenate((output1,output2), axis=0)

        cv2.imshow("output", output)
        vid_writer.write(output)

    vid_writer.release()
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cv2.destroyAllWindows()
