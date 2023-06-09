# Low-light-object-detection-and-tracking

Make a folder and name it "weights". Then copy the relevant low-light enhancement model and corresponding yolov5 weights file into the "weights" folder.
The relevant file download links are provided below.

| Model   | download link |
| :---:   | :---:         |
| EGAN-JT | [link](https://drive.google.com/file/d/10qJoa9k6wxfO2GphREKiEdbNtlDqecLM/view?usp=sharing) |
| Zero-DCE | [link](https://drive.google.com/file/d/1Kl983rRWquTNziR4hRgV5dEGxMn1zwKH/view?usp=sharing) |


For detecting videos: open detect.py file and update the source video filename.

USAGE: python3 detect.py -e ENHANCEMENT_MODEL -o OUTPUT_FILENAME.mp4
The library currently supports EGAN, Zero-DCE, MBLLEN and IAT image enhancement models.

HELP: python3 detect.py --help
