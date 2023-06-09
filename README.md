# Low-light-object-detection-and-tracking

Make a folder and name it "weights". Then copy the relevant low-light enhancement model and corresponding yolov5 weights file into the "weights" folder.
The relevant file download links are provided below.

| Model   | download link |
| :---:   | :---:         |
| EGAN-JT | [link](https://drive.google.com/file/d/10qJoa9k6wxfO2GphREKiEdbNtlDqecLM/view?usp=sharing) |
| Zero-DCE | [link](https://drive.google.com/file/d/1Kl983rRWquTNziR4hRgV5dEGxMn1zwKH/view?usp=sharing) |
| SCI | [link](https://drive.google.com/file/d/1BUg4ectcf2VV-BU4khpQXFCMGCsh8_Tv/view?usp=sharing) |
| IAT | [link](https://drive.google.com/file/d/1GJPQ8hgZcIGLeblM_41_MUWIERySBnW4/view?usp=sharing) |
| YOLOv5 (EGAN-JT) | [link](https://drive.google.com/file/d/1ehomjgjU28kkJhJvgtENX7i6p_zpnXgJ/view?usp=sharing) |
| YOLOv5 (low-light) | [link](https://drive.google.com/file/d/11a40xQDInFstfSz2IKTXROt7Z-5RJhB5/view?usp=sharing) |

For detecting videos: open detect.py file and update the source video filename.

USAGE: python3 detect.py -e ENHANCEMENT_MODEL -o OUTPUT_FILENAME.mp4
The library currently supports EGAN, Zero-DCE, MBLLEN and IAT image enhancement models.

HELP: python3 detect.py --help
