This program detects faces from either a static picture or a video.

detect_faces.py is the python code to detect faces in an image.
detect_faces_video.py is the python code to detect faces in a video.

Command to run:

1. To detect in a static image:

	python3 detect_faces.py --image IMAGE_NAME -p deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel 

2. To detect in a video:
	
	python3 detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --video VIDEO_NAME 