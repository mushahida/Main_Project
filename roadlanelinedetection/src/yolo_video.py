# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import math
import os
from src.sample_new import predict_image
from src.sample_z import zebra_line
labelsPath = os.path.sep.join([r"C:\Users\musha\Downloads\roadlanelinedetection\src\static\archive","coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS),3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([r"C:\Users\musha\Downloads\roadlanelinedetection\src\static\archive", "yolov3.weights"])
configPath = os.path.sep.join([r"C:\Users\musha\Downloads\roadlanelinedetection\src\static\archive", "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
###print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
# r"G:\yolo\yolo-object-detection\videos\airport.mp4"
currentposition=0
import pyttsx3
engine = pyttsx3.init()


engine.say(" Started")
engine.runAndWait()
vs = cv2.VideoCapture(r"C:\Users\musha\Downloads\roadlanelinedetection\src\Commercial Truck. This is how you safely change Lanes😂.mp4")
# vs = cv2.VideoCapture(r"pexels-kelly-4608284-3840x2160-24fps.mp4")
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	###print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	###print("[INFO] could not determine # of frames in video")
	###print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file
counti=0
voicecount=0
while True:

	counti=counti+1
	flag=0
	# read the next frame from the file

	(grabbed, frame) = vs.read()
	_, frame = vs.read()
	cv2.imwrite("sample.jpg",frame)

	r,v=predict_image("sample.jpg")
	cv2.imwrite("sampleout.jpg",r)
	stretch_near = cv2.resize(r, (780, 540),
							  interpolation=cv2.INTER_LINEAR)
	cv2.imshow("op",stretch_near)
	count=0
	line_pos=0
	cropflag=0
	cropframe=frame
	##print(v,len(v))
	##print("#############")
	ignore_mask_color = 255
	if len(v)>=4:
		# cv2.fillPoly(frame, v, ignore_mask_color)
		#print("okkkk")
		try:
			x1, y1=v[0]
			x2, y2=v[1]

			x3, y3=v[2]
			x4, y4=v[3]
			top_left_x = min([x1, x2, x3, x4])
			top_left_y = min([y1, y2, y3, y4])
			bot_right_x = max([x1, x2, x3, x4])
			bot_right_y = max([y1, y2, y3, y4])
			#print("==========================================")
			#print(top_left_x,top_left_y)
			#print(bot_right_x,bot_right_y)
			#print(frame.shape)
			width=bot_right_y-top_left_y
			print(width,"===========",frame.shape[1],frame.shape[0])

			cropframe=frame[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
			#print(bot_right_y-top_left_y)

			cv2.rectangle(frame, (top_left_y,bot_right_y+1), (top_left_x,bot_right_x+1), (0,0,255), 1)
			# cv2.imshow("rec",frame)
			print((top_left_y,bot_right_y+1), (top_left_x,bot_right_x+1))

			line_pos=top_left_y
			if top_left_x>line_pos:
				line_pos=top_left_x
			pos=frame.shape[1]-bot_right_y
			#print(pos)

			count=frame.shape[1]//(bot_right_y-top_left_y)
			#print(count)

			cc=bot_right_y//(bot_right_y-top_left_y)
			#print(cc)

			#print("==========================================")
			cv2.imwrite("sample_crop.png",cropframe)
			cropflag = 1
		except Exception as e:
			#print(e)
			cropflag = 1
	##print("cropflag",cropflag)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Apply edge detection using Canny algorithm
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)

	# Apply Hough transform to detect lines
	lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=75, minLineLength=100, maxLineGap=5)

	# Draw detected lines on the image
	minslop=-1
	maxdist=0
	maxline=[0,0,0,0]
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line[0]

			point1=(x1,y1)
			point2=(x2,y2)

			dist = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

			if dist>maxdist:
				maxdist=dist
				maxline=[x1, y1, x2, y2]


			cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
			slope = (y2 - y1) / (x2 - x1 + 1e-6)
			###print(abs(slope), "===============================")
			if minslop==-1:
				minslop=slope
			else:
				if slope<minslop:
					minslop=slope

			if abs(slope) > 100 and abs(slope) < 500:

				#print(abs(slope))
				#print("zig zag line detected")

				engine.say("Alert!!!! zig zag line detected")
				engine.runAndWait()
	print("minslop",minslop)
	print("max distance",maxdist)
	print("max distance",maxline)
	print("+++++++++++==================")
	print(frame.shape[1], frame.shape[0])
	frame_width=frame.shape[0]
	xval=maxline[0]
	if maxline[2]>xval:
		xval=maxline[2]

	ldist=abs(0-xval)
	cdist=abs((frame_width//2)-xval)
	rdist=abs(frame_width-xval)

	posi=1
	speed="40"
	if cdist<ldist and cdist<rdist:
		speed = "60"
		posi=2
	if rdist<ldist and rdist<cdist:
		posi=3
		speed = "80"
	if posi==1:
		cc=1
	if posi==2:
		if cdist>line_pos:
			speed = "60"
			cc = 2
		else:
			cc = 3
			speed = "80"
	if posi==3:
		speed = "100"
		cc = 4
	print(cc,"currentposition===============")
	if currentposition != cc:
		currentposition = cc
		engine.say("Alert!!!! you are in lane " + str(cc) + " you can go by speed of "+speed+" km/h")
		engine.runAndWait()





	###print("==> ",counti)
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(cropframe, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				##print(cropflag,"++++++++++++++++++++++")
				##print(cropflag,"++++++++++++++++++++++")
				##print(cropflag,"++++++++++++++++++++++")
				##print("====================================")
				##print("====================================")
				if cropflag==1:
					# umbrella
					if LABELS[classID]!='umbrella' and LABELS[classID]!='kite':
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top
						# and and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates,
						# confidences, and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4,
		0.5)
	##print("===",voicecount,idxs)
	# ensure at least one detection exists
	if len(idxs) > 0:
		if voicecount==0:
			if cropflag==1:
				engine.say("Alert!!!! Object detected")
				engine.runAndWait()
			# loop over the indexes we are keeping
		voicecount=voicecount+1
		if voicecount==5:
			voicecount=0
		for i in idxs.flatten():

				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(cropframe, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],
					confidences[i])
				cv2.putText(cropframe, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				cv2.imwrite(r"C:\Users\musha\Downloads\roadlanelinedetection\src\static\img\\"+str(counti)+".jpg",cropframe)

	###print("okkkk")
	frame=cv2.resize(frame, (750, 750))
	cv2.imshow('frame', frame)
	r,v=zebra_line("sample.jpg")
	if r!="na":
		engine.say("Alert!!!! "+r)
		engine.runAndWait()
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	# cv2.imwrite("output/"+str(counti)+".jpg",frame)

# release the file pointers
###print("[INFO] cleaning up...")
writer.release()
vs.release()