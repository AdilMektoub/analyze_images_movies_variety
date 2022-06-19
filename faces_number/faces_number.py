# # import numpy as np
# # import cv2
# #
# # face_cascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')
# # image = cv2.imread('/home/sarah/Documents/epitech/big_data_2/faces_number/tt0085244.jpg')
# # grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # faces = face_cascade.detectMultiScale(grayImage)
# # print (type(faces))
# # print (faces)
# #Import necessary packages
#
# import cv2
# import mediapipe as mp
#
# # Define mediapipe Face detector
#
# face_detection = mp.solutions.face_detection.FaceDetection(0.8)
#
# def detector(frame):
#     count = 0
#     height, width, channel = frame.shape
#     # Convert frame BGR to RGB colorspace
#     imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # Detect results from the frame
#     result = face_detection.process(imgRGB)
#     # print(result.detections)
#     # Extract information from the result
#     # If some detection is available then extract those information from result
#     try:
#           for count, detection in enumerate(result.detections):
#               # print(detection)
#               # Extract Score and bounding box information
#               score = detection.score
#               box = detection.location_data.relative_bounding_box
#               # print(score)
#               # print(box)
#               x, y, w, h = int(box.xmin*width), int(box.ymin * height), int(box.width*width),
#               int(box.height*height)
#               score = str(round(score[0]*100, 2))
#               print(x, y, w, h)
#               # Draw rectangles
#               cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#               cv2.rectangle(frame, (x, y), (x+w, y-25), (0, 0, 255), -1)
#               cv2.putText(frame, score, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255), 2)
#           count += 1
#           print("Found ",count, "Faces!")
#       # If detection is not available then pass
#     except:
#         pass
#     return count, frame
#
# img = cv2.imread ('/faces_number/images/tt0085121.jpg')
# detector(img)
# count, output = detector(img)

