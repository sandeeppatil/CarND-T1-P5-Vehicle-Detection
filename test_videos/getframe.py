import cv2
vidcap = cv2.VideoCapture('project_video.mp4')
success,image = vidcap.read()
count = 0
while success:
  if (count%30 == 0):
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    print('Read a new frame: ', success)
  success,image = vidcap.read()
  count += 1