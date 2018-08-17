import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
import glob
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from collections import deque

class CTracker:
    def __init__(self, filename, history):
        
        # load a pe-trained svc model from a serialized (pickle) file
        dist_pickle = pickle.load( open(filename, "rb" ) )

        # get attributes of our svc object
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle["scaler"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.hist_bins = dist_pickle["hist_bins"]
        self.history = history
        self.buffer = deque([])
        self.vidBuffer = deque([])

        print(self.X_scaler)
        print(self.orient)
        print(self.pix_per_cell)
        print(self.cell_per_block)
        print(self.spatial_size)
        print(self.hist_bins)

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
        
    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def getBinLabels(self, lab):
        lab[lab>0] = 1
        return lab

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        draw_img = np.copy(img)
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return draw_img

    def getLabeledBoxes(self, labels):

        boxlist = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            if abs(bbox[0][0]-bbox[1][0])*abs(bbox[0][1]-bbox[1][1]) > 2000:
                boxlist.append(bbox)

        return boxlist

    def find_cars(self, img):
        roi_list = [
            # [(0,390),(1280,646), 1.5],
            # #[(sx,sy), (ex,ey), scale]
            # [(0,400),(1280,400+72), 0.5],
            # [(0,390),(1280,390+144), 1],
            # [(0,380),(1280,380+192), 1.5],
            # [(0,370),(1280,370+224), 2],
            # [(0,370),(1280,370+280), 2.5]
            [(0,390),(1280,510), 1.5],
            [(8,398),(1280,518), 1.5],
            [(0,412),(1280,660), 2]
        ]

        heat = np.zeros_like(img[:,:,0]).astype(np.float)

        for roi in roi_list:
            boxlist = self.getBoxes(img, roi)

            # Add heat to each box in box list
            heat = self.add_heat(heat,boxlist)
                
        return heat

    def getBoxes(self, img, roi):
        
        img = img.astype(np.float32)/255
        
        img_tosearch = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0], :]
        ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YCrCb')
        if roi[2] != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/roi[2]), np.int(imshape[0]/roi[2])))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        
        box_list = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features = color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                test_prediction = self.svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*roi[2])
                    ytop_draw = np.int(ytop*roi[2])
                    win_draw = np.int(window*roi[2])
                    box_list.append([(xbox_left+roi[0][0], ytop_draw+roi[0][1]),(xbox_left+roi[0][0]+win_draw,ytop_draw+win_draw+roi[0][1])])

        return box_list
  
    def __call__(self, in_img):
        if self.history == True: #if video frames
            img = cv2.cvtColor(in_img, cv2.COLOR_RGB2BGR)
        else:
            img = in_img
        heatmap = self.find_cars(img)
        # print("max heat: ", np.max(heatmap))
        heatmap = self.apply_threshold(heatmap,2)
        heatmap = np.clip(heatmap, 0, 255)
        lab = label(heatmap)
        boxList = self.getLabeledBoxes(lab)
        if self.history == True:
            if (len(self.buffer)>25):
                self.buffer.popleft()
            #get bin labels
            binLab = self.getBinLabels(lab[0])
            #append to buffer
            self.buffer.append(binLab)
            #take average of buffer
            avLab = np.sum(self.buffer, axis=0)
            # print("max sum: ", np.max(avLab))
            #apply threshold of .7 for 15 frames per second video
            avLab = self.apply_threshold(avLab, 9)
            #calculate label of labels
            lab = label(avLab)
            #get boxlist from new label in boxList
            boxList = self.getLabeledBoxes(lab)
        out_img = self.draw_boxes(in_img, boxList)

        return(out_img)

#work on still images
imgList = glob.glob("../test_images/*.jpg")
track = CTracker("svc_pickle.p", history=False)
for file in imgList:
    img = cv2.imread(file)
    out_img = track(img)
    outfilename = "../output_images/"+file.split("/")[-1]
    cv2.imwrite(outfilename, out_img)
    print(outfilename)

#work on video files
# vidfile = "../test_videos/test_video.mp4"
vidfile = "../test_videos/project_video.mp4"
track = CTracker("svc_pickle.p", history=True)
clip1 = VideoFileClip(vidfile)#.subclip(21,31)
# clip1 = VideoFileClip(vidfile).set_fps(15)
white_clip = clip1.fl_image(track)
# white_clip.write_videofile("../output_videos/test_video.mp4", audio=False)
white_clip.write_videofile("../output_videos/project_video.mp4", audio=False)
print("\a")
