# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
import time
import subprocess

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("input_image", type=str,
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
#parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    #help="The path of the weights to restore.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/best_model_Epoch_52_step_21464_mAP_0.0385_loss_17.5814_lr_3e-05",
                    help="The path of the weights to restore.")

args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

img_ori = cv2.imread(args.input_image)
if args.letterbox_resize:
    img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
else:
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple(args.new_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    image_name = os.path.basename(args.input_image)[:-4]
    img_coloured = cv2.imread('..																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																			/monoResMatch-Tensorflow-master/output/disp/raw/'+image_name+'.png')


#************************************ INTENSIDAD EN PIXEL ***********************************
    '''start = time.time()
    for i in range(len(boxes_)):
    	print("Coche: "+ str(i))
    	h, w, c = img_coloured.shape
    	x0,y0,x1,y1=boxes_[i]
    	if x0 < 0:
    		x0 = 1
    	if x1 > w:
    		x1 = w-1
    	if y0 < 0:
    		y0 = 1
    	if y1 > h:
    		y1 = h-1
    	xM =(int) (x1+x0)//2
    	yM = (int) (y1+y0)//2
    	print(img_coloured[yM,xM])
    end = time.time()
    print (start)
    print(end)
    print("Tiempo pixel " + str(end - start))

#********************************* PROMEDIO 4 ESQUINAS ***************************************

    start = time.time()
    for i in range(len(boxes_)):
    	print("Coche: "+ str(i))
    	h, w, c = img_coloured.shape
    	x0,y0,x1,y1=boxes_[i]
    	if x0 < 0:
    		x0 = 1
    	if x1 > w:
    		x1 = w-1
    	if y0 < 0:
    		y0 = 1
    	if y1 > h:
    		y1 = h-1
    	r0,g0,b0 = img_coloured[int(y0),int(x0)]
    	r1,g1,b1 = img_coloured[int(y0)-1,int(x1)-1]
    	r2,g2,b2 = img_coloured[int(y1),int(x1)-1]
    	r3,g3,b3 = img_coloured[int(y1),int(x0)]
    	promedio = [ (r0+r1+r2+r3)//4, (g0+g1+g2+g3)//4, (b0+b1+b2+b3)//4]
    	print ( promedio )
    end = time.time()
    print (start)
    print(end)
    print("Tiempo 4 esquinas " + str(end - start))

#******************************* PROMEDIO CAJA ENTERA **************************************

    start = time.time()
    for i in range(len(boxes_)):
    	print("Coche: "+ str(i))
    	h, w, c = img_coloured.shape
    	x0,y0,x1,y1=boxes_[i]
    	if x0 < 0:
    		x0 = 1
    	if x1 > w:
    		x1 = w-1
    	if y0 < 0:
    		y0 = 1
    	if y1 > h:
    		y1 = h-1
    	rTotal = 0
    	gTotal = 0
    	bTotal = 0
    	for j in range(int(x0),int(x1)):
    		for k in range(int(y0),int(y1)):
    			r, g, b = img_coloured[k,j]
    			rTotal += r
    			gTotal += g
    			bTotal += b
    	promedio = [rTotal // ((x1-x0) * (y1-y0)) , bTotal // ((x1-x0) * (y1-y0)), gTotal // ((x1-x0) * (y1-y0))]
    	print(promedio)
    end = time.time()
    print (start)
    print(end)
    print("Tiempo caja entera " +  str(end - start))

#****************************** PROMEDIO CAJA MENOR *****************************************

    start = time.time()
    for i in range(len(boxes_)):
    	print("Coche: "+ str(i))
    	h, w, c = img_coloured.shape
    	x0,y0,x1,y1=boxes_[i]
    	if x0 < 0:
    		x0 = 1
    	if x1 > w:
    		x1 = w-1
    	if y0 < 0:
    		y0 = 1
    	if y1 > h:
    		y1 = h-1
    	xM =(int) (x1+x0)//2
    	yM = (int) (y1+y0)//2
    	rTotal = 0
    	gTotal = 0
    	bTotal = 0
    	for j in range(int(xM-20),int(xM+20)):
    		for k in range(int(yM-20),int(yM+20)):
    			r, g, b = img_coloured[k,j]
    			rTotal += r
    			gTotal += g
    			bTotal += b
    	promedio = [rTotal // 1600 , bTotal //1600, gTotal // 1600]
    	print(promedio)
    end = time.time()
    print (start)
    print(end)
    print("Tiempo caja menor " + str(end - start))'''




    #subprocess.Popen(['python3', '../../space_gap.py',str(boxes_), '/home/ruben/Documents/TFG/Redes/monoResMatch-Tensorflow-master/output/disp/raw/'+image_name+'.png',str(labels_)])



    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
        #plot_one_box(img_coloured, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
    #cv2.imshow('Detection result', img_ori)
    #cv2.imwrite('detection_result.jpg', img_ori)
    cv2.imwrite("./output/"+os.path.basename(args.input_image), img_ori)
    #cv2.imwrite("./mias/"+os.path.basename(args.input_image), img_coloured)
    cv2.waitKey(0)
