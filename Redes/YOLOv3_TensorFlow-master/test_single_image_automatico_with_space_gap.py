# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
import time
import subprocess
import space_gap as sg

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
parser.add_argument("--restore_path", type=str, default="./checkpoint/best_model_Epoch_42_step_10878_mAP_0.0611_loss_11.6598_lr_3e-05",
                    help="The path of the weights to restore.")

args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)




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
    totalPixel = 0 
    totalEsquinas = 0 
    totalEntera = 0 
    totalMenor = 0 
    count = 0
    confusion_matrix = np.zeros((2,2))
    

    for indiceImagen in os.listdir(args.input_image):
    #for indiceImagen in os.listdir("/home/ruben/Documents/Imagenes/Fase_laterales/Originales/"):
	    #print("leo imagen")
	    #img_ori = cv2.imread(args.input_image)

	    img_ori = cv2.imread(args.input_image+indiceImagen)
	    if args.letterbox_resize:
	    	img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
	    else:
	    	height_ori, width_ori = img_ori.shape[:2]
	    	img = cv2.resize(img_ori, tuple(args.new_size))
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	    img = np.asarray(img, np.float32)
	    img = img[np.newaxis, :] / 255.




	    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})


	    # rescale the coordinates to the original image
	    if args.letterbox_resize:
    		boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
    		boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
	    else:
    		boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
    		boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))


	    image_name = os.path.basename(indiceImagen)[:-4]
	    image_name_confusion = os.path.basename(indiceImagen)

	    #print(scores_) 
	    #print(labels_)    		   		
	    #print(boxes_) 
	    #Looking for the image line

	    line_count = 0
	    file_labels = open('/home/ruben/Documents/TFG/Redes/YOLOv3_TensorFlow-master/labels/train.txt','r')
	    encontrado = False
	    lines = list(file_labels)
	    mis_labels = labels_
	    for line in lines:
	    	word = line.split()
	    	name = word[1].split('/')
	    	if name[len(name) - 1] == image_name_confusion:
	    		encontrado = True
	    		break
	    	line_count += 1

	    if encontrado:
	    	#print(line_count)
	    	#print('Siguiente imagen')
	    	#print(lines[line_count])	
	    	#process the labels
	    	words = lines[line_count].split()
	    	true_positive = False
	    	false_positive = False
	    	true_negative = False
	    	aux_labels = mis_labels
	    	counter_labels = 0
	    	index_array = []
	    	#print('Mis labels antes ')
	    	#print(mis_labels)

	    	#print('Mis boxes')
	    	#print(boxes_)

	    	for label in mis_labels:
	    		if int(label) != 2:
	    			index_array.append(counter_labels)
	    		counter_labels += 1

	    	mis_labels = np.delete(mis_labels, index_array)

	    	#mis_labels = aux_labels
	    	#print('Mis labels despues')
	    	#print(mis_labels)


	    	aux_boxes = np.delete(boxes_, index_array, 0)
	    	#print('Mis boxes despues')
	    	#print(aux_boxes)
	    	gapExist = sg.space_gap(str(aux_boxes), '/home/ruben/Documents/prueba/'+image_name+'.png', str(mis_labels))


	    	true_positive_number = 0
	    	words_number = 0
	    	for i in range(4, len(words), 5):
	    		if int(words[i]) == 80 and gapExist:
	    			true_positive_number += 1
	    		if int(words[i]) == 80:
	    			words_number += 1
	    	'''for j in range(0, len(mis_labels), 1):
	    		#print(words[i])
	    		true_positive = False
	    		false_positive = False
	    		true_negative = False
	    		words_number = 0
	    		for i in range (4, len(words), 5):
	    			
	    			if str(words[i]) == str(mis_labels[j]) and int(words[i]) == 80:
	    				words_number += 1
	    				x1 = int(words[i+1])
	    				x2 = int(words[i+2])
	    				x3 = int(words[i+3])
	    				x4 = int(words[i+4])
	    				b1 = int(aux_boxes[j][0])
	    				b2 = int(aux_boxes[j][1])
	    				b3 = int(aux_boxes[j][2])
	    				b4 = int(aux_boxes[j][3])
	    				#print('evaluo datos'+ str(words[i]) + str(mis_labels[j]))
	    				if (b1-100) < x1 < (b1+100) and (b2-100) < x2 < (b2+100) and (b3-100) < x3 < (b3+100) and (b4-100) < x4 < (b4+100):
	    					#print('True positive')
	    					true_positive_number += 1
	    					#print(confusion_matrix)  
	    					true_positive = True'''
	    	#print('He salido con '+ str(words_number) + ' palabras y tantos positivos' +str(true_positive_number) )
	    	confusion_matrix[0][0] += true_positive_number
	    	#confusion_matrix[0][1] += (len(mis_labels) - words_number) + (len(mis_labels) - true_positive_number)
	    	#print('Tengo '+ str(gapExist) + ' words '+str(words_number) + 'true PN '+ str(true_positive_number))
	    	if gapExist == 1:
	    		confusion_matrix[0][1] += (gapExist - words_number)
	    	confusion_matrix[1][0] += words_number - true_positive_number
	    	if gapExist == 0 and words_number == 0:
	    		confusion_matrix[1][1] += 1

	    	#print(confusion_matrix)

	    '''for (line in file_labels):
    		words = line.split()
    		for i in range(5, len(words),4):'''
    			
    			

	    '''print("box coords:")
	    print(boxes_)
	    print('*' * 30)
	    print("scores:")
	    print(scores_)
	    print('*' * 30)
	    print("labels:")
	    print(labels_)'''


	    #img_coloured = cv2.imread('/home/ruben/Documents/TFG/Redes/monoResMatch-Tensorflow-master/output/disp/raw/'+image_name+'.png')
	    #print('/home/ruben/Documents/TFG/Redes/monoResMatch-Tensorflow-master/output/disp/raw/'+image_name+'.png'+ "                      " +args.input_image+indiceImagen )
	    count += 1
	    print(count)


	#************************************ INTENSIDAD EN PIXEL ***********************************
	    '''startPixel = time.time()
	    for i in range(len(boxes_)):
	    	print("Coche: "+ str(i))
	    	h, w, c = img_coloured.shape
    		x0,y0,x1,y1=boxes_[i]
	    	if x0 <= 0:
	    		x0 = 1
	    	if x1 >= w:
	    		x1 = w-1
	    	if y0 <= 0:
	    		y0 = 1
	    	if y1 >= h:
	    		y1 = h-1
	    	xM =(int) (x1+x0)//2
	    	yM = (int) (y1+y0)//2
	    	print(img_coloured[yM,xM])
	    endPixel = time.time()
	    print("Tiempo pixel " + str(endPixel - startPixel))
	    totalPixel += endPixel - startPixel

	#********************************* PROMEDIO 4 ESQUINAS ***************************************

	    startEsquinas = time.time()
	    for i in range(len(boxes_)):
	    	print("Coche: "+ str(i))
	    	h, w, c = img_coloured.shape
	    	x0,y0,x1,y1=boxes_[i]
	    	if x0 <= 0:
	    		x0 = 1
	    	if x1 >= w:
	    		x1 = w-1
	    	if y0 <= 0:
	    		y0 = 1
	    	if y1 >= h:
	    		y1 = h-1
	    	r0,g0,b0 = img_coloured[int(y0),int(x0)]
	    	r1,g1,b1 = img_coloured[int(y0)-1,int(x1)-1]
	    	r2,g2,b2 = img_coloured[int(y1),int(x1)-1]
	    	r3,g3,b3 = img_coloured[int(y1),int(x0)]
	    	promedio = [ (r0+r1+r2+r3)//4, (g0+g1+g2+g3)//4, (b0+b1+b2+b3)//4]
	    	print ( promedio )
	    endEsquinas = time.time()
	    print("Tiempo 4 esquinas " + str(endEsquinas - startEsquinas))
	    totalEsquinas += endEsquinas - startEsquinas

	#******************************* PROMEDIO CAJA ENTERA **************************************

	    startEntera = time.time()
	    for i in range(len(boxes_)):
	    	print("Coche: "+ str(i))
	    	h, w, c = img_coloured.shape
	    	x0,y0,x1,y1=boxes_[i]
	    	if x0 <= 0:
	    		x0 = 1
	    	if x1 >= w:
	    		x1 = w-1
	    	if y0 <= 0:
	    		y0 = 1
	    	if y1 >= h:
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
	    endEntera = time.time()
	    print("Tiempo caja entera " +  str(endEntera - startEntera))
	    totalEntera += endEntera - startEntera

	#****************************** PROMEDIO CAJA MENOR *****************************************

	    startMenor = time.time()
	    for i in range(len(boxes_)):
	    	print("Coche: "+ str(i))
	    	h, w, c = img_coloured.shape
	    	x0,y0,x1,y1=boxes_[i]
	    	if x0 <= 0:
	    		x0 = 1
	    	if x1 >= w:
	    		x1 = w-1
	    	if y0 <= 0:
	    		y0 = 1
	    	if y1 >= h:
	    		y1 = h-1
	    	xM =(int) (x1+x0)//2
	    	yM = (int) (y1+y0)//2
	    	rTotal = 0
	    	gTotal = 0
	    	bTotal = 0
	    	xM_izq = xM-20
	    	xM_der = xM+20
	    	yM_sup = yM-20
	    	yM_inf = yM+20
	    	if xM_izq < x0:
	    		xM_izq = x0 
	    	if xM_der > x1:
	    		xM_der = x1 
	    	if yM_sup < y0:
	    		yM_sup = y0 
	    	if yM_inf > y1:
	    		yM_inf = y1 
	    	for j in range(int(xM_izq),int(xM_der)):
	    		for k in range(int(yM_sup),int(yM_inf)):
	    			r, g, b = img_coloured[k,j]
	    			rTotal += r
	    			gTotal += g
	    			bTotal += b
	    	promedio = [rTotal // 1600 , bTotal //1600, gTotal // 1600]
	    	print(promedio)
	    endMenor = time.time()
	    print("Tiempo caja menor " + str(endMenor - startMenor))
	    totalMenor += endMenor - startMenor'''




	    #subprocess.Popen(['python3', '../../space_gap.py',str(boxes_), '/home/ruben/Documents/TFG/Redes/monoResMatch-Tensorflow-master/output/disp/raw/'+image_name+'.png',str(labels_)])


	    for i in range(len(boxes_)):
    		x0, y0, x1, y1 = boxes_[i]
    		plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
    		#plot_one_box(img_coloured, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
	    #cv2.imshow('Detection result', img_ori)
	    #cv2.imwrite('detection_result.jpg', img_ori)
	    cv2.imwrite("./output/"+os.path.basename(indiceImagen), img_ori)
	    #cv2.imwrite("./mias/"+os.path.basename(indiceImagen), img_coloured)
	    cv2.waitKey(0)

print(confusion_matrix)
'''print("Pixel: " + str(totalPixel/39))
print("Esquinas: " + str(totalEsquinas/39))
print("Entera: " + str(totalEntera/39))
print("Menor: " + str(totalMenor/39))'''
