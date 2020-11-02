import numpy as np
import os
def confusion_matrix (indiceImagen, labels_, boxes_):
	    confusion_matrix = np.zeros((2,2))
	    image_name = os.path.basename(indiceImagen)[:-4]
	    image_name_confusion = os.path.basename(indiceImagen)


	    line_count = 0

	    #file_labels = open('./Redes/YOLOv3_TensorFlow-master/labels/train.txt','r')
	    file_labels = open('./labels/train.txt','r')
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

	    	words = lines[line_count].split()
	    	true_positive = False
	    	false_positive = False
	    	true_negative = False
	    	aux_labels = mis_labels
	    	counter_labels = 0
	    	index_array = []

	    	for label in mis_labels:
	    		if int(label) != 80:
	    			index_array.append(counter_labels)
	    		counter_labels += 1

	    	mis_labels = np.delete(mis_labels, index_array)


	    	aux_boxes = np.delete(boxes_, index_array, 0)

	    	true_positive_number = 0
	    	words_number = 0
	    	for j in range(0, len(mis_labels), 1):
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
	    				if (b1-100) < x1 < (b1+100) and (b2-100) < x2 < (b2+100) and (b3-100) < x3 < (b3+100) and (b4-100) < x4 < (b4+100):

	    					true_positive_number += 1
	    					true_positive = True
	    	confusion_matrix[0][0] += true_positive_number
	    	confusion_matrix[0][1] += (len(mis_labels) - words_number) + (len(mis_labels) - true_positive_number)
	    	confusion_matrix[1][0] += words_number - true_positive_number
	    	if len(mis_labels) == 0 and words_number == 0:
	    		confusion_matrix[1][1] += 1
	    return confusion_matrix
