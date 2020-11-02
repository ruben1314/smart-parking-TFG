import xml.etree.ElementTree as ET
import os

def buscar_label(label):
	contador = 0
	label = label 
	labels = open('./Redes/YOLOv3_TensorFlow-master/data/coco_names.txt','r')
	lines = labels.read().splitlines()
	for line in lines:
		if line == label:
			return contador
		contador += 1

count = 0


train_file = open(os.path.abspath('../TFG/txts/train.txt'),'w')
val_file = open(os.path.abspath('../TFG/txts/val.txt'),'w')

for objects in os.listdir('./xmls/'):
	string = str(count)
	tree = ET.parse('./xmls/'+objects)
	root = tree.getroot()
	string = string + ' ./train_dataset/'+root.find('filename').text
	for element in root.findall('size'):
		string = string + ' '+element.find('width').text +' '+ element.find('height').text

	for element in root.findall('object'):
		numero = buscar_label(element.find('name').text)
		string = string + ' ' + str(numero)
		for element1 in element.findall('bndbox'):
			string = string + ' ' + element1.find('xmin').text+ ' ' + element1.find('ymin').text+ ' ' + element1.find('xmax').text+ ' ' + element1.find('ymax').text
			

	train_file.write(string+'\n')
	val_file.write(string+'\n')
	count += 1




train_file.close()
val_file.close()
