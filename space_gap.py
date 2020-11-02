import sys
import cv2
import math
import os

def space_gap(boxes, image_path,labels):
	
	image = cv2.imread(image_path)



	labels = labels.replace('[', '')
	labels = labels.replace(']', '')
	labelsList = labels.split()

	lines = boxes.splitlines()

	for i in range(len(lines)):
		lines[i] = lines[i].replace('[', '')
		lines[i] = lines[i].replace(']', '')

	for i in range(len(lines)-1):
		if int(labelsList[i]) != 2 or int(labelsList[i+1]) != 2:
			continue 
		x0,y0,x1,y1 = lines[i].split()
		x0_2,y0_2,x1_2,y1_2 = lines[i+1].split()
		h, w, c = image.shape
		x0 = int(float(x0))
		x1 = int(float(x1))
		y0 = int(float(y0))
		y1 = int(float(y1))
		x0_2 = int(float(x0_2))
		x1_2 = int(float(x1_2))
		y0_2 = int(float(y0_2))
		y1_2 = int(float(y1_2))
		if x0 <= 0:
			x0 = 1
		if x1 >= w:
			x1 = w-1
		if y0 <= 0:
			y0 = 1
		if y1 >= h:
			y1 = h-1
		if x0_2 <= 0:
			x0_2 = 1
		if x1_2 >= w:
			x1_2 = w-1
		if y0_2 <= 0:
			y0_2 = 1
		if y1_2 >= h:
			y1_2 = h-1


		rTotal = 0
		gTotal = 0
		bTotal = 0
		for j in range(x0,x0+40):
			for k in range(((y1+y0)//2)  , ((y1+y0)//2) + 40):
				r, g, b = image[k,j]
				rTotal += r
				gTotal += g
				bTotal += b
		promedio_cajaIzq_car1 = [rTotal // 1600 , bTotal //1600, gTotal // 1600]

		rTotal = 0
		gTotal = 0
		bTotal = 0
		for j in range(x1-40,x1):
			for k in range(((y1+y0)//2) , ((y1+y0)//2) + 40):
				r, g, b = image[k,j]
				rTotal += r
				gTotal += g
				bTotal += b
		promedio_cajaDer_car1 = [rTotal // 1600 , bTotal //1600, gTotal // 1600]
	
		rTotal = 0
		gTotal = 0
		bTotal = 0
		contador = 0
		for j in range(x1_2-40,x1_2):
			for k in range(((y1_2+y0_2)//2) , ((y1_2+y0_2)//2) + 40):
				contador +=1
				r, g, b = image[k,j]
				rTotal += r
				gTotal += g
				bTotal += b
		promedio_cajaDer_car2 = [rTotal // 1600 , bTotal //1600, gTotal // 1600]

		diferencia_car1 = [abs(promedio_cajaIzq_car1[0] - promedio_cajaDer_car1[0]), abs(promedio_cajaIzq_car1[1] - promedio_cajaDer_car1[1]),abs(promedio_cajaIzq_car1[2] - promedio_cajaDer_car1[2])]
		diferencia_car2 = [abs(promedio_cajaIzq_car1[0] - promedio_cajaDer_car2[0]), abs(promedio_cajaIzq_car1[1] - promedio_cajaDer_car2[1]),abs(promedio_cajaIzq_car1[2] - promedio_cajaDer_car2[2])]


	
		distancia = (2.5 *53) /  promedio_cajaIzq_car1[0]
	
		if(x1 > x0_2):
			espacio_en_pixeles = abs(x0 - x1_2)
		else:
			espacio_en_pixeles = abs(x1 - x0_2)


		distancia1 = (1 * 255) / promedio_cajaDer_car1[0]
		distanciaCoche2 = (1 * 255) / promedio_cajaDer_car2[0]


		if(abs(diferencia_car2[0] - diferencia_car1[0]) > 20):
			print('En la imagen ' + os.path.basename(image_path) + ' hay hueco para aparcar')
			return 1
	

	return 0


