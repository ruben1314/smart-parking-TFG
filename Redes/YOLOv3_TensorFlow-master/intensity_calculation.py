import numpy as np
import time

def pixel_intensity(boxes_, img_coloured):
	#************************************ INTENSIDAD EN PIXEL ***********************************
	    startPixel = time.time()
	    for i in range(len(boxes_)):
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
	    	print(img_coloured[xM,yM])
	    endPixel = time.time()
	    print("Tiempo pixel " + str(endPixel - startPixel))


def corners_intensity(boxes_,img_coloured):
	#********************************* PROMEDIO 4 ESQUINAS ***************************************
	    totalEsquinas = 0
	    startEsquinas = time.time()
	    for i in range(len(boxes_)):
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
	    	print(promedio)
	    endEsquinas = time.time()
	    print("Tiempo 4 esquinas " + str(endEsquinas - startEsquinas))
	    totalEsquinas += endEsquinas - startEsquinas


def full_box(boxes_, img_coloured):
	#******************************* PROMEDIO CAJA ENTERA **************************************
	    totalEntera = 0 
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

def little_box(boxes_, img_coloured):
	#****************************** PROMEDIO CAJA MENOR *****************************************
	    totalMenor = 0
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
	    totalMenor += endMenor - startMenor
