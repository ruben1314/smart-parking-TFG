import subprocess 
import os
import sys
import time
import cv2
import argparse

def cut_in_frames(path, name_path):

	if not os.path.exists('./output/'+name_path+'/'):
        	os.makedirs('./output/'+os.path.basename(path[:-4]))


	vidcap = cv2.VideoCapture(path)
	success,image = vidcap.read()
	count = 1
	descartar = 0
	success = True
	f = open('./Redes/monoResMatch-Tensorflow-master/names','w')
	while success:
	  if descartar == 1:
	  	cv2.imwrite("./output/"+os.path.basename(path[:-4])+"/%d.jpg" % count, image)     # save frame as JPEG file
	  	f.write("%d.jpg\n" % count)
	  	count += 1
	  success,image = vidcap.read()
	  
	  
	  #if cv2.waitKey(10) == 27:                     # exit if Escape is hit
	      #break
	  descartar = (descartar + 1) % 4


	f.close()




def write_names(path):
	f = open('./Redes/monoResMatch-Tensorflow-master/names','w')
	for files in os.listdir(path):
		f.write("%s\n" % files)


parser = argparse.ArgumentParser()
parser.add_argument('--video', dest='video', type=bool, default=0,
                   help='if it is a video')
parser.add_argument('--folder', dest='folder', type=bool, default=1,
                   help='if it is a folder')
parser.add_argument('--path', dest='path', type=str,
                   help='Video path or folder path')


args = parser.parse_args()
path = args.path


if(args.video):
	name_path = os.path.basename(path[:-4])
	cut_in_frames(path, name_path)

	startMono = time.time()

	os.chdir("./Redes/monoResMatch-Tensorflow-master/")
	subprocess.call(["python3","main.py", "--output_path", "./output", "--data_path_image", "../../output/"+name_path+"/", "--filenames_file", "./names", "--checkpoint_path","./CS_K_GT_200_700_raw/model-5000","--save_image"])


elif(args.folder):
	
	write_names(path)	
	
	startMono = time.time()
	os.chdir("./Redes/monoResMatch-Tensorflow-master/")
	subprocess.call(["python3","main.py", "--output_path", "./output", "--data_path_image", path, "--filenames_file", "./names", "--checkpoint_path","./CS_K_GT_200_700_raw/model-5000","--save_image"])	



endMono = time.time()

startYOLO = time.time()

os.chdir("../YOLOv3_TensorFlow-master/")
subprocess.call(["python3", "test_single_image_automatico_classic.py", "../monoResMatch-Tensorflow-master/output/disp/raw/"])

endYOLO = time.time()

print("Tiempo red Mono" +  str(endMono - startMono)) 

print("Tiempo red YOLO" +  str(endYOLO - startYOLO)) 



