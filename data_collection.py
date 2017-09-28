import numpy as np
import os
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

index=83

while True:
	file_name = 'training_data-{}'.format(index);
	if os.path.isfile(file_name):
		print('File found!', index)
		index += 1
	else:
		print('File does not exist!')

	break

def keyOutput(keys):
	output = [0,0,0,0,0,0,0,0]
	if 'W' in keys and 'D' in keys:
		output = w
	elif 'W' in keys and 'A' in keys:
		output = wa
	elif 'S' in keys and 'A' in keys:
		output = sa
	elif 'S' in keys and  'D' in keys:
		output = sd
	elif 'W' in keys:
		output = wd
	elif 'A' in keys:
		output = a
	elif 'S' in keys:
		output = s
	elif 'D' in keys:
		output = d
	else:
		output = nk

	return output

def main(file_name, index):
	training_data = []
	paused = False
	for i in list(range(4))[::-1]:
		print(i+1)
		time.sleep(1)

	while True:
		if(not paused):
			screen = grab_screen(region=(5,30,800, 625))
			screen = cv2.cvtColor(cv2.resize(screen, (288,162)), cv2.COLOR_BGR2RGB)
			keys = key_check()
			output = keyOutput(keys)
			training_data.append([screen, output])
	#		cv2.imshow("Test", cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
	#		if(cv2.waitKey(25) & 0xFF == ord('q')):
	#			cv2.destroyAllWindows()
	#			break
	#		print(training_data)
			if(len(training_data)%100==0):
	#			print(len(training_data))
				if(len(training_data)%500==0):
					np.save(file_name, training_data)
					print('Saved!')
					training_data=[]
					index += 1
					file_name = 'training_data-{}'.format(index)

		key = key_check()
		if('T' in key):
			if paused:
				paused = False
				print("Unpausing")
				time.sleep(1)
			else:
				paused = True
				print("Pausing")
				time.sleep(1)

main(file_name, index)
