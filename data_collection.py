import numpy as np
import os
import cv2
from grabscreen import grab_screen

index=1

while True:
	file_name = 'training_data-{}'.format(index);
	if os.path.isfile(file_name):
		print('File found!', index)
		index += 1
	else:
		print('File does not exist!')

	break

def main(file_name, index):
	training_data = []

	while True:
		screen = grab_screen(region=(5,30,800, 625))
		training_data.append(cv2.cvtColor(cv2.resize(screen, (480,270)), cv2.COLOR_BGR2GRAY))
#		cv2.imshow("Test", cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
#		if(cv2.waitKey(25) & 0xFF == ord('q')):
#			cv2.destroyAllWindows()
#			break
#	print(training_data)
		if(len(training_data)%100==0):
			print(len(training_data))
			if(len(training_data)%500==0):
				np.save(file_name, training_data)
				print('Saved!')
				training_data=[]
				index += 1
				file_name = 'training_data-{}'.format(index)

main(file_name, index)
