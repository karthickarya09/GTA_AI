import numpy as np
import os
import cv2
from grabscreen import grab_screen

index=1

while True:
	file_name = 'training_data-{}'.format(index);
	if os.path.isfile(file_name):
		print('File found!', index)
	else:
		print('File does not exist!')
		break

def main():
	training_data = []

	while True:
		screen = grab_screen(region=(5,30,800, 625))
		cv2.imshow("Test", cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
		if(cv2.waitKey(25) & 0xFF == ord('q')):
			cv2.destroyAllWindows()
			break

main()
