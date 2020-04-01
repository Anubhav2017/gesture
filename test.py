import tensorflow as tf
import numpy as np
import cv2

model=tf.keras.models.load_model("gesture_model")

cam=cv2.VideoCapture(0)
cv2.namedWindow("test")
categories=["down","palm","l","fist","fist_moved","thumb","index","ok","palm_moved","c"]

while(True):
	ret,frame=cam.read()
	

	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (320, 120))
	cv2.imshow("test",img)
	img=np.reshape(img,(1,120,320,1))
	#print(img.shape)
	y=model.predict(img)
	print(y)
	ind=(np.argmax(y))
	print(categories[ind])

	k=cv2.waitKey(100)

	if k%27 ==0:
		print("Escape pressed")
		break
	#print(k)

cam.release()

cv2.destroyAllWindows()


