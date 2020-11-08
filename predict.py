import numpy as np
import time
import cv2

capture = cv2.VideoCapture(0)

while True:
    # capture frame by frame
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.flip(gray_frame, 1)

    # Black and white video
    (thresh, black_white_frame) = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    floodfill_frame = black_white_frame.copy()

    dimension = {'height': black_white_frame.shape[0], 'width': black_white_frame.shape[1]}
    mask = np.zeros((dimension['height'] + 2, dimension['width'] + 2), np.uint8) # size needs to be 2 pixel than the image

    cv2.floodFill(floodfill_frame, mask, (0, 0), 255) # Floodfill from point (0, 0)
    floodfill_inv_frame = cv2.bitwise_not(floodfill_frame) # Invert floodfilled image

    # White video
    white_frame = black_white_frame | floodfill_inv_frame # Combine the two images to get the foreground

    cv2.imshow("frame", frame)
    cv2.imshow('gray', gray_frame)
    cv2.imshow('black white', black_white_frame)
    cv2.imshow('white', white_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()