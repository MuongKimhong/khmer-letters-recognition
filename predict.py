import numpy as np
import cv2

capture = cv2.VideoCapture(0)

while True:
    # capture frame by frame
    ret, frame = capture.read()
    
    # Gray video
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_video = cv2.flip(gray_video, 1)

    # Black and white video
    (thresh, black_white_video) = cv2.threshold(gray_video, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    im_floodfill = black_white_video.copy()

    dimension = {'height': black_white_video.shape[0], 'width': black_white_video.shape[1]}
    mask = np.zeros((dimension['height'] + 2, dimension['width'] + 2), np.uint8) # size needs to be 2 pixel than the image

    cv2.floodFill(im_floodfill, mask, (0, 0), 255) # Floodfill from point (0, 0)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill) # Invert floodfilled image

    # White video
    white_video = black_white_video | im_floodfill_inv # Combine the two images to get the foreground

    cv2.imshow('gray', gray_video)
    cv2.imshow('black white', black_white_video)
    cv2.imshow('white', white_video)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()