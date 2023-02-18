import cv2
import numpy as np
import glob
all_images = (glob.glob("**/*.jpg", recursive=True))

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def nothing(x):
    pass


# Set default value for Max HSV trackbars
(a, b, c), (d, e, f) = (16, 200, 73), (97, 255, 255)
# (16, 200, 73), (97, 255, 255)
for image_location in all_images:
    # Load image
    image = cv2.imread(image_location)
    image = ResizeWithAspectRatio(image, width=600)
    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)


    cv2.setTrackbarPos('HMin', 'image', a)
    cv2.setTrackbarPos('SMin', 'image', b)
    cv2.setTrackbarPos('VMin', 'image', c)

    cv2.setTrackbarPos('HMax', 'image', d)
    cv2.setTrackbarPos('SMax', 'image', e)
    cv2.setTrackbarPos('VMax', 'image', f)


    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        (a, b, c), (d, e, f) = (hMin, sMin, vMin), (hMax, sMax, vMax)

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    print(image_location)
    print("(%d, %d, %d), (%d, %d, %d)" % (hMin, sMin, vMin, hMax, sMax, vMax))

    cv2.destroyAllWindows()