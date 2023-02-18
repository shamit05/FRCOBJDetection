import cv2
import json
import numpy as np
import time
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

def object_detection(input_img, hsv_min, hsv_max):
    output_img = np.copy(input_img)
    height, width, c = input_img.shape
    print(height, width)
    # Convert to HSV and threshold image
    hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    binary_img = cv2.inRange(hsv_img, hsv_min, hsv_max)

    kernel = np.ones((3, 3), np.uint8)
    binary_img = cv2.erode(binary_img, kernel, iterations=1)


    contour_list, hierachy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        if cv2.contourArea(max(contour_list, key=cv2.contourArea)) < 15:
            raise Exception
        cnt = max(contour_list, key=cv2.contourArea)
    except:
        return output_img, None, None, None, None
    x, y, w, h = cv2.boundingRect(cnt)
    return output_img, x, y, w, h

def main():
    # with open('/boot/frc.json') as f:
    #    config = json.load(f)
    # camera = config['cameras'][0]
    #
    # width = camera['width']
    # height = camera['height']
    #
    # CameraServer.startAutomaticCapture()
    #
    # input_stream = CameraServer.getVideo()
    # output_stream = CameraServer.putVideo('Processed', width, height)
    #
    # # Table for vision output information
    # vision_nt = NetworkTables.getTable('Vision')
    #
    # # Allocating new images is very expensive, always try to preallocate
    # img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)
    #
    # # Wait for NetworkTables to start
    # time.sleep(0.5)

    for image_location in all_images:
        start_time = time.time()

        # frame_time, input_img = input_stream.grabFrame(img)
        # output_img = np.copy(input_img)
        #
        # # Notify output of error and skip iteration
        # if frame_time == 0:
        #    output_stream.notifyError(input_stream.getError())
        #    continue

        input_img = cv2.imread(image_location)

        input_img = ResizeWithAspectRatio(input_img, width=600)
        # cone
        output_img, x, y, w, h = object_detection(input_img, (16, 200, 73), (97, 255, 255))
        # (16, 176, 101), (97, 255, 255)

        if x:
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            print("CONE:")
            print('x: ', x)
            print('y: ', y)
            print('w: ', w)
            print('h: ', h)

        output_img, x, y, w, h = object_detection(output_img, (116, 98, 28), (163, 242, 229))

        if x:
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 255), 8)
            print("CUBE:")
            print('x: ', x)
            print('y: ', y)
            print('w: ', w)
            print('h: ', h)

        processing_time = time.time() - start_time
        fps = 1 / processing_time
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        # target doc:
        # x, y returns decimal from left and top (ex 0.25, 0.25)
        # w, h, returns decimal from x,y to end of object
        # if x + w == 0.5 robot/camera is facing the object directly and can drive straight to reach object
        # the bigger w or h is the closer the object is
        # vision_nt.putNumberArray('target_x', x / width)
        # vision_nt.putNumberArray('target_y', y / width)
        # vision_nt.putNumberArray('target_w', w / width)
        # vision_nt.putNumberArray('target_h', h / width)



        cv2.imshow('finalImg', output_img)
        cv2.waitKey(0)


main()
