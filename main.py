from cscore import CameraServer
from networktables import NetworkTables

import cv2
import json
import numpy as np
import time

detect_delay = 10

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
    # Convert to HSV and threshold image
    hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    binary_img = cv2.inRange(hsv_img, hsv_min, hsv_max)
    kernel = np.ones((8, 8), np.uint8)
    binary_img = cv2.erode(binary_img, kernel, iterations=1)

    contour_list, hierachy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        if cv2.contourArea(max(contour_list, key=cv2.contourArea)) < 25:
            raise Exception
        cnt = max(contour_list, key=cv2.contourArea)
    except Exception as e:
        return output_img, None, None, None, None
    x, y, w, h = cv2.boundingRect(cnt)
    return output_img, x, y, w, h

def main():
    # define a video capture object
    with open('/boot/frc.json') as f:
        config = json.load(f)
    camera = config['cameras'][0]

    width = camera['width']
    height = camera['height']

    CameraServer.startAutomaticCapture()

    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)

    # Table for vision output information
    NetworkTables.initialize(server='10.25.6.2')
    vision_nt = NetworkTables.getTable('Vision')

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

    # Wait for NetworkTables to start
    time.sleep(0.5)

    while True:


        final_object = 0
        final_x = 0
        final_y = 0
        final_w = 0
        final_h = 0

        start_time = time.time()

        frame_time, input_img = input_stream.grabFrame(img)

        # Notify output of error and skip iteration
        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue

        # cone
        output_img, x, y, w, h = object_detection(input_img, (16, 119, 93), (97, 255, 255))
        # (16, 176, 101), (97, 255, 255)

        if x != None:
            detected += 1
            if detected > 30:
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 255), 3)
                print("CONE:")
                print('x: ', x)
                print('y: ', y)
                print('w: ', w)
                print('h: ', h)

        prev_x = x
        prev_y = y
        prev_w = w
        prev_h = h


        output_img, x, y, w, h = object_detection(output_img, (107, 98, 28), (163, 242, 229))

        if x != None:
            detected += 1
            if detected > detect_delay:
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 255), 3)
                print("CUBE:")
                print('x: ', x)
                print('y: ', y)
                print('w: ', w)
                print('h: ', h)

        if prev_x == None and x == None:
            detected = 0

        processing_time = time.time() - start_time
        fps = 1 / processing_time
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        if detected > detect_delay:
            if prev_h:
                prev_area = prev_h * prev_w
            else:
                prev_area = 0

            if h:
                area = h * w
            else:
                area = 0
            if max(prev_area, area) is area:
                final_object = 1
                final_x = x
                final_y = y
                final_w = w
                final_h = h
                print("CUBE closer")
                cv2.putText(output_img, "CUBE closer", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            else:
                final_object = 2
                final_x = prev_x
                final_y = prev_y
                final_w = prev_w
                final_h = prev_h
                print("CONE closer")
                cv2.putText(output_img, "CONE closer", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        # target doc:
        # x, y returns decimal from left and top (ex 0.25, 0.25)
        # w, h, returns decimal from x,y to end of object
        # if x + w == 0.5 robot/camera is facing the object directly and can drive straight to reach object
        # the bigger w or h is the closer the object is
        vision_nt.putNumberArray('target', [final_object, final_x, final_y, final_w, final_h])

        output_stream.putFrame(output_img)


main()
