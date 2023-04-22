#!/usr/bin/python3
from picamera.array import PiRGBArray
from traceback import print_exc
from picamera import PiCamera
from threading import Thread
import RPi.GPIO as GPIO
import numpy as np
import math
import time
import cv2


in1 = 4
in2 = 17
in3 = 23
in4 = 24

# careful lowering this, at some point you run into the mechanical limitation of how quick your motor can move
step_sleep = 0.002

step_count = 4096 # 5.625*(1/64) per step, 4096 steps is 360°

direction = False # True for clockwise, False for counter-clockwise

# defining stepper motor sequence (found in documentation http://www.4tronix.co.uk/arduino/Stepper-Motors.php)
step_sequence = [[1,0,0,1],
                 [1,0,0,0],
                 [1,1,0,0],
                 [0,1,0,0],
                 [0,1,1,0],
                 [0,0,1,0],
                 [0,0,1,1],
                 [0,0,0,1]]

# setting up
GPIO.setmode( GPIO.BCM )
GPIO.setup( in1, GPIO.OUT )
GPIO.setup( in2, GPIO.OUT )
GPIO.setup( in3, GPIO.OUT )
GPIO.setup( in4, GPIO.OUT )

# initializing
GPIO.output( in1, GPIO.LOW )
GPIO.output( in2, GPIO.LOW )
GPIO.output( in3, GPIO.LOW )
GPIO.output( in4, GPIO.LOW )

motor_pins = [in1,in2,in3,in4]

def cleanup():
    GPIO.output( in1, GPIO.LOW )
    GPIO.output( in2, GPIO.LOW )
    GPIO.output( in3, GPIO.LOW )
    GPIO.output( in4, GPIO.LOW )
    GPIO.cleanup()


# --------------------- CV init -------------------

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)


# Define the range of red color in HSV
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])


lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])


direction = [-1, True]


def stepper_thread():
    global motor_pins, direction
    motor_step_counter = 0

    while direction[1]:
        off_a = direction[0]
        direction[0] = 0

        if off_a:
            off_steps = (off_a / 360) * step_count
            direc = off_steps < 0
            steps = int(abs(off_steps))

            print(f"off by {off_a}°. moving {steps} steps")

            for _ in range(steps):
                for pin in range(0, len(motor_pins)):
                    GPIO.output(motor_pins[pin], step_sequence[motor_step_counter][pin] )
                    
                if direc:
                    motor_step_counter = (motor_step_counter - 1) % 8

                else:
                    motor_step_counter = (motor_step_counter + 1) % 8

                time.sleep( step_sleep )

            # after moving, wait for camera to reload
            time.sleep(.3)

    print("stepper exit")


def pixel_to_angle(y, image_height, fov, mount_angle):
    cy = image_height / 2
    theta = math.atan((y - cy) * math.tan(fov/2) / cy)
    return math.degrees(theta + mount_angle)

# the meat
try:
    Thread(target=stepper_thread).start()
    i = 0

    # capture frames from the camera
    print("starting capture")
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        start = time.perf_counter()
        img = frame.array

        hsv = cv2.cvtColor(cv2.UMat(img), cv2.COLOR_BGR2HSV)


        # mask stuff
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Create cv2.UMat objects from numpy arrays
        mask1_umat = cv2.UMat(mask1)
        mask2_umat = cv2.UMat(mask2)

        # Combine the masks
        mask_umat = cv2.bitwise_or(mask1_umat, mask2_umat)

        # Convert the cv2.UMat object back to numpy array
        mask = mask_umat.get()

        # Apply morphology operations to remove noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)


        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour, which should correspond to the red circle
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x,y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)

            # circle to small
            if radius < 20:
                direction[0] = 0

            else:
                # Draw the circle and its center
                cv2.circle(img, center, radius, (0, 255, 0), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)

                # set direction
                image_height = 480  # Image height in pixels
                fov = math.radians(60)  # Vertical field of view in radians

                theta = pixel_to_angle(center[1], image_height, fov, 0)
                direction[0] = theta

        else:
            direction[0] = 0

        # perform some processing on the image
        # display the image
        cv2.imshow("Frame", img)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        print(f"took {(time.perf_counter() - start) * 1000:.2f}ms")

except KeyboardInterrupt:
    direction[1] = False

    time.sleep(1)

    cleanup()
    exit(1)

except Exception:
    # cleanup
    cv2.destroyAllWindows()

    direction[1] = False
    time.sleep(1)

    cleanup()

    print_exc()

finally:
    # cleanup
    cv2.destroyAllWindows()

    direction[1] = False
    time.sleep(1)

    cleanup()
    exit(0)
