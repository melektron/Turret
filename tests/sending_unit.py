#!/usr/bin/python3
from picamera.array import PiRGBArray
from traceback import print_exc
from picamera import PiCamera
from threading import Thread
import RPi.GPIO as GPIO
import time

import io
import socket
import struct
import time
import picamera


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



direction = [-1, True]


def stepper_thread():
    global motor_pins, direction
    motor_step_counter = 0

    while direction[1]:
        off_a = direction[0]
        direction[0] = 0

        if off_a:
            direc = off_a < 0
            steps = int(abs(off_a))

            for _ in range(steps):
                for pin in range(0, len(motor_pins)):
                    GPIO.output(motor_pins[pin], step_sequence[motor_step_counter][pin] )
                    
                if direc:
                    motor_step_counter = (motor_step_counter - 1) % 8

                else:
                    motor_step_counter = (motor_step_counter + 1) % 8

                time.sleep( step_sleep )

            # after moving, wait for camera to reload
            time.sleep(.05)

    print("stepper exit")


# Connect a client socket to my laptop's IP address and port 8000
client_socket = socket.socket()
client_socket.connect(('192.168.5.94', 8000))

# Make a file-like object out of the connection
connection = client_socket.makefile('rwb')

try:
    Thread(target=stepper_thread).start()

    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 24
        time.sleep(2) # Let camera warm up
        start = time.time()
        stream = io.BytesIO()

        for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
            # Send the image length and data over the network
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            stream.seek(0)
            connection.write(stream.read())
            stream.seek(0)
            stream.truncate()

            time.sleep(0.01)

            # # Receive an integer from the laptop
            response = struct.unpack('<i', connection.read(struct.calcsize('<i')))[0]
            direction[0] = int(response)


finally:
    connection.close()
    client_socket.close()

    direction[1] = False
    time.sleep(1)

    cleanup()
    exit(1)

