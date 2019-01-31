#!/usr/bin/python3
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example program shows how you can use dlib to make a HOG based object
# detector for things like faces, pedestrians, and any other semi-rigid
# object.  In particular, we go though the steps to train the kind of sliding
# window object detector first published by Dalal and Triggs in 2005 in the
# paper Histograms of Oriented Gradients for Human Detection.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import os
import sys
import glob
import cv2
import dlib



# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector = dlib.simple_object_detector("detector.svm")

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

# Now let's run the detector over the images in the faces folder and display the
# results.

print("starting wbcam")
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    dets = detector(img)
    print("Number of faces detected: {}".format(len(dets)))
   
    # Display the resulting image
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
        left = d.left()
        right = d.right()
        bottom =d.bottom()
        top = d.top()
        print(left)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 4)

    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break









