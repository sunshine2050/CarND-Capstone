from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime
import cv2


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        lower_red = np.array([230, 0, 0])
        upper_red = np.array([255, 200, 255])
    
        # Threshold the rgb image to get only red colors
        red_mask = cv2.inRange(img, lower_red, upper_red)    
        # Bitwise-AND mask and original image
        img = cv2.bitwise_and(img,img, mask= red_mask)
        cnt = 0
        for layer in img:
            cnt += cv2.countNonZero(layer)
        if cnt > 400:
            return TrafficLight.RED


        return TrafficLight.UNKNOWN
