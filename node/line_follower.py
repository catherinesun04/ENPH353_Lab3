#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower', anonymous=True)
        self.bridge = CvBridge()
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber('/rrbot/camera1/image_raw', Image, self.image_callback)
        self.move_cmd = Twist()

        # HSV parameters (calibrate these using lab2.ipynb)
        self.lower_hsv = np.array([90, 100, 20])
        self.upper_hsv = np.array([120, 255, 200])
        self.kernel = np.ones((5, 5), np.uint8)

        # Visualization windows
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Tracking Mask', cv2.WINDOW_NORMAL)
        rospy.loginfo("Line follower node initialized")

    def image_callback(self, data):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            display_image = cv_image.copy()
            height, width = cv_image.shape[:2]

            # Process bottom third of the image
            roi = cv_image[2*height//3:, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Create mask
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

            # Find largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00']) + 2*height//3  # Convert to full image coordinates

                    # Draw tracking visuals
                    cv2.circle(display_image, (cx, cy), 10, (0, 0, 255), -1)
                    cv2.line(display_image, (width//2, height), (cx, cy), (0, 255, 0), 2)

                    # Simple steering control
                    error = cx - width//2
                    self.move_cmd.linear.x = 0.3
                    self.move_cmd.angular.z = -error * 0.005  # Simple proportional control
                else:
                    self.move_cmd.linear.x = 0
                    self.move_cmd.angular.z = 0
            else:
                self.move_cmd.linear.x = 0
                self.move_cmd.angular.z = 0

            self.vel_pub.publish(self.move_cmd)

            # Display images
            cv2.imshow('Camera Feed', display_image)
            cv2.imshow('Tracking Mask', mask)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error: {str(e)}")

    def cleanup(self):
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.vel_pub.publish(self.move_cmd)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        follower = LineFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        follower.cleanup()