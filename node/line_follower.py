#! /usr/bin/env python3

import rospy
import cv2 as cv
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower')
        
        # ROS Publishers and Subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber('/rrbot/camera1/image_raw', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/processed_image', Image, queue_size=1)
        
        # OpenCV Bridge
        self.bridge = CvBridge()
        
        # Movement Command
        self.move = Twist()
        
        # Tuned HSV Limits (adjust for your track)
        self.low_H, self.high_H = 80, 120
        self.low_S, self.high_S = 100, 255  # Increased saturation range
        self.low_V, self.high_V = 100, 255  # Brighter value range

        # Control parameters
        self.avg_x = None
        self.prev_error = 0
        self.integral = 0
        
        # PID coefficients
        self.kp = 0.03    # Increased proportional gain
        self.kd = 0.025   # Derivative term for anticipatory control
        self.ki = 0.0001  # Small integral term

        # Speed parameters
        self.base_speed = .95     # Increased base speed
        self.min_speed = 0.6      # Minimum speed for tight turns
        self.max_angular = 1     # Maximum turning rate

        rospy.loginfo("High-Speed Line Follower Initialized")

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]

        # Convert to HSV and threshold
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(frame_HSV, (self.low_H, self.low_S, self.low_V), 
                                      (self.high_H, self.high_S, self.high_V))
        
        # Use larger ROI (bottom 1/3 of image)
        roi_height = height // 3
        roi_mask = mask[-roi_height:, :]
        
        # Find contours
        contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if self.avg_x is None:
            self.avg_x = width // 2

        if contours:
            # Calculate weighted average of contour centroids
            total_x = 0
            total_area = 0
            for cnt in contours:
                M = cv.moments(cnt)
                if M['m00'] > 0:
                    total_x += M['m10']
                    total_area += M['m00']
            
            if total_area > 0:
                self.avg_x = int(total_x / total_area)
                error = self.avg_x - width//2
                
                # Dynamic speed adjustment
                speed_factor = 1.0 - min(abs(error)/(width/2), 1.0)
                current_speed = self.min_speed + (self.base_speed - self.min_speed) * speed_factor
                
                # PID control
                derivative = error - self.prev_error
                self.integral += error
                steering = (self.kp * error) + (self.kd * derivative) + (self.ki * self.integral)
                
                # Update previous error
                self.prev_error = error
                
                # Set velocities with limits
                self.move.linear.x = current_speed
                self.move.angular.z = -steering
        else:
            # Lost line - slow down and search
            self.move.linear.x = self.min_speed
            self.move.angular.z = self.max_angular * 0.6
            self.prev_error = 0
            self.integral = 0

        self.cmd_pub.publish(self.move)
        
        # Display Debug Info
        cv.circle(frame, (self.avg_x, height - 30), 25, (0, 0, 255), -1)
        cv.putText(frame, f"Speed: {self.move.linear.x:.2f} m/s", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f"Steering: {self.move.angular.z:.2f} rad/s", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.imshow("Line Tracking", frame)
        cv.imshow("Threshold", mask)
        
        # Publish Processed Image
        processed_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_pub.publish(processed_image_msg)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User quit")

if __name__ == '__main__':
    try:
        LineFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv.destroyAllWindows()