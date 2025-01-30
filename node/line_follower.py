#!/usr/bin/env python3

## @package line_follower
# @brief High-speed line following node using PID control and computer vision

import rospy
import cv2 as cv
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LineFollower:
    ## @brief Main class implementing line following logic
    def __init__(self):
        ## @brief Node initialization and parameter setup
        rospy.init_node('line_follower')
        
        # ROS Publishers and Subscribers
        ## @brief Publisher for velocity commands
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        ## @brief Subscriber for camera images
        self.image_sub = rospy.Subscriber('/rrbot/camera1/image_raw', Image, self.image_callback)
        ## @brief Publisher for processed images
        self.image_pub = rospy.Publisher('/processed_image', Image, queue_size=1)
        
        ## @brief OpenCV-ROS bridge instance
        self.bridge = CvBridge()
        
        ## @brief Twist message for movement commands
        self.move = Twist()
        
        # Color detection parameters
        ## @name HSV Thresholds
        ## @{
        self.low_H, self.high_H = 80, 120       ##< Hue range for line detection (green colors)
        self.low_S, self.high_S = 100, 255      ##< Saturation range (higher values = more vivid colors)
        self.low_V, self.high_V = 100, 255      ##< Value range (brighter lighting conditions)
        ## @}
        
        # Control parameters
        ## @name PID Control Variables
        ## @{
        self.avg_x = None         ##< Weighted average of detected line position
        self.prev_error = 0       ##< Previous error for derivative calculation
        self.integral = 0         ##< Accumulated error for integral term
        ## @}
        
        ## @name PID Coefficients
        ## @{
        self.kp = 0.03    ##< Proportional gain - direct response to position error
        self.kd = 0.025   ##< Derivative gain - dampens rapid changes/oscillations
        self.ki = 0.0001  ##< Integral gain - corrects persistent offsets
        ## @}
        
        ## @name Speed Parameters
        ## @{
        self.base_speed = .985    ##< Normal operating speed (m/s)
        self.min_speed = 0.9      ##< Minimum speed during tight turns (m/s)
        self.max_angular = .95    ##< Maximum angular velocity (rad/s)
        ## @}

        rospy.loginfo("High-Speed Line Follower Initialized")

    ## @brief Image processing callback function
    # @param msg Incoming ROS Image message
    def image_callback(self, msg):
        ## @brief Convert ROS image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]

        # Convert to HSV and threshold
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(frame_HSV, (self.low_H, self.low_S, self.low_V), 
                                      (self.high_H, self.high_S, self.high_V))
        
        # Region of Interest (ROI) parameters
        roi_height = height // 3  ##< Use bottom third of image for processing
        roi_mask = mask[-roi_height:, :]
        
        # Find contours
        contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if self.avg_x is None:
            self.avg_x = width // 2  ##< Initialize to center position

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
                error = self.avg_x - width//2  ##< Position error from centerline
                
                # Dynamic speed adjustment
                speed_factor = 1.0 - min(abs(error)/(width/2), 1.0)  ##< 1=centered, 0=edge
                current_speed = self.min_speed + (self.base_speed - self.min_speed) * speed_factor
                
                # PID control calculations
                derivative = error - self.prev_error  ##< Error change rate
                self.integral += error                ##< Error accumulation
                steering = (self.kp * error) + (self.kd * derivative) + (self.ki * self.integral)
                
                # Update previous error
                self.prev_error = error
                
                # Set velocities with limits
                self.move.linear.x = current_speed  ##< Current forward speed
                self.move.angular.z = -steering     ##< Calculated steering command
        else:
            # Lost line behavior
            self.move.linear.x = self.min_speed     ##< Reduced speed when line lost
            self.move.angular.z = self.max_angular * 0.6  ##< Search rotation
            self.prev_error = 0       ##< Reset error history
            self.integral = 0         ##< Reset accumulated error

        self.cmd_pub.publish(self.move)
        
        # Debug visualization
        cv.circle(frame, (self.avg_x, height - 30), 25, (0, 0, 255), -1)  ##< Position indicator
        cv.putText(frame, f"Speed: {self.move.linear.x:.2f} m/s", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  ##< Speed display
        cv.putText(frame, f"Steering: {self.move.angular.z:.2f} rad/s", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  ##< Steering display
        
        cv.imshow("Line Tracking", frame)
        cv.imshow("Threshold", mask)
        
        # Publish processed image
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