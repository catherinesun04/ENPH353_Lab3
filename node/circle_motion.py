#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

try:
    # Initialize the ROS node
    rospy.init_node('circle_motion', anonymous=True)
    rospy.loginfo("Circle motion node initialized")
    
    # Create the publisher
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rate = rospy.Rate(2)
    rospy.loginfo("Publisher created")

    # Create and fill the Twist message
    move = Twist()
    move.linear.x = 0.5  # Forward velocity
    move.angular.z = 1  # Angular velocity for rotation
    
    rospy.loginfo("Starting motion loop")
    while not rospy.is_shutdown():
        pub.publish(move)
        rate.sleep()

except Exception as e:
    rospy.logerr(f"An error occurred: {str(e)}")
    raise