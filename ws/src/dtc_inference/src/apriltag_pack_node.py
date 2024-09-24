#!/usr/bin/env python3

import rospy

from apriltag_ros.msg import AprilTagDetectionArray
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Empty, Header
from dtc_inference.msg import InferenceEstimate

import cv2
import numpy as np
import os
import glob


class DTCApriltagPackNode:
    def __init__(self):
        self.result_pub = rospy.Publisher("tag_gps", InferenceEstimate, queue_size=10)

        gps_topic = "ublox/fix"
        self.gps_sub = rospy.Subscriber(gps_topic, NavSatFix, self.gps_callback)

        apriltag_topic = "tags"
        self.tag_sub = rospy.Subscriber(
            apriltag_topic, AprilTagDetectionArray, self.tag_callback
        )

    def gps_callback(self, msg):
        """Buffers the current message for the trigger callback

        Args:
            msg - sensor_msgs/NavSatFix

        Returns:
            None - ros callbacks do not take returns
        """
        self.cur_gps_msg = msg

    def tag_callback(self, msg):
        """Processes the buffered image if available

        Args:
            msg - apriltag_msgs/ApriltagArrayStamped

        Returns:
            None - ros callbacks do not take returns
        """

        if self.cur_gps_msg is None:
            rospy.loginfo("No gps available")
            return

        if len(msg.detections) == 0:
            rospy.loginfo("No tags detected")
            return

        ros_result = InferenceEstimate()
        ros_result.header = msg.header
        ros_result.request_header = msg.header
        ros_result.fix = self.cur_gps_msg
        ros_result.type = "tag_gps"

        ros_result.data = np.array([tag.id for tag in msg.detections]).astype(
            np.float32
        )

        self.result_pub.publish(ros_result)


def main():
    rospy.init_node("triage_apriltag_pack")

    inf_node = DTCApriltagPackNode()

    rospy.spin()


if __name__ == "__main__":
    main()
