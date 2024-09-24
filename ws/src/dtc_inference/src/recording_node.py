#!/usr/bin/env python3

import rospy
from std_msgs.msg import Empty, Header
import subprocess
import time


class DTCRecordingNode:
    def __init__(self):
        self.time = rospy.get_param("time", 30)
        self.topics = rospy.get_param("topics", ["ublox/fix", "image_raw"])

        self.basestation_name = rospy.get_param("basestation_name", "basestation")
        self.robot_name = rospy.get_param("robot_name", "callisto")

        self.base_folder = rospy.get_param("base_folder", f"/tmp/{self.robot_name}")

        trigger_topic = f"/{self.basestation_name}/{self.robot_name}/trigger"
        self.trigger_sub = rospy.Subscriber(
            trigger_topic, Header, self.trigger_callback
        )

        self.last_record_time = time.time()

    def trigger_callback(self, msg):
        """Processes the buffered image if available

        Args:
            msg - std_msgs/Header

        Returns:
            None - ros callbacks do not take returns
        """

        if time.time() - self.last_record_time < self.time:
            return

        self.last_record_time = time.time()

        command = [
            "rosbag",
            "record",
            f"--duration={self.time}",
            f"--output-prefix={self.base_folder}",
        ] + self.topics
        subprocess.run(command)


def main():
    rospy.init_node("triage_recording")

    inf_node = DTCRecordingNode()

    rospy.spin()


if __name__ == "__main__":
    main()
