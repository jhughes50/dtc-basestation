#!/usr/bin/env python3

import rospy
import tf
from dtc_inference.msg import InferenceEstimate
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import ColorRGBA

import matplotlib.pyplot as plt
import numpy as np
import rtree
import utm
from functools import partial


class DTCSpatialDatabase:
    def __init__(self):
        self.spatial_index = rtree.Index()
        self.message_index = {}
        self.br = tf.TransformBroadcaster()

        self.cur_tf = {}

        # Get colors for each robot
        topics = rospy.get_param(
            "inference_topics",
            [
                "/quad/id_estimates",
                "/callisto/heart_rate",
                "/callisto/respiratory_distress",
                "/quad/respiratory_distress",
            ],
        )
        robots = [topic.split("/")[1] for topic in topics]
        cmap = plt.get_cmap("Set1")
        self.robot_cmaplist = {
            robot: cmap(i % cmap.N) for i, robot in enumerate(robots)
        }  # multiple assignments are fine as long as there is one color per robot at the end

        self.topic_shape = {
            topic: (2 if "id_estimate" in topic else 1) for topic in topics
        }

        self.inference_subs = [
            rospy.Subscriber(
                topic,
                InferenceEstimate,
                partial(DTCSpatialDatabase.inference_callback, self, robot, topic),
            )
            for robot, topic in zip(robots, topics)
        ]

        topics = rospy.get_param(
            "gps_topics",
            ["/quad/ublox/fix", "/callisto/ublox/fix", "/basestation/ublox/fix"],
        )
        self.gps_subs = [
            rospy.Subscriber(
                topic, NavSatFix, partial(DTCSpatialDatabase.gps_callback, self, topic)
            )
            for topic in topics
        ]

        self.request_size = rospy.get_param("request_size", 5)
        self.tf_listener = tf.TransformListener()
        self.request_sub = rospy.Subscriber(
            "/clicked_point", PointStamped, self.request_callback
        )

        self.marker_pub = rospy.Publisher("/marker_array", MarkerArray, queue_size=10)

    def request_callback(self, msg):
        point_request = np.array([msg.point.x, msg.point.y])

        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                "/utm", msg.header.frame_id, rospy.Time(0)
            )
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as e:
            return

        utm_request = point_request + trans[:2]
        utm_min = utm_request - self.request_size
        utm_max = utm_request + self.request_size

        idx_result = self.spatial_index.intersection(
            (utm_min[0], utm_min[1], utm_max[0], utm_max[1])
        )

        for idx in idx_result:
            rospy.loginfo(self.message_index[idx])

        return

    def inference_callback(self, robot, topic, msg):
        """Adds the inference estimate into the spatial database

        Args:
            topic - The topic that the message is arriving from
            msg - dtc_inference/InferenceEstimate

        Returns:
            None - ros callbacks do not take returns
        """
        msg_id = hash(
            "".join(
                [
                    topic,
                    msg.header.frame_id,
                    str(msg.request_header.stamp.to_nsec()),
                    str(msg.header.stamp.to_nsec()),
                    msg.type,
                ]
            )
        )

        py_msg = {
            "robot": robot,
            "type": msg.type,
            "topic": topic,
            "data": msg.data,
            "utm": utm.from_latlon(msg.fix.latitude, msg.fix.longitude),
            "error": msg.fix.position_covariance[0],
            "request_time": msg.request_header.stamp,
            "sample_time": msg.header.stamp,
        }

        self.message_index[msg_id] = py_msg
        left, bottom, right, top = (
            py_msg["utm"][0] - py_msg["error"],
            py_msg["utm"][1] - py_msg["error"],
            py_msg["utm"][0] + py_msg["error"],
            py_msg["utm"][1] + py_msg["error"],
        )

        self.spatial_index.insert(msg_id, (left, bottom, right, top))

    def gps_callback(self, topic, msg):
        location = utm.from_latlon(msg.latitude, msg.longitude)
        self.cur_tf[msg.header.frame_id] = (location[0], location[1])

    def publish_visualization(self):
        if "map" not in self.cur_tf:
            return

        self.br.sendTransform(
            (self.cur_tf["map"][0], self.cur_tf["map"][1], 0.0),
            (0.0, 0.0, 0.0, 1.0),
            rospy.Time.now(),
            "map",
            "utm",
        )

        for k, v in self.cur_tf.items():
            if k == "map":
                continue

            self.br.sendTransform(
                (v[0] - self.cur_tf["map"][0], v[1] - self.cur_tf["map"][1], 0.0),
                (0.0, 0.0, 0.0, 1.0),
                rospy.Time.now(),
                k,
                "map",
            )

        if len(self.message_index) > 0:
            marker_array = MarkerArray()

            for msg_id, entry in self.message_index.items():
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = entry["robot"]
                marker.id = int(msg_id % 2147483647)

                marker.type = 1

                marker.action = 0
                marker.scale.x = entry["error"]
                marker.scale.y = entry["error"]
                marker.scale.z = entry["error"]
                marker.pose.position.x = entry["utm"][0] - self.cur_tf["map"][0]
                marker.pose.position.y = entry["utm"][1] - self.cur_tf["map"][1]
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0

                rgba = self.robot_cmaplist[entry["robot"]]
                marker.color.r = rgba[0]
                marker.color.g = rgba[1]
                marker.color.b = rgba[2]
                marker.color.a = rgba[3]

                marker_array.markers.append(marker)

            self.marker_pub.publish(marker_array)

        return


def main():
    rospy.init_node("triage_spatial_database")

    dsd_node = DTCSpatialDatabase()
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        dsd_node.publish_visualization()
        rate.sleep()


if __name__ == "__main__":
    main()
