#!/usr/bin/env python3

# supress warnings
import warnings

warnings.filterwarnings("ignore")

import time
from math import radians, cos, sin, atan2, sqrt
import portalocker
import getpass

import rospy
from dtc_inference.msg import ReceivedSignal, ReceivedImageData
from std_msgs.msg import String
from gone.msg import GroundDetection, GroundImage
from tdd2.msg import TDDetection
from cv_bridge import CvBridge

import os

import numpy as np
import pandas as pd
import cv2

# fmt: off
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# fmt: on

LABEL_CLASSES = [
    "trauma_head",
    "trauma_torso",
    "trauma_lower_ext",
    "trauma_upper_ext",
    "alertness_ocular",
    "severe_hemorrhage",
]

class WSReceiverNode:
    def __init__(self):
        # create a run directory with a timestamp
        self.run_dir = (
            f"/home/{getpass.getuser()}/data/{time.strftime('%Y%m%d_%H%M%S')}/"
        )
        os.makedirs(self.run_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.model_path = rospy.get_param(
            "model_path",
            "/mnt/dtc/perception_models/llava/llava-onevision-qwen2-7b-ov/",
        )
        self.device = rospy.get_param("device", "gpu")
        rospy.loginfo(f"Set up device {self.device}.")

        # TODO: don't do creation here in case of crashes, this will overwrite data
        # create id to gps database
        self.id_to_gps_path = rospy.get_param(
            "id_to_gps_path", os.path.join(self.run_dir, "id_to_gps.csv")
        )
        _df = pd.DataFrame(
            columns=["casualty_id", "lat", "long", "img_path"]
        )
        _df.to_csv(self.id_to_gps_path, index=False)
        rospy.loginfo(f"Created file at {self.id_to_gps_path}.")

        # create prediction database
        self.database_path = rospy.get_param(
            "id_to_gps_path", os.path.join(self.run_dir, "database.csv")
        )
        _df = pd.DataFrame(
            columns=[
                "robot_name",
                "casualty_id",
                "lat",
                "long",
                "img_1",
                "img_2",
                "img_3",
                "hr_model",
                "hr_cv",
                "rr",
                "motion",
                "whisper",
                *LABEL_CLASSES,
            ]
        )
        _df.to_csv(self.database_path, index=False)
        rospy.loginfo(f"Created file at {self.database_path}.")

        self.image_data_path = rospy.get_param(
            "image_data_path", os.path.join(self.run_dir, "image_data.csv")
        )
        _df = pd.DataFrame(
            columns=["casualty_id", "img_path"]
        )
        _df.to_csv(self.image_data_path, index=False)
        rospy.loginfo(f"Created file at {self.image_data_path}.")

        
        robot_name = rospy.get_param("~ground_robot")
        drone_name = rospy.get_param("~aerial_robot")

        # create subscriber to drone and ground
        self.drone_sub = rospy.Subscriber(
            "/" + drone_name + "/drone_detection", TDDetection, self.drone_callback
        )
        self.ground_sub = rospy.Subscriber(
            "/" + robot_name + "/ground_detection", GroundDetection, self.ground_detection_callback
        )
        self.ground_image = rospy.Subscriber(
            "/" + robot_name + "/ground_image", GroundImage, self.ground_image_callback
        )
        rospy.loginfo(f"Created subscribers to drone and ground.")

        self.signal_publisher = rospy.Publisher(
            "/received_signals", ReceivedSignal, queue_size=2
        )
        self.image_path_publisher = rospy.Publisher(
            "/received_images", ReceivedImageData, queue_size=2
        )
        rospy.loginfo(f"Created publishers for signals and images.")

        # publish the run directory
        path_publisher = rospy.Publisher(
            "/run_dir", String, queue_size=2
        )
        rospy.sleep(rospy.Duration(3))
        msg = String()
        msg.data = self.run_dir
        path_publisher.publish(msg)
        rospy.loginfo(f"Published run directory at {self.run_dir}.")

    def drone_callback(self, msg):
        """Callback that is triggrered when drone sends a TDDetection.

        Args:
            msg (TDDetection): Message containing the drone's detection.

        Returns:
            bool: Whether or not new instance was added.
        """
        with portalocker.Lock(self.id_to_gps_path, timeout=1):
            df = pd.read_csv(self.id_to_gps_path)

        if len(df[df["casualty_id"] == msg.casualty_id]) > 0:
            return False
        
        lat = msg.gps.latitude
        long = msg.gps.longitude
        casualty_id = msg.casuality_id
        np_arr_image = np.frombuffer(msg.image.data, np.uint8)
        drone_img = cv2.imdecode(np_arr_image, cv2.IMREAD_UNCHANGED)

        img_path = os.path.join(
            self.run_dir, casualty_id, f"drone_img.png"
        )
        cv2.imwrite(img_path, drone_img)

        append_dict = {
            "casualty_id": casualty_id,
            "lat": lat,
            "long": long,
            "img_path": img_path,
        }
        # add vlm predictions to the append_dict
        df = df._append(append_dict, ignore_index=True)

        with portalocker.Lock(self.id_to_gps_path, timeout=1):
            df.to_csv(self.id_to_gps_path, index=False)

        return True

    def ground_image_callback(self, msg):
        print("Received Image Message")
        timestamp = msg.header.stamp
        robot_name = msg.header.frame_id
        casualty_id = msg.casualty_id.data

        # TODO: adjust this with jason
        np_arr_image = np.frombuffer(msg.image1.data, np.uint8)
        ground_img_1 = cv2.imdecode(np_arr_image, cv2.IMREAD_UNCHANGED)
        rospy.loginfo("Successfully decoded img 1.")
        np_arr_image = np.frombuffer(msg.image2.data, np.uint8)
        ground_img_2 = cv2.imdecode(np_arr_image, cv2.IMREAD_UNCHANGED)
        rospy.loginfo("Successfully decoded img 2.")
        np_arr_image = np.frombuffer(msg.image3.data, np.uint8)
        ground_img_3 = cv2.imdecode(np_arr_image, cv2.IMREAD_UNCHANGED)
        rospy.loginfo("Successfully decoded img 3.")
        all_ground_images = [ground_img_1, ground_img_2, ground_img_3]

        # load all previous images
        with portalocker.Lock(self.image_data_path, timeout=1):
            image_df = pd.read_csv(self.image_data_path)
            
        num_images_for_id = 3
        if casualty_id in image_df["casualty_id"].values:
            num_images_for_id = len(image_df[image_df["casualty_id"] == casualty_id]) + 3
        rospy.loginfo(f"Received {num_images_for_id} images for casualty ID {casualty_id}.")

        if num_images_for_id > 6:
            rospy.loginfo("Received more than 6 images for same casualty ID. Skipping.")
            return False
        
        img_paths = [os.path.join(
            self.run_dir, str(casualty_id), f"ground_img_{i}.png") for i in range(num_images_for_id - 3, num_images_for_id)
        ]
        os.makedirs(os.path.join(self.run_dir, str(casualty_id)), exist_ok=True)

        for img_path, ground_img in zip(img_paths, all_ground_images):
            # write into the image df
            image_df = image_df._append({"casualty_id": casualty_id, "img_path": img_path}, ignore_index=True)
            cv2.imwrite(img_path, ground_img)
            rospy.loginfo(f"Added new image to Dataframe and saved image at {img_path}.")

        with portalocker.Lock(self.image_data_path, timeout=1):
            image_df.to_csv(self.image_data_path, index=False)
        rospy.loginfo("Successfully wrote all images to files.")            
        
        msg = ReceivedImageData()
        msg.casualty_id = casualty_id
        msg.image_path_list = img_paths
        self.image_path_publisher.publish(msg)
        rospy.loginfo("Successfully published image message.")

        return True

    def ground_detection_callback(self, msg):
        """Callback that is triggrered when ground robot sends a GroundDetection.

        Args:
            msg (GroundDetection): Message containing gps and the detection values
        """
        start_time_callback = time.time()

        with portalocker.Lock(self.database_path, timeout=1):
            database_df = pd.read_csv(self.database_path)

        casualty_id = msg.casualty_id
        if len(database_df[database_df["casualty_id"] == casualty_id]) > 1:
            return False
        
        # Parse the message
        rospy.loginfo("Received Ground Message")
        gps = msg.gps
        whisper = msg.whisper.data
        acc_respiration_rate = msg.acconeer_respiration_rate.data
        event_respiration_rate = msg.event_respiration_rate.data
        neural_heart_rate = msg.neural_heart_rate.data
        cv_heart_rate = msg.cv_heart_rate.data
        robot_name = msg.header.frame_id
        rospy.loginfo("Successfully parsed msg.")

        append_dict = {
            "robot_name": robot_name,
            "casualty_id": casualty_id,
            "lat": gps.latitude,
            "long": gps.longitude,
            "neural_heart_rate": neural_heart_rate,
            "cv_heart_rate": cv_heart_rate,
            "event_respiration_rate": event_respiration_rate,
            "acc_respiration_rate": acc_respiration_rate,
            "whisper": whisper,
        }       

        database_df = database_df._append(append_dict, ignore_index=True)

        with portalocker.Lock(self.database_path, timeout=1):
            database_df.to_csv(self.database_path, index=False)
        rospy.loginfo("Successfully wrote to database.")

        msg = ReceivedSignal()
        msg.casualty_id = casualty_id
        msg.heart_rate = neural_heart_rate
        msg.respiratory_rate = acc_respiration_rate
        self.signal_publisher.publish(msg)
        
        return True

def main():
    rospy.init_node("ground_receiver")
    inf_node = WSReceiverNode()

    rospy.spin()

if __name__ == "__main__":
    main()
