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
    "alertness_motion",
    "alertness_verbal",
]

class WSReceiverNode:
    def __init__(self):
        # create a run directory with a timestamp
        run_name = rospy.get_param("run_name", time.strftime("%Y%m%d_%H%M%S"))
        self.run_dir = (
            f"/home/{getpass.getuser()}/data/{run_name}/"
        )
        os.makedirs(self.run_dir, exist_ok=True)

        self.model_path = rospy.get_param(
            "model_path",
            "/mnt/dtc/perception_models/llava/llava-onevision-qwen2-7b-ov/",
        )
        self.device = rospy.get_param("device", "gpu")
        rospy.loginfo(f"Set up device {self.device}.")

        # create a file to store the seen whisper texts

        self.seen_whisper_texts_path = os.path.join(self.run_dir, "seen_whisper_texts.csv")
        # if files do not exist, create it
        if not os.path.exists(self.seen_whisper_texts_path):
            _df = pd.DataFrame(columns=["casualty_id", "whisper_id"])
            _df.to_csv(self.seen_whisper_texts_path, index=False)
            rospy.loginfo(f"Created file at {self.seen_whisper_texts_path}.")
        else:
            rospy.loginfo(f"File already exists at {self.seen_whisper_texts_path}. Continuing.")

        self.id_to_gps_path = os.path.join(self.run_dir, "id_to_gps.csv")
        if not os.path.exists(self.id_to_gps_path):
            _df = pd.DataFrame(
                columns=["casualty_id", "lat", "long", "img_path"]
            )
            _df.to_csv(self.id_to_gps_path, index=False)
            rospy.loginfo(f"Created file at {self.id_to_gps_path}.")
        else:
            rospy.loginfo(f"File already exists at {self.id_to_gps_path}. Continuing.")

        self.database_path = os.path.join(self.run_dir, "database.csv")
        if not os.path.exists(self.database_path):
            _df = pd.DataFrame(
                columns=[
                    "casualty_id",
                    "heart_rate",
                    "respiratory_rate",
                    *LABEL_CLASSES,
                ]
            )
            _df.to_csv(self.database_path, index=False)
            rospy.loginfo(f"Created file at {self.database_path}.")
        else:
            rospy.loginfo(f"File already exists at {self.database_path}. Continuing.")

        self.drone_database_path = os.path.join(self.run_dir, "drone_data.csv")
        if not os.path.exists(self.drone_database_path):
            _df = pd.DataFrame(
                columns=[
                    "casualty_id",
                    *LABEL_CLASSES,
                ]
            )
            _df.to_csv(self.drone_database_path, index=False)
            rospy.loginfo(f"Created file at {self.drone_database_path}.")
        else:
            rospy.loginfo(f"File already exists at {self.drone_database_path}. Continuing.")

        self.image_data_path = os.path.join(self.run_dir, "image_data.csv")
        if not os.path.exists(self.image_data_path):
            _df = pd.DataFrame(
                columns=["casualty_id", "img_path"]
            )
            _df.to_csv(self.image_data_path, index=False)
            rospy.loginfo(f"Created file at {self.image_data_path}.")
        else:
            rospy.loginfo(f"File already exists at {self.image_data_path}. Continuing.")
        
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
        with portalocker.Lock(self.id_to_gps_path, "r+", timeout=1):
            df = pd.read_csv(self.id_to_gps_path)

            casualty_id = msg.casuality_id.data
            if len(df[df["casualty_id"] == casualty_id]) > 0:
                return False
            
            lat = msg.gps.latitude
            long = msg.gps.longitude
            np_arr_image = np.frombuffer(msg.image.data, np.uint8)
            drone_img = cv2.imdecode(np_arr_image, cv2.IMREAD_UNCHANGED)

            img_path = os.path.join(
                self.run_dir, casualty_id, f"drone_img.png"
            )

            append_dict = {
                "casualty_id": casualty_id,
                "lat": lat,
                "long": long,
                "img_path": img_path,
            }
            # add vlm predictions to the append_dict
            df = df._append(append_dict, ignore_index=True)
            df.to_csv(self.id_to_gps_path, index=False, mode="w")

        cv2.imwrite(img_path, drone_img)

        return True

    def ground_image_callback(self, msg):
        print("Received Image Message")
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
        with portalocker.Lock(self.image_data_path, "r+", timeout=1):
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

            image_df.to_csv(self.image_data_path, index=False, mode="w")
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
        with portalocker.Lock(self.database_path, "r+", timeout=1):
            database_df = pd.read_csv(self.database_path)

        casualty_id = msg.casualty_id.data
        if len(database_df[database_df["casualty_id"] == casualty_id]) > 1:
            return False
        
        # Parse the message
        rospy.loginfo("Received Ground Message")
        whisper = msg.whisper.data
        acc_respiration_rate = msg.acconeer_respiration_rate.data
        neural_heart_rate = msg.neural_heart_rate.data
        rospy.loginfo("Successfully parsed msg.")

        # save the data into the database
        with portalocker.Lock(self.database_path, "r+", timeout=1):
            database_df = pd.read_csv(self.database_path)
            append_dict = {
                "casualty_id": casualty_id,
                "heart_rate": neural_heart_rate,
                "respiratory_rate": acc_respiration_rate,
            }
            database_df = database_df._append(append_dict, ignore_index=True)
            database_df.to_csv(self.database_path, index=False, mode="w")

        # save the whisper text
        with portalocker.Lock(self.seen_whisper_texts_path, "r", timeout=1):
            seen_whisper_texts_df = pd.read_csv(self.seen_whisper_texts_path)
        
        # check the smallest id for the whisper text for the casualty id
        if len(seen_whisper_texts_df[seen_whisper_texts_df["casualty_id"] == casualty_id]) > 0:
            smallest_id = seen_whisper_texts_df["whisper_id"].min()
        else:
            smallest_id = 0

        if smallest_id < 2:
            os.makedirs(os.path.join(self.run_dir, str(casualty_id)), exist_ok=True) 
            
            if "Timeout: No speech detected" in whisper:
                whisper = " "

            # save whisper text to a .txt file
            with open(os.path.join(self.run_dir, str(casualty_id), f"whisper_{smallest_id}.txt"), "w") as f:
                f.write(whisper)
            rospy.loginfo("Successfully saved whisper text.")

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
