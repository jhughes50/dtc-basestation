#!/usr/bin/env python3

# supress warnings
import warnings

warnings.filterwarnings("ignore")

import time
import portalocker
import getpass

import rospy
from dtc_inference.msg import ReceivedImageData
from std_msgs.msg import String, Int8
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
    "alertness_motor",
    "alertness_verbal",
]

class WSReceiverNode:
    def __init__(self):
        # create a run directory with a timestamp
        run_name = rospy.get_param("~run_name", time.strftime("%Y%m%d_%H%M%S"))
        self.run_dir = (
            f"/home/{getpass.getuser()}/data/{run_name}/"
        )
        os.makedirs(self.run_dir, exist_ok=True)

        self.model_path = rospy.get_param(
            "model_path",
            "/mnt/dtc/perception_models/llava/llava-onevision-qwen2-7b-ov/",
        )
        self.device = rospy.get_param("device", "gpu")
        rospy.loginfo(f"REC: Set up device {self.device}.")
        
        robot_name = rospy.get_param("~ground_robot")

        # create subscriber to drone and ground
        self.drone_sub = rospy.Subscriber(
            "/dione/sync/aerial_detections", TDDetection, self.drone_callback
        )
        self.ground_sub = rospy.Subscriber(
            "/" + robot_name + "/ground_detection", GroundDetection, self.ground_detection_callback
        )
        self.ground_image = rospy.Subscriber(
            "/" + robot_name + "/ground_image", GroundImage, self.ground_image_callback
        )
        rospy.loginfo(f"REC: Created subscribers to drone and ground.")

        self.signal_publisher = rospy.Publisher(
            "/received_signals", Int8, queue_size=2
        )
        self.image_path_publisher = rospy.Publisher(
            "/received_images", ReceivedImageData, queue_size=2
        )
        rospy.loginfo(f"REC: Created publishers for signals and images.")

        self._create_databases()

        # publish the run directory
        path_publisher = rospy.Publisher(
            "/run_dir", String, queue_size=2
        )
        rospy.sleep(rospy.Duration(3))
        msg = String()
        msg.data = self.run_dir
        path_publisher.publish(msg)
        rospy.loginfo(f"REC: Published run directory at {self.run_dir}.")

    def _create_databases(self):
        self.whisper_data_path = os.path.join(self.run_dir, "whisper_data.csv")
        if not os.path.exists(self.whisper_data_path):
            _df = pd.DataFrame(columns=["casualty_id", "whisper_id", "whisper_text"])
            _df.to_csv(self.whisper_data_path, index=False)
            rospy.loginfo(f"REC: Created file at {self.whisper_data_path}.")
        else:
            rospy.loginfo(f"REC: File already exists at {self.whisper_data_path}. Continuing.")

        self.aerial_image_data_path = os.path.join(self.run_dir, "aerial_image_data.csv")
        if not os.path.exists(self.aerial_image_data_path):
            _df = pd.DataFrame(
                columns=["casualty_id", "img_path"]
            )
            _df.to_csv(self.aerial_image_data_path, index=False)
            rospy.loginfo(f"REC: Created file at {self.aerial_image_data_path}.")
        else:
            rospy.loginfo(f"REC: File already exists at {self.aerial_image_data_path}. Continuing.")

        self.ground_image_data_path = os.path.join(self.run_dir, "ground_image_data.csv")
        if not os.path.exists(self.ground_image_data_path):
            _df = pd.DataFrame(
                columns=[
                    "casualty_id",
                    "img_path",
                ]
            )
            _df.to_csv(self.ground_image_data_path, index=False)
            rospy.loginfo(f"REC: Created file at {self.ground_image_data_path}.")
        else:
            rospy.loginfo(f"REC: File already exists at {self.ground_image_data_path}. Continuing.")

        self.signal_database_path = os.path.join(self.run_dir, "signal_data.csv")
        if not os.path.exists(self.signal_database_path):
            _df = pd.DataFrame(
                columns=[
                    "casualty_id",
                    "heart_rate",
                    "respiratory_rate",
                ]
            )
            _df.to_csv(self.signal_database_path, index=False)
            rospy.loginfo(f"REC: Created file at {self.signal_database_path}.")
        else:
            rospy.loginfo(f"REC: File already exists at {self.signal_database_path}. Continuing.")
        
    def drone_callback(self, msg):
        """Callback that is triggrered when drone sends a TDDetection.

        Args:
            msg (TDDetection): Message containing the drone's detection.

        Returns:
            bool: Whether or not new instance was added.
        """
        rospy.loginfo("REC: Received Drone Image Message. Processing...")
        with portalocker.Lock(self.aerial_image_data_path, "r+", timeout=1):
            aerial_image_df = pd.read_csv(self.aerial_image_data_path)

            casualty_id = msg.casualty_id
            if len(aerial_image_df[aerial_image_df["casualty_id"] == casualty_id]) > 0:
                return False
            
            np_arr_image = np.frombuffer(msg.image.data, np.uint8)
            drone_img = cv2.imdecode(np_arr_image, cv2.IMREAD_UNCHANGED)
            rospy.loginfo("REC: Successfully decoded drone image.")

            img_path = os.path.join(
                self.run_dir, str(casualty_id), f"drone_img.png"
            )

            append_dict = {
                "casualty_id": casualty_id,
                "img_path": img_path,
            }
            # add vlm predictions to the append_dict
            new_row = pd.DataFrame([append_dict])
            aerial_image_df = pd.concat([aerial_image_df, new_row], ignore_index=True)
            aerial_image_df.to_csv(self.aerial_image_data_path, index=False)
            rospy.loginfo("REC: Appended to aerial_image_data database.")

        os.makedirs(os.path.join(self.run_dir, str(casualty_id)), exist_ok=True)
        cv2.imwrite(img_path, drone_img)

        return True

    def ground_image_callback(self, msg):
        rospy.loginfo(f"REC: Received Ground Image Message from {msg.header.frame_id}")
        casualty_id = msg.casualty_id.data

        # TODO: adjust this with jason
        np_arr_image = np.frombuffer(msg.image1.data, np.uint8)
        ground_img_1 = cv2.imdecode(np_arr_image, cv2.IMREAD_UNCHANGED)
        rospy.loginfo("REC: Successfully decoded img 1.")
        np_arr_image = np.frombuffer(msg.image2.data, np.uint8)
        ground_img_2 = cv2.imdecode(np_arr_image, cv2.IMREAD_UNCHANGED)
        rospy.loginfo("REC: Successfully decoded img 2.")
        np_arr_image = np.frombuffer(msg.image3.data, np.uint8)
        ground_img_3 = cv2.imdecode(np_arr_image, cv2.IMREAD_UNCHANGED)
        rospy.loginfo("REC: Successfully decoded img 3.")
        all_ground_images = [ground_img_1, ground_img_2, ground_img_3]

        # load all previous images
        with portalocker.Lock(self.ground_image_data_path, "r+", timeout=1):
            image_df = pd.read_csv(self.ground_image_data_path)
            
            num_images_for_id = 3
            if casualty_id in image_df["casualty_id"].values:
                num_images_for_id = len(image_df[image_df["casualty_id"] == casualty_id]) + 3
            rospy.loginfo(f"REC: Received images {num_images_for_id - 3} to {num_images_for_id}.")

            if num_images_for_id > 6:
                rospy.loginfo("REC: Received more than 6 images for same ID. Skipping.")
                return False
            
            img_paths = [os.path.join(
                self.run_dir, str(casualty_id), f"ground_img_{i}.png") for i in range(num_images_for_id - 3, num_images_for_id)
            ]
            os.makedirs(os.path.join(self.run_dir, str(casualty_id)), exist_ok=True)

            for img_path, ground_img in zip(img_paths, all_ground_images):
                # save the image
                cv2.imwrite(img_path, ground_img)
                # write into the image df
                append_df = pd.DataFrame(
                    [{
                        "casualty_id": casualty_id,
                        "img_path": img_path,
                    }]
                )
                image_df = pd.concat([image_df, append_df], ignore_index=True)
                image_df.to_csv(self.ground_image_data_path, index=False) 
            rospy.loginfo(f"REC: Saved images and appended to ground_image_data database.")
        
        msg = ReceivedImageData()
        msg.casualty_id = casualty_id
        msg.image_path_list = img_paths
        self.image_path_publisher.publish(msg)
        rospy.loginfo("REC: Published image path message.")

        return True

    def ground_detection_callback(self, msg):
        """Callback that is triggrered when ground robot sends a GroundDetection.

        Args:
            msg (GroundDetection): Message containing gps and the detection values
        """

        # Parse the message
        rospy.loginfo(f"REC: Received Ground Detection Message from {msg.header.frame_id}")
        whisper = msg.whisper.data
        acc_respiration_rate = msg.acconeer_respiration_rate.data
        neural_heart_rate = msg.neural_heart_rate.data
        rospy.loginfo("REC: Successfully parsed Message content.")

        with portalocker.Lock(self.signal_database_path, "r+", timeout=1):
            signal_database_df = pd.read_csv(self.signal_database_path)

            casualty_id = msg.casualty_id.data
            if len(signal_database_df[signal_database_df["casualty_id"] == casualty_id]) > 1:
                return False
            else:
                append_dict = {
                    "casualty_id": casualty_id,
                    "heart_rate": neural_heart_rate,
                    "respiratory_rate": acc_respiration_rate,
                }
                append_df = pd.DataFrame([append_dict])
                signal_database_df = pd.concat([signal_database_df, append_df], ignore_index=True)
                signal_database_df.to_csv(self.signal_database_path, index=False)
                rospy.loginfo("REC: Appended to signal database.")

        # save the whisper text
        with portalocker.Lock(self.whisper_data_path, "r", timeout=1):
            seen_whisper_texts_df = pd.read_csv(self.whisper_data_path)
        
            # check the smallest id for the whisper text for the casualty id
            if len(seen_whisper_texts_df[seen_whisper_texts_df["casualty_id"] == casualty_id]) > 0:
                smallest_id = seen_whisper_texts_df["whisper_id"].min() + 1
            else:
                smallest_id = 0

            if smallest_id < 2:
                os.makedirs(os.path.join(self.run_dir, str(casualty_id)), exist_ok=True) 
                
                if "Timeout: No speech detected" in whisper:
                    whisper = " "

                # save whisper text to a .txt file
                with open(os.path.join(self.run_dir, str(casualty_id), f"whisper_{smallest_id}.txt"), "w") as f:
                    f.write(whisper)
                rospy.loginfo("REC: Saved whisper text.")

            # append the whisper text to the whisper data
            append_dict = {
                "casualty_id": casualty_id,
                "whisper_id": smallest_id,
                "whisper_text": whisper,
            }
            append_df = pd.DataFrame([append_dict])
            seen_whisper_texts_df = pd.concat([seen_whisper_texts_df, append_df], ignore_index=True)
            seen_whisper_texts_df.to_csv(self.whisper_data_path, index=False)
            rospy.loginfo("REC: Appended to whisper database.")

        msg = Int8()
        msg.data = casualty_id
        self.signal_publisher.publish(msg)
        
        return True

def main():
    rospy.init_node("ground_receiver")
    inf_node = WSReceiverNode()

    rospy.spin()

if __name__ == "__main__":
    main()
