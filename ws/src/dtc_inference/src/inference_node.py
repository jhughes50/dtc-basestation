#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Empty, Header
from dtc_inference.msg import InferenceEstimate
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import os
import glob
import omegaconf

import torch

# fmt: off
import logging
logging.basicConfig(level=logging.INFO)
rospy.log= logging.getLogger(__name__)
# fmt: on


def load_jitted_model(hydra_base_path, model_path, model_nr=None):
    """Load a jitted model for inference.

    Args:
        hydra_base_path (Union[str, Path]): Path to the hydra base directory containing config and models.
        model_path (str): Path to the model directory.
        model_nr (int, optional): Model number to load. Defaults to None. When None, the last model is loaded.

    Returns:
        nn.Module: Loaded model.
        OmegaConf: Loaded hydra config.
    """
    # get hydra config
    cfg = omegaconf.OmegaConf.load(os.path.join(hydra_base_path, ".hydra/config.yaml"))
    rospy.loginfo(f"Loaded hydra config from: {hydra_base_path}")

    # get model path
    if model_nr is None:
        # get all possible paths
        all_possible_paths = glob.glob(os.path.join(hydra_base_path, model_path, "*"))

        # now, sort the models by their model_nr
        all_possible_paths = sorted(
            all_possible_paths, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        rospy.loginfo(f"Found {len(all_possible_paths)} models in {model_path}")

        # now, get the last one
        model_file = all_possible_paths[-1]
        rospy.loginfo(f"Loading model: {model_file}")
    else:
        model_file = f"model_{model_nr}.pt"
        rospy.loginfo(f"Loading model: {model_file}")

    # load model
    # model = torch.jit.load(os.path.join(hydra_base_path, model_path, model_file))
    model = DummyModel(hydra_base_path)

    return model, cfg


def inference_wrapper(
    model,
    cfg,
    image,
    vitals,
):
    """Wrapper function for model inference.

    Args:
        model (nn.Module): Model to be used for inference.
        cfg (OmegaConf): Hydra config.
        image (torch.Tensor): Image tensor, shape (C, H, W) or (N, C, H, W).
        vitals (torch.Tensor): Vitals tensor, shape (N, D).

    Returns:
        torch.Tensor: Inference result.
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    assert (
        image.shape[1] >= cfg.model.image_enc.num_input_chan
    ), f"Incorrect number of channels: {image.shape[1]} vs {cfg.model.image_enc.num_input_chan}"
    assert (
        image.shape[2] >= cfg.model.image_enc.input_res[0]
    ), f"Incorrect height: {image.shape[2]} vs {cfg.model.image_enc.input_res[0]}"
    assert (
        image.shape[3] >= cfg.model.image_enc.input_res[1]
    ), f"Incorrect width: {image.shape[3]} vs {cfg.model.image_enc.input_res[1]}"

    if image.shape != (
        1,
        cfg.model.image_enc.num_input_chan,
        cfg.model.image_enc.input_res[0],
        cfg.model.image_enc.input_res[1],
    ):
        rospy.logwarning(
            f"Image shape {image.shape} does not match model input shape: {cfg.model.image_enc.input_res}"
            + f"Reshaping image to {cfg.model.image_enc.input_res}"
        )
        image = image[
            :,
            : cfg.model.image_enc.num_input_chan,
            cfg.model.image_enc.input_res[0],
            cfg.model.image_enc.input_res[1],
        ]

    if len(vitals.shape) == 1:
        vitals = vitals.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        result = model(image, vitals)
        model.train()

    return result


class DummyModel:
    def __init__(self, path):
        self.path = path

    def to(self, device):
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, x, y):
        return torch.rand((1, 2))


class DTCInferenceNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.base_path = rospy.get_param(
            "hydra_base_path", "/home/dcist/2024-05-28_12-32-38/s0"
        )
        self.model_path = rospy.get_param("model_path", "models")
        self.device = rospy.get_param("device", "cpu")

        self.model, self.cfg = load_jitted_model(
            self.base_path, self.model_path, model_nr=None
        )
        self.model.to(self.device)

        self.cur_image_msg = None

        self.result_pub = rospy.Publisher(
            self.cfg.data.target, InferenceEstimate, queue_size=10
        )

        image_topic = "image_raw"
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)

        gps_topic = "ublox/fix"
        self.gps_sub = rospy.Subscriber(gps_topic, NavSatFix, self.gps_callback)

        self.basestation_name = rospy.get_param("basestation_name", "basestation")
        self.robot_name = rospy.get_param("robot_name", "callisto")

        trigger_topic = f"/{self.basestation_name}/{self.robot_name}/trigger"
        self.trigger_sub = rospy.Subscriber(
            trigger_topic, Header, self.trigger_callback
        )

    def gps_callback(self, msg):
        """Buffers the current message for the trigger callback

        Args:
            msg - sensor_msgs/NavSatFix

        Returns:
            None - ros callbacks do not take returns
        """
        self.cur_gps_msg = msg

    def image_callback(self, msg):
        """Buffers the current message for the trigger callback

        Args:
            msg - sensor_msgs/Image

        Returns:
            None - ros callbacks do not take returns
        """
        self.cur_image_msg = msg

    def trigger_callback(self, msg):
        """Processes the buffered image if available

        Args:
            msg - std_msgs/Empty

        Returns:
            None - ros callbacks do not take returns
        """

        if self.cur_gps_msg is None:
            rospy.loginfo("No gps available")
            return

        if self.cur_image_msg is None:
            rospy.loginfo("No image available")
            return

        ros_result = InferenceEstimate()
        ros_result.header = self.cur_image_msg.header
        ros_result.request_header = msg
        ros_result.fix = self.cur_gps_msg
        ros_result.type = str(self.cfg.data.target)

        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(self.cur_image_msg, "bgr8")
        except CvBridgeError as e:
            # Print error and exit if not possible
            rospy.logerror(e)

            return

        torch_image = torch.from_numpy(np.transpose(cv2_img, (2, 0, 1)))[
            None, ...
        ].float()
        torch_image = torch_image[:, :, :256, :256].to(self.device)
        torch_vitals = torch.randn(1, 32).to(self.device)
        torch_result = inference_wrapper(
            self.model, self.cfg, torch_image, torch_vitals
        )

        numpy_result = torch_result.cpu().numpy()

        ros_result.data = numpy_result.ravel().astype(np.float32)

        self.result_pub.publish(ros_result)


def main():
    rospy.init_node("triage_inference")

    inf_node = DTCInferenceNode()

    rospy.spin()


if __name__ == "__main__":
    main()
