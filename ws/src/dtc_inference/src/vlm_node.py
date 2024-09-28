#!/usr/bin/env python3

# supress warnings
import warnings

warnings.filterwarnings("ignore")

import importlib
import copy
import json
import time
import portalocker
import getpass
import glob

import rospy
from http_streamer import HttpStreamer
from dtc_inference.msg import ReceivedImageData
from std_msgs.msg import String, Int8

import os

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    process_images,
)

from llava.conversation import conv_templates

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)

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

LABEL_MEANINGS_INT_TO_STR = {
    "trauma_head": {
        0: "absence",
        1: "presence",
    },
    "trauma_torso": {
        0: "absence",
        1: "presence",
    },
    "trauma_lower_ext": {
        0: "normal",
        1: "wound",
        2: "amputation",
    },
    "trauma_upper_ext": {
        0: "normal",
        1: "wound",
        2: "amputation",
    },
    "alertness_ocular": {
        0: "open",
        1: "closed",
        2: "untestable",
    },
    "severe_hemorrhage": {
        0: "absence",
        1: "presence",
    },
    "alertness_motor": {
        0: "normal",
        1: "abnormal",
        2: "absence",
        3: "untestable",
    },
    "alertness_verbal": {
        0: "normal",
        1: "abnormal",
        2: "absence",
        3: "untestable",
    }
}

LABEL_MEANINGS_STR_TO_INT = {
    "trauma_head": {
        "absence": 0,
        "presence": 1,
    },
    "trauma_torso": {
        "absence": 0,
        "presence": 1,
    },
    "trauma_lower_ext": {
        "normal": 0,
        "wound": 1,
        "amputation": 2,
    },
    "trauma_upper_ext": {
        "normal": 0,
        "wound": 1,
        "amputation": 2,
    },
    "alertness_ocular": {
        "open": 0,
        "closed": 1,
        "untestable": 2,
    },
    "severe_hemorrhage": {
        "absence": 0,
        "presence": 1,
    },
    "alertness_motor": {
        "normal": 0,
        "abnormal": 1,
        "absence": 2,
        "untestable": 3,
    },
    "alertness_verbal": {
        "normal": 0,
        "abnormal": 1,
        "absence": 2,
        "untestable": 3,
    },
}


def parse_label_dict_int_to_str(label_dict):
    """Parse a label dictionary from integer to string values

    Args:
        label_dict (dict): A dictionary with keys as the label names and
            values as the integer labels

    Returns:
        dict: A dictionary with keys as the label names and values as the
            string labels
    """
    for key in label_dict.keys():
        label_dict[key] = LABEL_MEANINGS_INT_TO_STR[key][label_dict[key]]
    return label_dict


def parse_label_dict_str_to_int(label_dict):
    """Parse a label dictionary from string to integer values

    Args:
        label_dict (dict): A dictionary with keys as the label names and
            values as the string labels

    Returns:
        dict: A dictionary with keys as the label names and values as the
            integer labels
    """
    for key in label_dict.keys():
        label_dict[key] = LABEL_MEANINGS_STR_TO_INT[key][label_dict[key]]
    return label_dict

def parse_dict_response(response, label_class):
    """Parse the response from the VLM to a dictionary.
       Runs several checks to ensure the response is in the correct format.

    Args:
        response (str): The response from the VLM. Hopefully
            this contains a dictionary.

    Returns:
        dict: The parsed response.
    """
    # first, figure out where the dictionary starts and ends
    start_idx = response.find("{")
    end_idx = response.find("}")
    if start_idx == -1 or end_idx == -1:
        start_idx = response.find("```python")
        if start_idx == -1:
            logger.info(f"Response does not contain a dictionary. Response: {response}")
            return "The response does not contain a dictionary. Please provide a response in the format requested."
        else:
            # if the response contains code but no dictionary, we check if the code contains the right key: value pair. E.g.
            # ```python
            # trauma_torso: 'absence'
            # ```
            # needs to be formatted as a dictionary {'trauma_torso': 'absence'}
            response = response[start_idx + len("```python") :]
            end_idx = response.find("```")
            response = response[:end_idx]
            response = response.replace("\n", "")
            response = response.replace(" ", "")
            if "=" in response:
                first_word = response.split("=")[0]
                if not first_word.startswith("'"):
                    first_word = f"'{first_word}"
                if not first_word.endswith("'"):
                    first_word = f"{first_word}'"
                second_word = response.split("=")[1]
                response = f"{first_word}: {second_word}"

            response = f"{{{response}}}"
            logger.info(f"Response after formatting to dict: {response}")

    else:
        response = response[start_idx : end_idx + 1]

    response = response.replace("'", '"')
    response = response.replace(" ", "")
    response = response.replace("\n", "")
    response = response.replace("\t", "")
    response = response.replace("\r", "")
    response = response.replace("\\", "")
    logger.info(f"Response: {response}")

    try:
        response = eval(response)
    except:
        return "The response does not seem to be a valid dictionary. Please provide a response in the format requested."

    # check that the response values fit the expected values
    for key in response.keys():  # TODO fix this to use LABELING constants
        if key != label_class:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. "
        if key == "trauma_head" and response[key] not in ["absence", "presence"]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for trauma_head are: absense, presence "
        if key == "trauma_torso" and response[key] not in ["absence", "presence"]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for trauma_torso are: absense, presence "
        if key == "trauma_upper_ext" and response[key] not in [
            "normal",
            "wound",
            "amputation",
        ]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for trauma_upper_ext are: normal, wound, amputation "
        if key == "trauma_lower_ext" and response[key] not in [
            "normal",
            "wound",
            "amputation",
        ]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for trauma_lower_ext are: normal, wound, amputation "
        if key == "alertness_ocular" and response[key] not in [
            "open",
            "closed",
            "untestable",
        ]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for alertness_ocular are: open, closed, untestable "
        if key == "alertness_motor" and response[key] not in [
            "normal",
            "abnormal",
            "absence",
            "untestable",
        ]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for alertness_motor are: normal, abnormal, absence, untestable "
        if key == "alertness_verbal" and response[key] not in [
            "normal",
            "abnormal",
            "absence",
            "untestable",
        ]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for alertness_verbal are: normal, abnormal, absence, untestable "
            
    parsed_response = parse_label_dict_str_to_int(response)

    return parsed_response


def old_parse_dict_response(response, label_class):
    """Parse the response from the VLM to a dictionary.
       Runs several checks to ensure the response is in the correct format.

    Args:
        response (str): The response from the VLM. Hopefully
            this contains a dictionary.

    Returns:
        dict: The parsed response.
    """
    # first, figure out where the dictionary starts and ends
    start_idx = response.find("{")
    end_idx = response.find("}")
    if start_idx == -1 or end_idx == -1:
        rospy.loginfo(f"Response does not contain a dictionary. Response: {response}")
        return "The response does not seem to contain a dictionary. Please provide a response in the format requested."
    response = response[start_idx : end_idx + 1]
    response = response.replace("'", '"')
    response = response.replace(" ", "")
    response = response.replace("\n", "")
    response = response.replace("\t", "")
    response = response.replace("\r", "")
    response = response.replace("\\", "")
    rospy.loginfo(f"Response: {response}")

    try:
        response = eval(response)
    except:
        return "The response does not seem to be a valid dictionary. Please provide a response in the format requested."

    # check that the response values fit the expected values
    for key in response.keys():  # TODO fix this to use LABELING constants
        if key != label_class:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. "
        if key == "trauma_head" and response[key] not in ["absence", "presence"]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for trauma_head are: absense, presence "
        if key == "trauma_torso" and response[key] not in ["absence", "presence"]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for trauma_torso are: absense, presence "
        if key == "trauma_upper_ext" and response[key] not in [
            "normal",
            "wound",
            "amputation",
        ]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for trauma_upper_ext are: normal, wound, amputation "
        if key == "trauma_lower_ext" and response[key] not in [
            "normal",
            "wound",
            "amputation",
        ]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for trauma_lower_ext are: normal, wound, amputation "
        if key == "alertness_ocular" and response[key] not in [
            "open",
            "closed",
            "untestable",
        ]:
            return f"The entry for key {key} seems to be parsed incorrectly. Please provide a response in the format requested. Options for alertness_ocular are: open, closed, untestable "

    parsed_response = parse_label_dict_str_to_int(response)

    return parsed_response


class VLMNode:
    def __init__(self):
        # create a run directory with a timestamp
        self.model_path = rospy.get_param(
            "model_path",
            "/mnt/dtc/perception_models/llava/llava-onevision-qwen2-7b-ov-chat-rank128-alpha8-ourdata_lr5e-6-lora/",
        )
        self.model_base = rospy.get_param(
            "model_path",
            "/mnt/dtc/perception_models/llava/llava-onevision-qwen2-7b-ov-chat/",
        )
        self.device = rospy.get_param("device", "gpu")
        rospy.loginfo(f"Set up device {self.device}.")

        self.run_dir = rospy.wait_for_message("/run_dir", String, timeout=None).data

        # create subscriber to drone and ground
        self.image_sub = rospy.Subscriber(
            "/received_images", ReceivedImageData, self.vlm_callback
        )
        self.image_analysis_publisher = rospy.Publisher(
            "/image_analysis_results", Int8, queue_size=2
        )
        rospy.loginfo("Created subscribers to /received_images and publishers to /image_analysis_results.")

        self._create_databases()

        # load the VLM model into memory once
        disable_torch_init()
        self.model_name = get_model_name_from_path(self.model_path)
        rospy.loginfo(f"Loading model {self.model_name} from {self.model_path}.")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path,
            self.model_base, #TODO: fix this to use the lora models properly
            self.model_name,
            load_8bit=False,
            load_4bit=False,
        )
        rospy.loginfo(f"Successfully loaded model {self.model_name}.")

        # create the http streamer
        self.streamer = HttpStreamer(
            server_url="http://localhost:8080",
            tokenizer=self.tokenizer,
            skip_prompt=False,
            skip_special_tokens=True
        )

        # parameters for VLM prediction
        self.temperature = 0.5
        self.max_new_tokens = 512
        self.device = "cuda"
        self.num_tries = 2

        # get all prompts: TODO: clean this up, move more to _get_prompts function
        base_prompt_path = (
            f"/home/{getpass.getuser()}/ws/src/DTC_internal/dtc_vlm/_prompting_final/"
        )
        user_response_fn_path = os.path.join(base_prompt_path, "user_response_fn.py")
        final_prompt_paths = {
            "trauma_head": os.path.join(base_prompt_path, "final_prompt_head.txt"),
            "trauma_torso": os.path.join(base_prompt_path, "final_prompt_torso.txt"),
            "trauma_lower_ext": os.path.join(base_prompt_path, "final_prompt_legs.txt"),
            "trauma_upper_ext": os.path.join(base_prompt_path, "final_prompt_arms.txt"),
            "alertness_ocular": os.path.join(base_prompt_path, "final_prompt_eyes.txt"),
            "severe_hemorrhage": os.path.join(
                base_prompt_path, "final_prompt_blood.txt"
            ),
        }

        self.init_prompt_flat, self.urf_flat, self.final_prompts_flat = (
            self._get_prompts(
                initial_prompt_path=os.path.join(
                    base_prompt_path, "flat", "initial_prompt.txt"
                ),
                final_prompt_paths=final_prompt_paths,
                user_response_fn_path=user_response_fn_path,
            )
        )
        self.init_prompt_upright, self.urf_upright, self.final_prompts_upright = (
            self._get_prompts(
                initial_prompt_path=os.path.join(
                    base_prompt_path, "upright", "initial_prompt.txt"
                ),
                final_prompt_paths=final_prompt_paths,
                user_response_fn_path=user_response_fn_path,
            )
        )
        self.init_prompt_drone, self.urf_drone, self.final_prompts_drone = (
            self._get_prompts(
                initial_prompt_path=os.path.join(
                    base_prompt_path, "drone", "initial_prompt.txt"
                ),
                final_prompt_paths=final_prompt_paths,
                user_response_fn_path=user_response_fn_path,
            )
        )

    def _create_databases(self):
        self.ground_image_pred_path = os.path.join(self.run_dir, "ground_image_pred.csv")
        if not os.path.exists(self.ground_image_pred_path):
            _df = pd.DataFrame(
                columns=[
                    "casualty_id",
                    *LABEL_CLASSES[:-1],
                ]
                )
            _df.to_csv(self.ground_image_pred_path, index=False)
            rospy.loginfo(f"Created file at {self.ground_image_pred_path}.")
        else:
            rospy.loginfo(f"File already exists at {self.ground_image_pred_path}. Continuing.")

        self.aerial_image_pred_path = os.path.join(self.run_dir, "aerial_image_pred.csv")
        if not os.path.exists(self.aerial_image_pred_path):
            _df = pd.DataFrame(
                columns=[
                    "casualty_id",
                    # all label classes except for alertness_motor and alertness_verbal
                    *LABEL_CLASSES[:-2],
                ]
            )
            _df.to_csv(self.aerial_image_pred_path, index=False)
            rospy.loginfo(f"Created file at {self.aerial_image_pred_path}.")
        else:
            rospy.loginfo(f"File already exists at {self.aerial_image_pred_path}. Continuing.")

        self.whisper_pred_path = os.path.join(self.run_dir, "whisper_pred.csv")
        if not os.path.exists(self.whisper_pred_path):
            _df = pd.DataFrame(
                columns=[
                    "casualty_id",
                    "whisper_id",
                    "alertness_verbal",
                ]
            )
            _df.to_csv(self.whisper_pred_path, index=False)
            rospy.loginfo(f"Created file at {self.whisper_pred_path}.")
        else:
            rospy.loginfo(f"File already exists at {self.whisper_pred_path}. Continuing.")

    def _get_prompts(
        self, initial_prompt_path, final_prompt_paths, user_response_fn_path
    ):
        # load initial and final prompt from text files
        with open(initial_prompt_path, "r") as f:
            initial_prompt = f.read()

        final_prompts = {}
        # transform the final prompt paths from omegaconf object to a dictionary
        final_prompt_paths = dict(final_prompt_paths)

        for label_class in LABEL_CLASSES:
            if label_class == "alertness_motor" or label_class == "alertness_verbal":
                continue

            with open(final_prompt_paths[label_class], "r") as f:
                final_prompts[label_class] = f.read()

        # define the user response function by importing it from the config
        absolute_fn_path = os.path.join(user_response_fn_path)
        module_name = os.path.splitext(absolute_fn_path)[0]
        spec = importlib.util.spec_from_file_location(module_name, absolute_fn_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        user_response_fn = getattr(module, "user_response_fn")

        return initial_prompt, user_response_fn, final_prompts


    def _get_conv_mode(self, model_name):
        """Get the conversation mode based on the model name."""
        if "llama2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "llama3" in model_name.lower():
            conv_mode = "llava_llama_3"
        elif "qwen" in model_name.lower():
            conv_mode = "qwen_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        rospy.loginfo(f"Got conversation mode: {conv_mode}")

        return conv_mode

    def _get_initial_prediction_llava_ov(self, image_tensor, image_sizes, init_prompt):
        conv_mode = self._get_conv_mode(self.model_name)
        conv = conv_templates[conv_mode].copy()

        # # Initial prompt
        # if self.model.config.mm_use_im_start_end:
        #     inp = (
        #         DEFAULT_IM_START_TOKEN
        #         + DEFAULT_IMAGE_TOKEN
        #         + DEFAULT_IM_END_TOKEN
        #         + "\n"
        #         + init_prompt
        #     )
        # else:
        #     inp = DEFAULT_IMAGE_TOKEN + init_prompt

        conv.append_message(conv.roles[0], init_prompt)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        tokenized_prompt = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # get the response from the model
        with torch.inference_mode():
            tokenized_response = self.model.generate(
                tokenized_prompt,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=self.streamer,
                use_cache=False,
                # stopping_criteria=[stopping_criteria],
            )
        string_response = self.tokenizer.decode(tokenized_response[0]).strip()
        conv.messages[-1][-1] = string_response

        return conv, string_response

    def _predict_downstream_llava_ov(
        self,
        conv,
        string_response,
        image_tensor,
        image_sizes,
        label_class,
        user_response_fn,
        final_prompt,
    ):
        keep_prompting = True
        num_steps_in_user_fn = 0
        while keep_prompting:
            user_response, should_break = user_response_fn(
                string_response, num_steps_in_user_fn, label_class
            )

            if should_break:
                break

            conv.append_message(conv.roles[0], user_response)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            tokenized_prompt = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )

            num_steps_in_user_fn += 1
            # get the response from the model
            with torch.inference_mode():
                tokenized_response = self.model.generate(
                    tokenized_prompt,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    streamer=self.streamer,
                    use_cache=False,
                    # stopping_criteria=[stopping_criteria],
                )
            string_response = self.tokenizer.decode(tokenized_response[0]).strip()
            conv.messages[-1][-1] = string_response

        # add the final prompt
        conv.append_message(conv.roles[0], final_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        tokenized_prompt = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # get the response from the model
        with torch.inference_mode():
            tokenized_response = self.model.generate(
                tokenized_prompt,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                # streamer=self.streamer,
                use_cache=False,
            )

        string_response = self.tokenizer.decode(tokenized_response[0]).strip()
        conv.messages[-1][-1] = string_response
        rospy.loginfo("Made it to final prompt, tying to parse the response")

        failed_to_parse = True
        for i in range(self.num_tries):
            final_response = parse_dict_response(string_response, label_class)
            if isinstance(final_response, dict):
                rospy.loginfo(f"Successfully parsed the response after {i} tries.")
                failed_to_parse = False
                break
            else:
                rospy.loginfo(
                    f"Failed to parse the response after {i} tries. Trying again."
                )
                conv.append_message(conv.roles[0], final_response)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                tokenized_prompt = (
                    tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0)
                    .to(self.device)
                )
                # get the response from the model
                with torch.inference_mode():
                    tokenized_response = self.model.generate(
                        tokenized_prompt,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        do_sample=True if self.temperature > 0 else False,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                        streamer=self.streamer,
                        use_cache=False,
                    )
                    string_response = self.tokenizer.decode(
                        tokenized_response[0]
                    ).strip()
                conv.messages[-1][-1] = string_response

        if i == self.num_tries - 1 and failed_to_parse:
            rospy.loginfo(
                f"Failed to parse the response after {i} tries. Restarting prediction process. "
                + "Not incrementing k_round."
            )
            return False, {}, ""

        prompt = conv.get_prompt()

        return True, final_response, prompt

    def _get_prompts_from_image_type(self, image_type):
        if image_type == "flat":
            return self.init_prompt_flat, self.urf_flat, self.final_prompts_flat
        elif image_type == "upright":
            return (
                self.init_prompt_upright,
                self.urf_upright,
                self.final_prompts_upright,
            )
        elif image_type == "drone":
            return self.init_prompt_drone, self.urf_drone, self.final_prompts_drone
        else:
            raise ValueError(f"Invalid image type: {image_type}")

    def _predict_all_labels_from_vlm(self, images, image_type="flat"):
        """Predict all labels from the VLM.

        Args:
            images (list): List of images to predict. Needs to be in PIL format.

        Returns:
            dict: A dictionary with the predictions for each label class.
        """
        init_prompt, user_response_fn, final_prompt_dict = (
            self._get_prompts_from_image_type(image_type)
        )
        init_prompt += f"{DEFAULT_IMAGE_TOKEN} Describe this image. Be very brief. Do not be wordy."

        predictions = {}
        image_sizes = [img.size for img in images]
        image_tensor = process_images(
            images, self.image_processor, self.model.config
        ).half()

        conv, string_response = self._get_initial_prediction_llava_ov(
            image_tensor, image_sizes, init_prompt
        )

        all_prompts = []
        for label_class in LABEL_CLASSES:
            if label_class == "alertness_motor" or label_class == "alertness_verbal":
                continue

            final_prompt = final_prompt_dict[label_class]

            success = False
            while not success:
                success, prediction, prompt = self._predict_downstream_llava_ov(
                    copy.deepcopy(conv),
                    string_response,
                    image_tensor,
                    image_sizes,
                    label_class,
                    user_response_fn,
                    final_prompt,
                )
            predictions[label_class] = prediction[label_class]
            if len(prompt) > 1:
                all_prompts.append(prompt)

        return predictions, all_prompts

    def _predict_motion_from_video(self, images):
        rospy.loginfo("Predicting motion from video.")
        failed_to_parse = True

        num_tries = 0
        while failed_to_parse:
            conv_mode = self._get_conv_mode(self.model_name)
            conv = conv_templates[conv_mode].copy()

            QUESTION_STRING = f"{DEFAULT_IMAGE_TOKEN} , {DEFAULT_IMAGE_TOKEN} , {DEFAULT_IMAGE_TOKEN} " + \
                "You are given a set of 3 images. All images contain a person. " + \
                "Did the person move between the images? Respond with a dictionary with the key 'alertness_motor' " + \
                "and the value 'normal' for no movement, 'abnormal' for movement, 'absence' for no person, or " + \
                "'untestable' if the movement is untestable."
            
            conv.append_message(conv.roles[0], QUESTION_STRING)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            tokenized_prompt = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )

            # get the response from the model
            # resize to make sure we don't run out of memory
            images = [img.resize((256, 256)) for img in images]
            image_sizes = [img.size for img in images]

            # process the images and run the model
            image_tensor = process_images(
                images, self.image_processor, self.model.config
            ).half()
            
            with torch.inference_mode():
                tokenized_response = self.model.generate(
                    tokenized_prompt,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    # streamer=self.streamer,
                    use_cache=False,
                    # stopping_criteria=[stopping_criteria],
                )
            string_response = self.tokenizer.decode(tokenized_response[0]).strip()
            conv.messages[-1][-1] = string_response

            for i in range(self.num_tries):
                final_response = parse_dict_response(string_response, "alertness_motor")
                if isinstance(final_response, dict):
                    rospy.loginfo(f"Successfully parsed the response after {i} tries.")
                    failed_to_parse = False
                    break
                rospy.loginfo(
                    f"Failed to parse the response after {i} tries. Trying again."
                )
                rospy.loginfo(f"Response: {string_response}")

            num_tries += 1

            if num_tries == 3 and failed_to_parse:
                rospy.loginfo(
                    f"Failed to parse the response after {num_tries} tries. " + \
                    "Defaulting to untestable."
                )
                return {"alertness_motor": 3}

        return final_response


    def _predict_if_whisper_is_text(self, whisper):
        rospy.loginfo("Predicting if whisper is text.")
        failed_to_parse = True
        num_tries = 0

        while failed_to_parse:
            conv_mode = self._get_conv_mode(self.model_name)
            conv = conv_templates[conv_mode].copy()

            QUESTION_STRING = f"Is this a sentence: '{whisper}'. Respond with a dictonary with the key 'alertness_verbal' " + \
                "and the value 'normal' for yes, 'abnormal' for unclear speech, " + \
                "'absence' for no speech. Here is the sentence: "
            conv.append_message(conv.roles[0], QUESTION_STRING + "'" + whisper + "'")
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            tokenized_prompt = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )

            # get the response from the model
            with torch.inference_mode():
                tokenized_response = self.model.generate(
                    tokenized_prompt,
                    images=None,
                    image_sizes=None,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    # streamer=self.streamer,
                    use_cache=False,
                    # stopping_criteria=[stopping_criteria],
                )
            string_response = self.tokenizer.decode(tokenized_response[0]).strip()
            conv.messages[-1][-1] = string_response

            for i in range(self.num_tries):
                final_response = parse_dict_response(string_response, "alertness_verbal")
                if isinstance(final_response, dict):
                    rospy.loginfo(f"Successfully parsed the response after {i} tries.")
                    failed_to_parse = False
                    break
                rospy.loginfo(
                    f"Failed to parse the response after {i} tries. Trying again."
                )
                rospy.loginfo(f"Response: {string_response}")
            
            num_tries += 1

            if num_tries == 3 and failed_to_parse:
                rospy.loginfo(
                    f"Failed to parse the response after {num_tries} tries. " + \
                    "Defaulting to absence."
                )
                return {"alertness_verbal": 2}

        return final_response
    
    def _save_ground_predictions(self, ground_vlm_pred, motion_vlm_pred, casualty_id):
        with portalocker.Lock(self.ground_image_pred_path, "r+") as f:
            df = pd.read_csv(f)

            # save predictions to df
            predictions = {
                "casualty_id": casualty_id,
                "trauma_head": ground_vlm_pred["trauma_head"],
                "trauma_torso": ground_vlm_pred["trauma_torso"],
                "trauma_lower_ext": ground_vlm_pred["trauma_lower_ext"],
                "trauma_upper_ext": ground_vlm_pred["trauma_upper_ext"],
                "alertness_ocular": ground_vlm_pred["alertness_ocular"],
                "alertness_motor": motion_vlm_pred["alertness_motor"],
            }
            new_row = pd.DataFrame([predictions])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(f, index=False)
            rospy.loginfo(f"Successfully saved ground predictions.")

    def _save_aerial_predictions(self, aerial_preds, casualty_id):
        with portalocker.Lock(self.aerial_image_pred_path, "r+") as f:
            df = pd.read_csv(f)

            # save predictions to df
            predictions = {
                "casualty_id": casualty_id,
                "trauma_head": aerial_preds["trauma_head"],
                "trauma_torso": aerial_preds["trauma_torso"],
                "trauma_lower_ext": aerial_preds["trauma_lower_ext"],
                "trauma_upper_ext": aerial_preds["trauma_upper_ext"],
                "alertness_ocular": aerial_preds["alertness_ocular"],
                "severe_hemorrhage": aerial_preds["severe_hemorrhage"],
            }
            append_df = pd.DataFrame([predictions])
            df = pd.concat([df, append_df], ignore_index=True)
            df.to_csv(f, index=False)
            rospy.loginfo(f"Successfully saved aerial predictions.")

    def _save_whisper_predictions(self, whisper_preds, whisper_id, casualty_id):
        with portalocker.Lock(self.whisper_pred_path, "r+") as f:
            df = pd.read_csv(f)

            # save predictions to df
            predictions = {
                "casualty_id": casualty_id,
                "whisper_id": whisper_id,
                "alertness_verbal": whisper_preds["alertness_verbal"],
            }
            df = pd.DataFrame([predictions])
            append_df = pd.DataFrame([predictions])
            df = pd.concat([df, append_df], ignore_index=True)
            df.to_csv(f, index=False)
            rospy.loginfo(f"Successfully saved whisper predictions for id {whisper_id}.")

    def vlm_callback(self, msg):
        """Callback that is triggrered when ground robot sends a GroundDetection.

        Args:
            msg (GroundDetection): Message containing gps and the detection values
        """
        start_time_callback = time.time()

        casualty_id = msg.casualty_id
        image_path_list = msg.image_path_list
        rospy.loginfo(f"Successfully parsed message in VLM node.")

        # Check if the image_path_list starts with the 0th path in the filename
        # or the 3rd path. 
        if "0" in os.path.basename(image_path_list[0]):
            round_number = 0
        elif "3" in os.path.basename(image_path_list[0]):
            round_number = 1
        else:
            rospy.logerr(f"Could not determine round number from image_path_list: {image_path_list}")
            return

        num_images = len(image_path_list)
        rospy.loginfo(f"Received {num_images} image paths in VLM, starting prediction.")

        for image_path in image_path_list:
            self.streamer.send_image(Image.open(image_path))

        ### START WITH GROUND
        # Run the VLM for the ground image
        ground_img_list = [Image.open(image_path_list[0])]
        rospy.loginfo("Received images, triggering VLM.")
        ground_vlm_pred, ground_vlm_prompts = self._predict_all_labels_from_vlm(ground_img_list)
        rospy.loginfo(f"Successfully predicted ground labels.")
        
        # save the ground prompts into the directory that contains the image
        try:
            for i, prompt in enumerate(ground_vlm_prompts):
                with open(os.path.join(os.path.dirname(image_path_list[-1]), f"prompt_{i}_{round_number}.txt"), "w") as f:
                    f.write(prompt)
            rospy.loginfo(f"Successfully saved ground prompts.")
        except:
            rospy.logerr(f"Could not save ground prompts.")

        ### VIDEO PREDICTION WITH GROUND
        video_img_list = [Image.open(image_path_list[i]) for i in range(len(image_path_list))]
        motion_response_dict = self._predict_motion_from_video(video_img_list)
        rospy.loginfo(f"Successfully predicted motion labels.")
        
        # save all the ground predictions
        self._save_ground_predictions(ground_vlm_pred, motion_response_dict, casualty_id)

        ### CONTINUE TO DRONE
        # check if we have already seen the drone image for this casualty_id
        with portalocker.Lock(self.aerial_image_pred_path, "r") as f:
            aerial_pred_df = pd.read_csv(f)

        if casualty_id in aerial_pred_df["casualty_id"].values:
            rospy.loginfo(
                f"Already seen drone image for casualty_id {casualty_id}. Skipping drone image."
            )
            drone_img_list = None
        else:
            # Try to find the drone image
            all_drone_images_paths = glob.glob(os.path.join(os.path.dirname(image_path_list[0]), "drone_img.png"))
            if len(all_drone_images_paths) == 1:
                rospy.loginfo(f"Found drone image for casualty_id {casualty_id}.")

                drone_img_list = [Image.open(all_drone_images_paths[0])]
                drone_vlm_pred, drone_vlm_prompts = self._predict_all_labels_from_vlm(drone_img_list, image_type="drone")
                rospy.loginfo(f"Successfully predicted drone labels.")
                self._save_aerial_predictions(drone_vlm_pred, casualty_id)
            else:
                rospy.loginfo(
                    f"Did not find drone image for casualty_id {casualty_id}. Skipping drone image."
                )

                # save the air prompts into the directory that contains the drone_image
                for i, prompt in enumerate(drone_vlm_prompts):
                    try:
                        with open(os.path.join(os.path.dirname(all_drone_images_paths[-1]), f"drone_prompt_{i}.txt"), "w") as f:
                            f.write(prompt)
                        rospy.loginfo(f"Successfully saved air prompts.")
                    except:
                        rospy.logerr(f"Could not save air prompts.")
                                    
        ### CONTINUE TO WHISPER
        # load the seens whisper ids:
        with portalocker.Lock(self.whisper_pred_path, "r") as f:
            whisper_data_df = pd.read_csv(f)

        # get missing whisper id's 
        missing_whisper_ids = [i for i in range(2) if i not in whisper_data_df[whisper_data_df["casualty_id"] == casualty_id]["whisper_id"].values]
        rospy.loginfo(f"Missing whisper ids for casualty_id {casualty_id}: {missing_whisper_ids}")

        if len(missing_whisper_ids) > 0:
            rospy.loginfo(f"Starting to check missing whisper strings for casualty_id {casualty_id}.")
            # Check if there is a whisper string available
            whispers_to_check = {}
            for idx in missing_whisper_ids:
                whisper_path = os.path.join(os.path.dirname(image_path_list[0]), f"whisper_{idx}.txt")
                rospy.loginfo(f"Looking for path {whisper_path}")
                if os.path.exists(whisper_path):
                    rospy.loginfo(f"Found path {whisper_path}")
                    with portalocker.Lock(whisper_path, "r") as f:
                        with open(whisper_path, "r") as f:
                            whisper = f.read()
                    rospy.loginfo(f"Found whisper string for casualty_id {casualty_id} and whisper_id {idx}.")
                    whispers_to_check[idx] = whisper
                else:
                    rospy.loginfo(f"Did not find whisper string for casualty_id {casualty_id} and whisper_id {idx}. Skipping whisper string.")

            if len(whispers_to_check) > 0:
                for whisper_id, whisper in whispers_to_check.items():
                    # run the text checker
                    if whisper == " ":
                        response_dict = {"alertness_verbal": 2}
                        rospy.loginfo("No speech detected, defaulting to 2.")
                    else:
                        response_dict = self._predict_if_whisper_is_text(whisper)
                        rospy.loginfo(f"Predicted whisper string for casualty_id {casualty_id} and whisper_id {whisper_id}.")
                    
                    self._save_whisper_predictions(response_dict, whisper_id, casualty_id)
            else:
                rospy.loginfo(f"Did not find any whisper strings for casualty_id {casualty_id}.")
        else:
            rospy.loginfo(f"Already seen all whisper strings for casualty_id {casualty_id}. Skipping whisper strings.")

        # publish the results
        msg = Int8()
        msg.data = casualty_id
        self.image_analysis_publisher.publish(msg)
        rospy.loginfo(
            f"Total time for VLM callback: {time.time() - start_time_callback} seconds."
        )

def main():
    rospy.init_node("vlm_inference")
    inf_node = VLMNode()

    rospy.spin()

if __name__ == "__main__":
    main()
