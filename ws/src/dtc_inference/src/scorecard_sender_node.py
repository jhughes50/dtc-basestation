#!/usr/bin/env python3

# supress warnings
import warnings

warnings.filterwarnings("ignore")

import rospy
from std_msgs.msg import String
from dtc_inference.msg import ReceivedSignal, ImageAnalysisResult

import os
import requests
import pandas as pd
import portalocker


# fmt: off
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# fmt: on


class ScoringClient:
    def __init__(self, ip="http://localhost"):
        self.api_url = os.path.join(ip, "api")
        # token_response = self._get_login_token()
        self.token_type = "Bearer"  # token_response["token_type"]
        self.token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4M2Q3OGM4ZS04MzhhLTQ0NzctOWM3Yi02N2VmMTZlNWY3MTYiLCJpIjowfQ.i4KuwEtc5_6oIYz5TDWcdzl5bMkvCpLZTSZG2Avy84w"  # token_response["access_token"]
        self.headers = {"Authorization": f"{self.token_type} {self.token}"}

        rospy.loginfo(f"Scoring client created with API URL: {self.api_url}")

    def _get_login_token(self):
        """Get a login token from the scoring server

        Returns:
            dict: The response from the server containing the token
        """
        url = os.path.join(self.api_url, "login", "token")
        response = requests.post(url)
        return response.json()

    def _check_post_response(self, response):
        rospy.loginfo(f"Received status code {response.status_code}.")
        if response.status_code == 200 or response.status_code == 201:
            return response
        elif response.status_code == 400:
            raise Exception("Bad request, JSON parsing failed")
        elif response.status_code == 404:
            raise Exception("Casualty not found")
        elif response.status_code == 422:
            response = response.json()
            raise Exception(
                f"Invalid {response['detail'][0]['loc'][1]}: {response['detail'][0]['msg']}"
            )
        elif response.status_code == 429:
            raise Exception("Too many requests submitted in given time")
        else:
            raise Exception("Unknown error")

    def get_status(self):
        """Get the status of the scoring server

        Raises:
            Exception: If the token is invalid

        Returns:
            dict: The response from the server
        """
        url = os.path.join(self.api_url, "status")
        response = requests.get(url, headers=self.headers)
        if response.status_code == 403:
            raise Exception("Token is invalid")

        return response.json()

    def get_new_run(self):
        """Get a new run from the scoring server

        Returns:
            dict: The response from the server
        """
        url = os.path.join(self.api_url, "run", "new")
        response = requests.get(url, headers=self.headers)

        return response.json()

    def get_start_run(self):
        """Start a run on the scoring server

        Returns:
            dict: The response from the server
        """
        url = os.path.join(self.api_url, "run", "start")
        response = requests.get(url, headers=self.headers)

        return response.json()

    def post_critical(self, system, casualty_id, report_type, report_value):
        """Post a critical report to the scoring server

        Args:
            system (str): The system responsible for the identification of the casualty
            casualty_id (int): The ID of the casualty
            report_type (str): The type of critical report, can be one of the following:
                - severe_hemorrhage
                - respiratory_distress
            report_value (int): The value of the critical report, can be either 0 or 1
                for absence or presence of the reported type

        Returns:
            dict: The response from the server
        """
        url = os.path.join(self.api_url, "critical")
        report = {
            "system": system,
            "casualty_id": casualty_id,
            "type": report_type,
            "value": report_value,
        }

        rospy.loginfo(
            f"Posting critical report: with system: {system}, casualty_id: "
            + f"{casualty_id}, type: {report_type}, value: {report_value}"
        )
        response = requests.post(url, headers=self.headers, json=report)
        response = self._check_post_response(response)

        return response.json()

    def post_vitals(self, system, casualty_id, report_type, report_value, time_ago):
        """Post a vitals report to the scoring server.

        Args:
            system (str): The system responsible for the identification of the casualty.
            casualty_id (int): The ID of the casualty.
            report_type (str): The type of vitals report, can be one of the following:
                - hr
                - rr
            report_value (int): The value of the vitals report. This is either the heart rate
                in beats per minute or the respiratory rate in breaths per minute as an integer.
            time_ago (int): The time ago in seconds that the vitals were measured.

        Returns:
            dict: The response from the server
        """
        url = os.path.join(self.api_url, "vitals")

        report = {
            "system": system,
            "casualty_id": casualty_id,
            "type": report_type,
            "value": report_value,
            "time_ago": time_ago,
        }

        rospy.loginfo(
            f"Posting vitals report: with system: {system}, casualty_id: "
            + f"{casualty_id}, type: {report_type}, value: {report_value}, time_ago: {time_ago}"
        )
        response = requests.post(url, headers=self.headers, json=report)
        response = self._check_post_response(response)

        return response.json()

    def post_injury(self, system, casualty_id, report_type, report_value):
        """Post an injury report to the scoring server.

        Args:
            system (str): The system responsible for the identification of the casualty.
            casualty_id (int): The  ID of the casualty.
            report_type (str): The type of injury report, can be one of the following:
                - trauma_head
                - trauma_torso
                - trauma_lower_ext
                - trauma_upper_ext
                - alertness_ocular
                - alertness_verbal
                - alertness_motor
            report_value (int): The value of the injury report. For each specific injury type,
                the value can be one of the following:
                - trauma_head: 0 or 1, for absence or presence of head trauma
                - trauma_torso: 0 or 1, for absence or presence of torso trauma
                - trauma_lower_ext: 0, 1, or 2, for normal, wound or amputation
                - trauma_upper_ext: 0, 1, or 2, for normal, wound or amputation
                - alertness_ocular: 0, 1, or 2, open, closed, or untestable
                - alertness_verbal: 0, 1, 2, or 3, normal, abnormal, absent, or untestable
                - alertness_motor: 0, 1, 2, or 3, normal, abnormal, absent, or untestable

        Returns:
            dict: The response from the server
        """

        url = os.path.join(self.api_url, "injury")

        report = {
            "system": system,
            "casualty_id": casualty_id,
            "type": report_type,
            "value": report_value,
        }

        rospy.loginfo(
            f"Posting injury report: with system: {system}, casualty_id: "
            + f"{casualty_id}, type: {report_type}, value: {report_value}"
        )
        response = requests.post(url, headers=self.headers, json=report)
        response = self._check_post_response(response)

        return response.json()

    def send_partial_scorecard(self, system, casualty_id, scorecard):
        """Send a partial scorecard to the scoring server where the partial scorecard is a
           pandas dataframe with the columns type and value.

        Args:
            system (str): The system responsible for the identification of the casualty.
            casualty_id (int): The ID of the casualty.
            scorecard (pd.DataFrame): The partial scorecard as a pandas dataframe. The columns
                should be type and value where type is the type of the report and value is the
                value of the report.

        Raises:
            Exception: If the report type is invalid

        Returns:
            bool: True if the scorecard was sent successfully
        """
        for _, row in scorecard.iterrows():
            if row["type"] in get_possible_criticals().keys():
                self.post_critical(system, casualty_id, row["type"], row["value"])
            elif row["type"] in get_possible_vitals().keys():
                self.post_vitals(system, casualty_id, row["type"], row["value"], 0)
            elif row["type"] in get_possible_injuries().keys():
                self.post_injury(system, casualty_id, row["type"], row["value"])
            else:
                raise Exception(f"Invalid report type {row['type']}")

        return True


def get_possible_criticals():
    """Get the possible criticals that can be reported

    Returns:
        dict: A dictionary with the possible criticals and their possible values
    """
    criticals = {
        "severe_hemorrhage": [0, 1],
        "respiratory_distress": [0, 1],
    }
    return criticals


def get_possible_vitals():
    """Get the possible vitals that can be reported

    Returns:
        dict: A dictionary with the possible vitals and their possible values
    """
    vitals = {
        "hr": int,
        "rr": int,
    }
    return vitals


def get_possible_injuries():
    """Get the possible injuries that can be reported

    Returns:
        dict: A dictionary with the possible injuries and their possible values
    """
    injuries = {
        "trauma_head": [0, 1],
        "trauma_torso": [0, 1],
        "trauma_lower_ext": [0, 1, 2],
        "trauma_upper_ext": [0, 1, 2],
        "alertness_ocular": [0, 1, 2],
        "alertness_verbal": [0, 1, 2, 3],
        "alertness_motor": [0, 1, 2, 3],
    }
    return injuries


def get_all_possible_types():
    """Get all possible types that can be reported

    Returns:
        list: A list with all possible types
    """
    all_possible_types = []
    for t in [get_possible_criticals(), get_possible_vitals(), get_possible_injuries()]:
        all_possible_types += t.keys()
    return all_possible_types


def create_empty_scorecard(file_location):
    """Create an empty scorecard and save it to a csv file

    Args:
        file_location (str): The location where the scorecard should be saved
    """
    # Create empty scorecard where all values are set to ""
    criticals = get_possible_criticals()
    vitals = get_possible_vitals()
    injuries = get_possible_injuries()

    # create an empty dataframe
    empty_df = pd.DataFrame(columns=["type", "value"])

    # add all keys to the dataframe in column type with "" value
    for t in [criticals, vitals, injuries]:
        for key in t.keys():
            empty_df = empty_df._append({"type": key, "value": 0}, ignore_index=True)

    # save the dataframe to a csv file
    empty_df.to_csv(file_location, index=False)


def load_scorecard(file_location):
    """Load a scorecard from a csv file

    Args:
        file_location (str): The location of the scorecard

    Returns:
        pd.DataFrame: The scorecard as a pandas dataframe
    """
    # Load the scorecard from a csv file
    scorecard = pd.read_csv(file_location)
    return scorecard


def save_scorecard(scorecard, file_location):
    """Save a scorecard to a csv file

    Args:
        scorecard (pd.DataFrame): The scorecard as a pandas dataframe
        file_location (str): The location where the scorecard should be saved

    Returns:
        str: The location of the saved scorecard
    """
    # Save the scorecard to a csv file
    scorecard.to_csv(file_location, index=False)
    return file_location


def add_value_to_scorecard(scorecard, report_type, report_value):
    """Add a value to a scorecard

    Args:
        scorecard (pd.DataFrame): The scorecard as a pandas dataframe
        report_type (str): The type of the report
        report_value (int): The value of the report

    Returns:
        pd.DataFrame: The updated scorecard
    """
    all_possible_types = get_all_possible_types()
    assert report_type in all_possible_types, f"Invalid report type {report_type}"

    # Add a new in the report_type column
    if report_type not in scorecard["type"].values:
        scorecard = scorecard.append(
            {"type": report_type, "value": report_value}, ignore_index=True
        )
        return scorecard
    else:
        # Update the value in the report_type column
        scorecard.loc[scorecard["type"] == report_type, "value"] = report_value

    return scorecard


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ip', default="http://10.200.2.100")
#     args = parser.parse_args()

#     sc = ScoringClient(args.ip)
#     print(sc.post_critical("test", 9, "severe_hemorrhage", 0))

# create_empty_scorecard("test_scorecard.csv")
# scorecard = load_scorecard("test_scorecard.csv")
# scorecard = add_value_to_scorecard(scorecard, "severe_hemorrhage", 1)
# save_scorecard(scorecard, "test_scorecard.csv")

# sc.get_new_run()
# sc.get_start_run()
# print(sc.get_status())
# sc.send_partial_scorecard("test_system", 1, scorecard)
# print(sc.get_status())

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

    # for key in LABEL_CLASSES:
    #     if key not in response.keys():
    #         return f"The entry for key {key} seems to be missing. Please provide a response in the format requested."

    # parse response string to int
    parsed_response = parse_label_dict_str_to_int(response)

    return parsed_response


class ScorecardSenderNode:
    def __init__(self):
        self.run_dir = rospy.wait_for_message("/run_dir", String, timeout=None).data
        self.vlm_sub_1 = rospy.Subscriber(
            "/image_analysis_results", ImageAnalysisResult, self.image_callback
        ) # TODO: fix this
        self.receiver_sub = rospy.Subscriber(
            "/received_signals", ReceivedSignal, self.signal_callback
        )

        self.ground_database_path = os.path.join(self.run_dir, "ground_data.csv")
        self.drone_database_path = os.path.join(self.run_dir, "drone_data.csv")
        self.whisper_database_path = os.path.join(self.run_dir, "whisper_data.csv")
        self.signal_database_path = os.path.join(self.run_dir, "signal_data.csv")

        self.message_dict = {}
        self.scoring_client = ScoringClient(ip="http://172.17.0.1:8000")

    def _aggregate_results(self, results):
        def custom_aggregation_fn(x, label_class):
            """Custom aggregation functions that aggregates keys in
            the dictionary differently for every key.
            For all wounds, the presence of a wound is the most important.
            For alertness, the presence of open eyes is the most important.
            For hemorrhage, the presence of severe hemorrhage is the most important.

            Args:
                x (list): A list of predictions for a label class.
                label_class (str): The label class that is being aggregated.

            Returns:
                int: The aggregated prediction.
            """
            # if the labels are numbers as strings, we need to convert them to integers
            if len(x) == 0:
                return 0
            
            if isinstance(x[0], str):
                x = [int(x) for x in x]

            if label_class == "trauma_head":
                return 1 if 1 in x else 0
            if label_class == "trauma_torso":
                return 1 if 1 in x else 0
            if label_class == "trauma_lower_ext":
                return 2 if 2 in x else (1 if 1 in x else 0)
            if label_class == "trauma_upper_ext":
                return 2 if 2 in x else (1 if 1 in x else 0)
            if label_class == "alertness_ocular":
                return 0 if 0 in x else (1 if 1 in x else 2)
            if label_class == "severe_hemorrhage":
                return 1 if 1 in x else 0
            if label_class == "alertness_verbal":
                return 0 if 0 in x else (1 if 1 in x else 2 if 2 in x else 3)
            if label_class == "alertness_motor":
                return 0 if 0 in x else (1 if 1 in x else 2 if 2 in x else 3)
            
            return 0
        
        agg_res = {}
        
        for label_class in LABEL_CLASSES:
            agg_res[label_class] = custom_aggregation_fn(results[label_class], label_class)

        return agg_res
    

    def image_callback(self, msg):
        casualty_id = msg.casualty_id
        # check if casualty_id is in message_dict
        if casualty_id not in self.message_dict.keys():
            self.message_dict[casualty_id] = []

        # add message to message_dict
        self.message_dict[casualty_id].append(msg)

        # check if message_dict has 2 messages
        # if so, send to scorecard
        if len(self.message_dict[casualty_id]) == 2:
            trauma_head_list = []
            trauma_torso_list = []
            trauma_lower_ext_list = []
            trauma_upper_ext_list = []
            alertness_ocular_list = []
            severe_hemorrhage_list = []
            alertness_motor_list = []
            alertness_verbal_list = []

            for msg in self.message_dict[casualty_id]:
                trauma_head_list += msg.trauma_head
                trauma_torso_list += msg.trauma_torso
                trauma_lower_ext_list += msg.trauma_lower_ext
                trauma_upper_ext_list += msg.trauma_upper_ext
                alertness_ocular_list += msg.alertness_ocular
                severe_hemorrhage_list += msg.severe_hemorrhage
                alertness_motor_list += msg.alertness_motor
                alertness_verbal_list += msg.alertness_verbal

            # aggregate results
            agg_res = self._aggregate_results(
                {
                    "trauma_head": trauma_head_list,
                    "trauma_torso": trauma_torso_list,
                    "trauma_lower_ext": trauma_lower_ext_list,
                    "trauma_upper_ext": trauma_upper_ext_list,
                    "alertness_ocular": alertness_ocular_list,
                    "severe_hemorrhage": severe_hemorrhage_list,
                    "alertness_motor": alertness_motor_list,
                    "alertness_verbal": alertness_verbal_list,
                }
            )
            
            scorecard_frame = pd.DataFrame(columns=["type", "value"])
            for key in agg_res.keys():
                scorecard_frame = scorecard_frame._append(
                    {"type": key, "value": agg_res[key]}, ignore_index=True
                )

            rospy.loginfo(f"Attempting to send image analysis to scorecard. \n " + \
                          f"Scorecard: {scorecard_frame}")
            self.scoring_client.send_partial_scorecard("test", casualty_id, scorecard_frame)
            rospy.loginfo("Successfully sent image analysis to scorecard.")            


    def signal_callback(self, msg):
        rospy.loginfo(f"Received signal call")
        casualty_id = msg.casualty_id
        heart_rate = msg.heart_rate
        respiratory_rate = msg.respiratory_rate

        # save the data into the database
        with portalocker.Lock(self.signal_database_path, "r+", timeout=1):
            signal_database_df = pd.read_csv(self.signal_database_path)
            append_dict = {
                "casualty_id": casualty_id,
                "heart_rate": heart_rate,
                "respiratory_rate": respiratory_rate,
            }
            signal_database_df = signal_database_df._append(append_dict, ignore_index=True)
            signal_database_df.to_csv(self.signal_database_path, index=False, mode="w")
                    
        # if casualty id in signal_database_df has two entries
        # send a majority vote to the scorecard
        signal_values = signal_database_df[signal_database_df["casualty_id"] == casualty_id]
        if len(signal_values) == 2:
            heart_rate_to_send = signal_values["heart_rate"].mode()[0]
            respiratory_rate_to_send = signal_values["respiratory_rate"].mode()[0]

            scorecard_frame = pd.DataFrame(columns=["type", "value"])
            scorecard_frame = scorecard_frame._append(
                {"type": "heart_rate", "value": heart_rate_to_send}, ignore_index=True
            )
            scorecard_frame = scorecard_frame._append(
                {"type": "respiratory_rate", "value": respiratory_rate_to_send}, ignore_index=True
            )

            rospy.loginfo(f"Attempting to send signal to scorecard. \n " + \
                            f"Scorecard: {scorecard_frame}")
            self.scoring_client.send_partial_scorecard("test", casualty_id, scorecard_frame)
            rospy.loginfo("Successfully sent signal to scorecard.")
        else:
            return False


def main():
    rospy.init_node("scorecard_sender")
    inf_node = ScorecardSenderNode()

    rospy.spin()


if __name__ == "__main__":
    main()
