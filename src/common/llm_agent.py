import re
from typing import Dict, Optional, Union
import os
import sys
from pathlib import Path

import cv2
sys.path.append(str(Path(str(os.getcwd())).resolve()))
import gc
import time
import lmdb
import tqdm
import math
import random
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from typing import List, Optional, DefaultDict
import msgpack_numpy

from utils.logger import logger
from utils.utils import get_rank, is_dist_avail_and_initialized, is_main_process, init_distributed_mode
from utils.vision import VisionClient
from Model.il_trainer import VLNCETrainer
from Model.utils.tensor_dict import DictTree, TensorDict
from Model.aux_losses import AuxLosses
from Model.utils.tensorboard_utils import TensorboardWriter
from Model.utils.common import append_text_to_image, images_to_video

from src.common.param import args
from src.vlnce_src.env import AirVLNLLMENV
from src.common.llm_wrapper import LLMWrapper, GPT3, GPT4, GPT4O_MINI, LLAMA3, RWKV, QWEN, INTERN, GEMMA2, DEEPSEEKR1_32B, DEEPSEEKR1_8B
from src.common.vlm_wrapper import VLMWrapper, LLAMA3V

class HistoryManager():
    def __init__(self, model_name=GPT4O_MINI):
        self.model_name = model_name
        self.history_actions = []
        self.history_observations = []
        self.history = None
        self.llm = LLMWrapper()
        self.plan = None

    def update(self, action, observation, instructions, log_dir=None):
        self.history_observations.append(observation)
        actions = actions_description.split('\n')
        prompt = """[Task Description]
You are a drone operator. Your task is to update the execution history for the current instruction. Based on the provided navigation instruction, the current scene observation, the planned action (which is about to be executed), and the previous history, generate a concise summary that captures the key observations, the guidance from the navigation instruction, and the actions that have been executed so far. **Assume that the planned action is successfully executed**, and update the history accordingly so that the next usage of history reflects the current situation.


[Important Note]
When the instruction says **"turn right"** (or **"turn left"**) without a specified degree, it means a large turn, usually 90 degrees.
When the instruction says **"turn around"** without a specified degree, it means a large turn, usually 180 degrees.
When the valid action in the list says **"TURN_RIGHT"** (or **"TURN_LEFT"**), it refers to a small 15-degree turn. Be sure to distinguish between these cases.

[Output Format]
Return the result in JSON format with a single key "history". The value should be a brief string summarizing the progress of the current instruction, incorporating the executed planned action.

[Example]
If the inputs are:
- Navigation Instructions: "turn left and move forward until you see a building"
- Current Scene: "There is no building in front of me."
- Planned Action: 1: MOVE_FORWARD (5 meters)
- Previous History: "I have turned left 90 degrees."
Then a valid output could be:
Output:
```json
{{"history": "I have turned left and moved forward 5 meters. I am finding the building."}}
```

[Input]
Navigation Instruction: {navigation_instruction}
Current Scene: {observation}
Planned Action: {action}
Previous History: {history}
"""
        responses_raw = ''
        try: 
            if action is not None:
                self.history_actions.append(action)
                prompt = prompt.format(history=self.history, action=actions[action], observation=observation, navigation_instruction=instructions)
            else:
                prompt = prompt.format(history=self.history, action=None, observation=observation, navigation_instructions=instructions)
            responses_raw = self.llm.request(prompt=prompt, model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            response = json.loads(responses[-1])
            self.history = response['history']
        except Exception as e:
            logger.error(f"Failed to parse response: {responses_raw}")
        
        if log_dir is not None:
            with open(os.path.join(log_dir, 'history.txt'), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(responses_raw)
                f.write("\n---\n")
                f.write(self.history)

    def update_plan(self, plan):
        if plan is not None:
            self.plan = plan

    def get(self):
        return self.history, self.plan

    def get_actions(self):
        return self.history_actions

    def clear(self):
        self.history_actions = []
        self.history_observations = []
        self.history = None
        self.plan = None

actions_description = """0: TASK_FINISH  
1: MOVE_FORWARD (5 meters)
2: TURN_LEFT (15 degrees)
3: TURN_RIGHT (15 degrees)
4: GO_UP (2 meters)
5: GO_DOWN (2 meters)
6: MOVE_LEFT (5 meters)
7: MOVE_RIGHT (5 meters)"""

class LLMParser():
    def __init__(self, model_name=GPT4O_MINI, detector='dino'):
        self.model_name = model_name
        self.detector = detector
        self.llm = LLMWrapper()
    
    def parse_response(self, llm_output, log_dir=None):
        prompt = """[Task Description]Based on the following string, parse the 'Thought', 'Plan', and 'Action' in the output.

'Thought': thoughts about the task, which may include comprehension, surroundings, history, and etc. 'Plan': updated plan. 'Action': next action.

[Actions Description]:
{actions_description}

[Output Format] Provide them in JSON format with the following keys: thoughts, plan, action. And make sure that the 'action' is an integer.
[Output Example]:
```json
{{"thoughts": "...", "plan": "...", "action": ...}}
```"""
        responses_raw = ''
        try:
            responses_raw = self.llm.request(llm_output, prompt.format(actions_description=actions_description), model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            response = json.loads(responses[-1])
            thoughs = response['thoughts']
            plan = response['plan']
            action = response['action']
            try:
                action = int(action)
            except Exception as e:
                logger.error(f"Failed to parse action: {action}")
                action = 1

            if log_dir is not None:
                with open(os.path.join(log_dir, 'parse_response.txt'), 'w+') as f:
                    f.write(self.model_name)
                    f.write("\n---\n")
                    f.write(prompt.format(actions_description=actions_description))
                    f.write("\n---\n")
                    f.write(llm_output)
                    f.write("\n---\n")
                    f.write(responses_raw)
                    f.write("\n---\n")
                    f.write(thoughs)
                    f.write("\n---\n")
                    f.write(plan)
                    f.write("\n---\n")
                    f.write(str(action))

            return thoughs, plan, action
        except Exception as e:
            logger.error(f"Failed to parse response: {responses_raw}")
            logger.error(e)
            action = 1
            return None, None, None
    
    def parse_observation(self, observation, instructions, landmarks=None, log_dir=None):
        prompt = """[Task Description]You are an embodied drone that navigates in the real world. You need to follow the navigation instructions to find the destination and stop. Now, based on the 'Observation', describe the scene you see. Make sure to include the visible landmarks and any other relevant details.

[Observation]:
{observation}

[Output Format] Provide them in JSON format with the following keys: scene. And make sure that the 'scene' is a string.
[Output Example]:
```json
{{"scene": "..."}}
```"""
        prompt = prompt.format(instructions=instructions, observation=observation, landmarks=landmarks)
        responses_raw = ''
        try:
            responses_raw = self.llm.request(prompt, model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            response = json.loads(responses[-1])
            scene = response['scene']
            if log_dir is not None:
                with open(os.path.join(log_dir, 'parse_observation.txt'), 'w+') as f:
                    f.write(self.model_name)
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(responses_raw)
                    f.write("\n---\n")
                    f.write(scene)
            return scene
        except Exception as e:
            logger.error(f"Failed to parse response: {responses_raw}")
            return observation
    
class LLMPlanner():
    def __init__(self, model_name, history_manager: HistoryManager):
        self.model_name = model_name
        self.llm = LLMWrapper()
        self.history_manager = history_manager
        # arxiv:2410.08500
#         self.prompt = """[Task Description]You are an embodied drone that navigates in the real world. You need to explore between some places marked and ultimately find the destination to stop.

# [Input Format]: 'Instruction' is a global, step-by-step detailed guidance. 'History' is your previously executed actions and the scenes you have observed. 'Observation' is the description of the current scene.

# [Output Format] 'Thought': your thoughts about the task, which may include your comprehension, surroundings, history, and etc. 'Plan': your updated plan. 'Action': your next action.

# Think step by step. First, judge by the 'Observation', give a first 'Thought', and depict your orientation. Second, check that if a landmark in the current 'plan' is within 5 meters of your current position, then based on 'Instruction' and the previous 'Plan', update your multi-step 'Plan'. Each plan needs to follow a state word (Completed, In Process, TODO). Finally, judge by the 'Observation' again, and select a specific 'Action' in the action list. Make sure that the 'Action' is an integer in the list, and give the reason.

# 'Action List':
# {actions_description}

# Observation':{scene_description}

# 'Instruction':{navigation_instructions}

# 'History':{history}

# 'Plan':{plan}"""
        self.prompt = """[General Task Description]
You are an embodied drone that navigates in the real world. You need to explore between some places marked and ultimately find the destination to stop. To finish the task, you need to follow the navigation instructions.
For each step, assign a probability distribution over the valid action list (0-7). The action with the highest probability will be executed.

[Hard Constraints]
1. **Valid Actions** (0-7):
    0: TASK_FINISH  
    1: MOVE_FORWARD (5 meters)  
    2: TURN_LEFT (15 degrees)  
    3: TURN_RIGHT (15 degrees)  
    4: GO_UP (2 meters)  
    5: GO_DOWN (2 meters)  
    6: MOVE_LEFT (5 meters)  
    7: MOVE_RIGHT (5 meters)

2. **Probability Rules**:
    - Output probabilities for **ALL 8 actions** (0-7)
    - Sum of probabilities must equal 1.0
    - Higher probability = stronger preference
    - Only if the Current Instruction is finished and the Next Instruction is None, can choose the TASK_FINISH action

3. **Instruction Finish Rule**:
    - If the Current Instruction is finished, set "instruction_finished" to true

4. **Important Note**:
    - When the instruction says **"turn right"** (or **"turn left"**) without a specified degree, it means a large turn, usually 90 degrees(about 6 times).
    - When the instruction says **"turn around"** without a specified degree, it means a large turn, usually 180 degrees(about 12 times).
    - When the valid action in the list says **"TURN_RIGHT"** (or **"TURN_LEFT"**), it refers to a small 15-degree turn. Be sure to distinguish between these cases.

[Output Format]
Strictly output **JSON**:
```json
{{
  "thought": "Analyze the association between the current scene and the current instruction, and consider the next instruction to plan the next action.",
  "probabilities": {{"0": 0.0, "1": 0.3, "2": 0.1, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.6, "7": 0.0}},
  "selected_action": 6,
  "instruction_finished": false
}}
```

[Examples]
---
Example 1:
Previous Instruction: None
Current Instruction: "take off and turn to your right until you face the shore"
Next Instruction: "move forward until you see a building"
Current scene: Now I am near a building but I can't see the shore.
History: I have already taken off and turned to the right of 15 degree.
Valid Output:
```json
{{
  "thought": "I have already taken off and turned to the right, but I can't see the shore. I should turn right again based on the current instruction, and move forward once I can see the shore based on the next instruction.",
  "probabilities": {{"0": 0.0, "1": 0.1, "2": 0.1, "3": 0.3, "4": 0.2, "5": 0.1, "6": 0.1, "7": 0.1}},
  "selected_action": 3,
  "instruction_finished": false
}}
```

Example 2:
Previous Instruction: "take off and turn to your right until you face the shore"
Current Instruction: "move forward until you see a building"
Next Instruction: "turn right until you can see a green roof building"
Current scene: I am near a building and I can see the shore.
History: I have already moved forward 20 meters.
Valid Output:
```json
{{
  "thought": "I can see the building now, so the Current Instruction is finished, I should follow the Next Instruction to turn right to find the green roof building.",
  "probabilities": {{"0": 0.0, "1": 0.1, "2": 0.0, "3": 0.7, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.1}},
  "selected_action": 3,
  "instruction_finished": true
}}
```

[Input]
Previous Instruction: {previous_instruction}
Current Instruction: {current_instruction}
Next Instruction: {next_instruction}
Current scene: {scene_description}
History: {history}
Valid Output:"""

    # def plan(self, navigation_instructions, scene_description, log_dir=None):
    #     history, plan = self.history_manager.get()
    #     prompt = self.prompt.format(actions_description=actions_description, scene_description=scene_description, navigation_instructions=navigation_instructions, history=history, plan=plan)
    #     # system_prompt = self.system_prompt.format(actions_description=actions_description)
    #     response = self.llm.request(prompt, self.model_name)
        
    #     if log_dir is not None:
    #         with open(os.path.join(log_dir, 'plan.txt'), 'w+') as f:
    #             f.write(self.model_name)
    #             f.write("\n---\n")
    #             f.write(prompt)
    #             f.write("\n---\n")
    #             f.write(response)
    #     return response
    def plan(self, navigation_instructions, scene_description, log_dir=None):
        history, plan = self.history_manager.get()
        prompt = self.prompt.format(actions_description=actions_description, scene_description=scene_description, navigation_instructions=navigation_instructions, history=history)
        # system_prompt = self.system_prompt.format(actions_description=actions_description)
        responses_raw = self.llm.request(prompt, model_name=self.model_name)
        return responses_raw
        responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
        response = json.loads(responses[-1])
        thoughs = response['thought']
        probabilities = response['probabilities']
        action = response['selected_action']

        
        if log_dir is not None:
            with open(os.path.join(log_dir, 'plan.txt'), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(responses_raw)
                f.write("\n---\n")
                f.write(thoughs)
                f.write("\n---\n")
                f.write(json.dumps(probabilities))
                f.write("\n---\n")
                f.write(str(action))
        return thoughs, probabilities, action
    
    def plan_split(self, navigation_instructions, scene_description, index, log_dir=None, replan:bool=False):
        history, plan = self.history_manager.get()
        previous_instruction = navigation_instructions[index - 1]
        current_instruction = navigation_instructions[index]
        next_instruction = navigation_instructions[index + 1]
        prompt = self.prompt.format(actions_description=actions_description, scene_description=scene_description, previous_instruction=previous_instruction, current_instruction=current_instruction, next_instruction=next_instruction, history=history)
        # system_prompt = self.system_prompt.format(actions_description=actions_description)
        responses_raw = self.llm.request(prompt, model_name=self.model_name)
        responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
        response = json.loads(responses[-1])
        thoughs = response['thought']
        probabilities = response['probabilities']
        action = response['selected_action']
        finished = response['instruction_finished']
        
        if log_dir is not None:
            file_name = 'plan.txt' if not replan else 'replan.txt'
            with open(os.path.join(log_dir, file_name), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(responses_raw)
                f.write("\n---\n")
                f.write(thoughs)
                f.write("\n---\n")
                f.write(json.dumps(probabilities))
                f.write("\n---\n")
                f.write(str(action))
                f.write("\n---\n")
                f.write(str(finished))
        return thoughs, probabilities, action, finished
    
    def extract_landmarks(self, navigation_instructions: str, log_dir=None):
        prompt = f"""You are a drone operator. Based on the following navigation instructions, extract the key landmarks you may need and provide them in a JSON string array format.

Please follow these steps:
1. Carefully analyze the navigation instructions to identify key landmarks mentioned.
2. Merge any similar landmarks into a single entry (for example, merge "building" and "the second building" into "building").
3. List each key landmark as a string in the JSON array format, ensuring that the landmarks are enclosed in double quotes.

Output Example:
```json
["roof", "building", "road"]
```

Navigation Instructions:
{navigation_instructions}"""
        response = self.llm.request(prompt, model_name=self.model_name)
        landmarks = re.findall(r"```json(?:\w+)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)

        if log_dir is not None:
            with open(os.path.join(log_dir, 'extract_landmarks.txt'), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(response)
                f.write("\n---\n")
                f.write(landmarks[-1])
        return json.loads(landmarks[-1])

class Agent():
    def __init__(self, detector, parser, planner, history, vlm_model=LLAMA3V):
        self.history_manager = HistoryManager(history)
        self.detector = detector
        self.vlm_model = vlm_model
        self.vision = VisionClient(detector, vlm_model=vlm_model)
        self.parser = LLMParser(parser, detector)
        self.planner = LLMPlanner(planner, self.history_manager)
        self.instruction_indexes = [1]

    @property
    def device(self):
        return self._device

    def eval(self):
        pass

    def preprocess(self, observations, log_dir=None):
        self.history_manager.clear()
        instructions = observations['instruction']
        self.instruction_indexes = [1] * len(instructions)
        self.landmarks = []
        if self.detector == 'dino' or self.detector == 'vlm':
            for instruction in instructions:
                self.landmarks.append(self.planner.extract_landmarks(instruction, log_dir=log_dir))
    
    def act(self, observations, prev_actions, step = 0, log_dir=None):
        if log_dir is not None:
            log_dir = os.path.join(log_dir, f'step_{step}')
            os.makedirs(log_dir, exist_ok=True)
            img_path = os.path.join(log_dir, f'{step}.jpg')
        else:
            img_path = None
        actions = []
        instructions = observations['instruction']
        rgbs = observations['rgb']
        def get_scene(instruction, rgb, landmark, log_dir=None):
            if self.detector == 'yolo':
                self.vision.detect_capture(frame=rgb)
                observation = self.vision.get_obj_list()
            elif self.detector == 'dino':
                prompt=" . ".join(landmark)
                self.vision.detect_capture(frame=rgb, prompt=prompt, save_path=os.path.join(log_dir, 'annotated.jpg'))
                observation = self.vision.get_obj_list()
            elif self.detector == 'vlm':
                prompt = """You are a drone operator. Given the first-person view image of the scene and Navigation Instructions, identify the VISIBLE landmarks in the image. For each visible landmark, create a description with the following attributes: 
- **location**: the position of the landmark in the image
- **distance**: the distance of the landmark from the drone
- **size**: the relative size of the landmark
- **details**: any notable details about the landmark

For multiple instances of the same type of landmark, number them sequentially. Output the results in a JSON dictionary where the key is the landmark name, and the value is another dictionary containing the location, distace, size, and details.


Output Format: Provide them in JSON dictionary.
Output Example:
```json
{{
    "Landmark1": {{
        "location": "the left top corner of the image",
        "distance": "far",
        "size": "medium",
        "details": "..."
    }},
    "Landmark2": {{
        "location": "the center of the image",
        "distance": "near",
        "size": "large",
        "details": "..."
    }}
}}
```

Navigation Instructions: 
{navigation_instructions}"""
                prompt = prompt.format(navigation_instructions=instruction, landmarks=landmark)
                observation_raw = self.vision.detect_capture(frame=rgb, prompt=prompt, save_path=img_path)
                observations = re.findall(r"```json(?:\w+)?\n(.*?)```", observation_raw, re.DOTALL | re.IGNORECASE)
                if len(observations) == 0:
                    observation = observation_raw
                else: 
                    observation = json.loads(observations[-1])
            scene = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
            if log_dir is not None and self.detector == 'vlm':
                with open(os.path.join(log_dir, 'scene.txt'), 'w+') as f:
                    f.write(self.vlm_model)
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(observation_raw)
                    f.write("\n---\n")
                    f.write(str(observation))
            if log_dir is not None and self.detector == 'dino':
                with open(os.path.join(log_dir, 'dino.txt'), 'w+') as f:
                    f.write("dino")
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(str(observation))
            return scene
        # if self.detector == 'yolo':
        #     for instruction, rgb, prev_action in zip(instructions, rgbs, prev_actions):
        #         # depth = observation['depth'].cpu().numpy()
        #         response = self.planner.plan(navigation_instructions=instruction, scene_description=scene)
        #         thoughs, plan, action = self.parser.parse_response(response)
        #         self.history_manager.update_plan(plan)
        #         self.history_manager.update(action, scene, instructions=instruction)
        #         actions.append(action)
        # elif self.detector == 'dino': 
        #     for instruction, rgb, prev_action, landmark in zip(instructions, rgbs, prev_actions, self.landmarks):
        #         self.vision.detect_capture(frame=rgb, prompt=" . ".join(landmark))
        #         observation = self.vision.get_obj_list()
        #         scene = self.parser.parse_observation(observation, instructions=instruction, landmarks=self.landmarks)
        #         response = self.planner.plan(navigation_instructions=instruction, scene_description=scene)
        #         thoughs, plan, action = self.parser.parse_response(response)
        #         self.history_manager.update_plan(plan)
        #         self.history_manager.update(action, scene, instructions=instruction)
        #         actions.append(action)
        for i in range(len(instructions)):
            instruction = instructions[i]
            rgb = rgbs[i]
            index = self.instruction_indexes[i]
            instruction = [None] + instruction.split('. ') + [None]
            scene = get_scene(instruction=instruction[index], rgb=rgb, landmark=self.landmarks[i], log_dir=log_dir)
            if self.planner.model_name == DEEPSEEKR1_32B:
                response = self.planner.plan(navigation_instructions=instruction, scene_description=scene, index = index, log_dir=log_dir)
                thoughs, probabilities, action, finished = self.parser.parse_response(response, log_dir=log_dir)
            else:
                thoughs, probabilities, action, finished = self.planner.plan_split(navigation_instructions=instruction, scene_description=scene, index = index, log_dir=log_dir)
            if finished:
                self.instruction_indexes[i] = index + 1
                print(f'Instruction {index} finished')
                index = index + 1
                if index + 1 == len(instruction):
                    action = 0
                elif action == 0:
                    print(f'Wrong finished')
                    thoughs, probabilities, action, finished = self.planner.plan_split(navigation_instructions=instruction, scene_description=scene, index=index, log_dir=log_dir, replan=True)
                self.history_manager.clear()
            # thoughs, plan, action = self.parser.parse_response(response, log_dir=log_dir)
            # self.history_manager.update_plan(plan)
            self.history_manager.update(action, scene, instructions=instruction[index], log_dir=log_dir)
            actions.append(action)
        print(f'Action: {actions}')
        return actions