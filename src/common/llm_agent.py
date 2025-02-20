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
        prompt = """[[Task Instruction]]
You are a navigation instruction parser. Follow these steps precisely:

1. SENTENCE SEGMENTATION 
- Split input text into individual sentences using periods as separators
- Preserve original wording including leading conjunctions (e.g., "and...")
- Maintain original capitalization and spacing

2. LANDMARK EXTRACTION
- Identify ALL navigational landmarks (physical objects/locations)
- Capture full noun phrases following prepositions: to/at/near/above/before
- Retain modifiers: "small building", "shop entrance", etc.

3. JSON STRUCTURING
- Create array of objects with STRICT format:
{{
  "sub-instruction_[N]": "<original_sentence>",  // N starts at 1
  "landmark": ["<noun_phrase1>", "<noun_phrase2>"] 
}}
- Always use arrays even for single landmarks
- No explanatory text - ONLY valid JSON

[[Critical Requirements]]
✓ Double-check period placement for correct segmentation
✓ Include ALL landmarks per sentence (1-3 typical)
✓ Never omit/modify original wording in sub-instructions
✓ Strictly avoid JSON syntax errors

[[Demonstration]]
Input:
{{ descend toward blue warehouse then circle around its parking lot. avoid the tall crane during approach. }}

Output:
[
  {{
    "sub-instruction_1": "descend toward blue warehouse then circle around its parking lot.",
    "landmark": ["blue warehouse", "parking lot"]
  }},
  {{
    "sub-instruction_2": "avoid the tall crane during approach.",
    "landmark": ["crane"]
  }}
]

[[Your Target]]
Process this navigation instruction:
Input: {{ {navigation_instruction} }}
Output:"""
        prompt = prompt.format(navigation_instruction=navigation_instructions)
        response = self.llm.request(prompt, model_name=self.model_name)
        landmarks = re.findall(r"```json(?:\w+)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if len(landmarks) == 0:
            landmarks = [re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)]
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
    def __init__(self, detector, parser, planner, history, vlm_model=LLAMA3V, manual_mode: bool = False):
        self.history_manager = HistoryManager(history)
        self.detector = detector
        self.vlm_model = vlm_model
        self.manual_mode = manual_mode
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
                prompt = """[ROLE]  
You are an advanced multimodal perception system for a drone executing Vision-Language Navigation (VLN). Your task is to analyze first-person view RGB-D imagery and generate mission-aware environmental semantics for the given [Instruction].

[Processing Requirements:]  
1. Hierarchical Semantic Parsing
	 Detect RELEVANT objects at two levels:
	 a) Primary categories: building, vegetation, vehicle, road, sky
   b) Functional components: e.g. if building detected: ['entrance', 'window', 'balcony', 'roof_antenna']  

2. Spatial Configuration:  
   Bounding box: [x_min, y_min, x_max, y_max] normalized to [0,1]
   Relative position (self-center): left/right/center
   Depth: Metric estimate with confidence interval (22.5m ± 3.2m)
   3D size: {{"width": _, "height": _, "depth": _}} from monocular depth  

3. Navigation-Relevant Taggin
   Relavant_to_instruction: confidence score from 0 to 1

### Example ###
[Instruction]: Proceed to the building with a glass entrance
[OUTPUT FORMAT]
```json
[
    {{
        "object_id": "building_01",
        "primary_category": "building",
        "functional_components": [
            "entrance",
            "window",
            "glass_facade"
        ],
        "spatial_config": {{
            "bbox": [
                0.32,
                0.15,
                0.68,
                0.83
            ],
            "position": "center",
            "depth_estimate": "28.4m ± 2.1",
            "3d_size": {{
                "width": 15.2,
                "height": 32.7,
                "depth": 12.8
            }}
        }},
        "navigation_tags": {{
            "relevant_to_instruction": 0.92
        }}
    }},
    {{
        "object_id": "vehicle_03",
        "primary_category": "vehicle",
        "spatial_config": {{
            "bbox": [
                0.12,
                0.65,
                0.23,
                0.72
            ],
            "position": "left",
            "depth_estimate": "8.7m ± 1.4",
            "3d_size": {{
                "width": 2.3,
                "height": 1.8,
                "depth": 4.1
            }}
        }},
        "navigation_tags": {{
            "relevant_to_instruction": 0.86
        }}
    }},
    {{
        "object_id": "vegetation_12",
        "primary_category": "vegetation",
        "spatial_config": {{
            "bbox": [
                0.78,
                0.45,
                0.89,
                0.55
            ],
            "position": "right",
            "depth_estimate": "14.2m ± 2.8",
            "3d_size": {{
                "width": 5.7,
                "height": 8.2,
                "depth": 5.1
            }}
        }},
        "navigation_tags": {{
            "relevant_to_instruction": 0.23
        }}
    }}
]
```

### Input ###
[Instruction]: {navigation_instructions}
"""
                prompt = prompt.format(navigation_instructions=instruction, landmarks=landmark)
                observation_raw = self.vision.detect_capture(frame=rgb, prompt=prompt, save_path=img_path)
                observations = re.findall(r"```json(?:\w+)?\n(.*?)```", observation_raw, re.DOTALL | re.IGNORECASE)
                if len(observations) == 0:
                    observation = observation_raw
                else: 
                    observation = observations[-1]
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
            # instruction = [None] + instruction.split('. ') + [None]
            instruction = [None] + self.landmarks[i] + [None]
            scene = get_scene(instruction=instruction[index], rgb=rgb, landmark=self.landmarks[i][index - 1]['landmark'], log_dir=log_dir)
            if self.manual_mode:
                action, finished = map(int, input('Enter action and finished: ').split())
                finished = finished == 1
            else: 
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