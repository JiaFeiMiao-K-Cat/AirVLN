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
import json5
import json_repair
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
from utils.vision import Frame, VisionClient
from Model.il_trainer import VLNCETrainer
from Model.utils.tensor_dict import DictTree, TensorDict
from Model.aux_losses import AuxLosses
from Model.utils.tensorboard_utils import TensorboardWriter
from Model.utils.common import append_text_to_image, images_to_video

from src.common.param import args
from src.vlnce_src.env import AirVLNLLMENV
from src.common.llm_wrapper import LLMWrapper, GPT3, GPT4, GPT4O_MINI, LLAMA3, RWKV, QWEN, INTERN, GEMMA2, DEEPSEEKR1_32B, DEEPSEEKR1_8B
from src.common.vlm_wrapper import VLMWrapper, LLAMA3V, image_to_base64


scene_prompt_preprocess = """You are an embodied drone that navigates in the real world. Your task is to generate a JSON request for a vision perception expert to help you plan next action.

The JSON must include the following information:
- "required_information": An object containing:
  - "overall_scene_description": A string instructing the expert to provide a comprehensive description of the overall scene with emphasis on environmental layout, obstacles, and significant spatial details.
  - "objects": A list (array) where each element is an object with the following properties:
      - "name": The name of object.
      - "focus_area": A string specifying which region of the image should receive special attention (for example, "upper left", "center", "lower right", etc.), guiding the expert on where to concentrate their analysis.
  - "additional_guidance": A string instructing the expert to identify any obstacles or hazards present in the scene and to suggest potential navigation actions if applicable.

Your output must be strictly in JSON format, without any additional commentary or explanation.

Instruction: {navigation_instructions}

Current Instruction: {current_instruction}"""

scene_prompt_activate = """[ROLE]  
You are an advanced multimodal perception system for a drone executing Vision-Language Navigation (VLN). Your task is to analyze first-person view RGB-D imagery and generate mission-aware environmental semantics for the given [Instruction].

The JSON must include the following information:
- "scene": An object containing:
  - "objects": An array where each element is an object representing a key element in the scene. Each object should include the following properties:
    - "name": The unique identifier or name of the object.
    - "position": The object's horizontal position relative to the drone. Use one of these categories: "especially left", "left", "somewhat left", "center", "somewhat right", "right", "especially right".
    - "distance": The estimated distance from the drone (e.g. "16m").
    - "bbox_2d": The bounding box [x1, y1, x2, y2] normalized to [0,1]
    - "actions": What actions can the drone take to go to the object.

**Note: If multiple objects share the same "name", differentiate them by appending a unique number to their name (e.g., "vehicle_1", "vehicle_2").**
**Note: giving special priority to the areas specified by "focus_area" properties in [Suggestion]**
**Valid Actions**:
MOVE_FORWARD (5 meters)
TURN_LEFT (45 degrees)
TURN_RIGHT (45 degrees)
GO_UP (2 meters)
GO_DOWN (2 meters)
MOVE_LEFT (5 meters)
MOVE_RIGHT (5 meters)

Your output must strictly be valid JSON without any additional commentary or explanation. 

### Input ###
[Instruction]: {navigation_instructions}

[Suggestion]: {suggestion}"""

class HistoryManager():
    def __init__(self, model_name=GPT4O_MINI):
        self.model_name = model_name
        self.history_actions = []
        self.history_observations = []
        self.history_thoughts = None
        self.history = None
        self.history_raw = None
        self.llm = LLMWrapper()
        self.plan = None

    def update(self, action, observation, instructions, log_dir=None):
        self.history_observations.append(observation)
        actions = actions_description.split('\n')
        prompt = """You are a memory expert. Below is the structured description of a historical scene, the current scene, and the recent action performed. Please update the memory based on these descriptions, distinguishing between new and existing objects, and recording their positions. The output should be in JSON format. 

#### Step 1: Track Object's Status
For each object, determine its current status based on its movement relative to the drone's position. Use the following statuses:
- **new**: Object just discovered by the dron.
- **old**: Object was already in the historical scene.

#### Step 2: Determine Object's Location
For each object, track its location relative to the drone's position. Use one of these categories: "especially left", "left", "somewhat left", "center", "somewhat right", "right", "especially right" and "behind".

#### Step 3: Update Memory
After considering the status and location of each object, update the visual memory in the following format:
- "scene": An object containing:
  - "objects": An array where each element is an object representing a key element in the scene. Each object should include the following properties:
    - "name": Object unique identifier (e.g., "building_1", "vehicle_2", etc.)
    - "status": The determined status from Step 1 ("new" or "old")
    - "position": The determined location from Step 2 (e.g., "center", "left", "right", etc.)
    - "bbox_2d": The bounding box [x1, y1, x2, y2] normalized to [0,1]
    - "actions": What actions can the drone take to go to the object.

**Note: If multiple objects share the same "name", differentiate them by appending a unique number to their name (e.g., "vehicle_1", "vehicle_2").**
**Valid Actions**:
MOVE_FORWARD (5 meters)
TURN_LEFT (45 degrees)
TURN_RIGHT (45 degrees)
GO_UP (2 meters)
GO_DOWN (2 meters)
MOVE_LEFT (5 meters)
MOVE_RIGHT (5 meters)

Your output must strictly be valid JSON without any additional commentary or explanation. 

########

[INPUT]
Previous Memory: {visual_memory}

Current Scene: {scene_description}

Executed Action: {action}"""
        responses_raw = ''
        try: 
            if action is not None:
                self.history_actions.append(actions[action])
                # if len(self.history_actions) > 20:
                #     self.history_actions.pop(0)
                # return
                prompt = prompt.format(visual_memory=self.history, action=actions[action], scene_description=observation, navigation_instruction=instructions)
            else:
                # return
                prompt = prompt.format(visual_memory=self.history, action=None, scene_description=observation, navigation_instructions=instructions)
            responses_raw = self.llm.request_with_history(prompt=prompt, model_name=self.model_name, history_id='visual_memory')
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            if len(responses) == 0:
                response = json_repair.loads(responses_raw)
            else:
                response = json_repair.loads(responses[-1])
            if self.history_raw is not None:
                self.llm.update_history('visual_memory', {"role": "assistant", "content": self.history_raw})
            self.history_raw = responses_raw
            self.history = response
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
                f.write(json.dumps(self.history))
                f.write("\n---\n")
                f.write(str(self.history_actions))

    def update_plan(self, plan):
        if plan is not None:
            self.plan = plan

    def get(self):
        history = {}
        history['executed_actions'] = self.history_actions
        # history['previous_thoughts'] = self.history_thoughts
        history['visual_memory'] = self.history
        return history, self.plan

    def get_memory(self):
        return self.history
    
    def get_actions(self):
        return self.history_actions
    
    def set_history(self, history):
        self.history = history
    
    def set_actions(self, actions):
        self.history_actions = actions

    def clear(self):
        self.history_actions = []
        self.history_observations = []
        self.history_raw = None
        self.history = None
        self.plan = None
        self.llm.clear_history('visual_memory')

actions_description = """TASK_FINISH  
MOVE_FORWARD (5 meters)
TURN_LEFT (45 degrees)
TURN_RIGHT (45 degrees)
GO_UP (2 meters)
GO_DOWN (2 meters)
MOVE_LEFT (5 meters)
MOVE_RIGHT (5 meters)"""

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
            response = json_repair.loads(responses[-1])
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
        prompt = """[ROLE]  
You are an advanced multimodal perception system for a drone. Your task is to analyze observation and generate mission-aware environmental semantics for the given [Instruction].

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

3. Navigation-Relevant Tagging
   Relavant_to_instruction: confidence score from 0 to 1

4. Output Format
    JSON with the following keys:
    - object_id: unique identifier
    - primary_category: primary object category
    - functional_components: list of functional components
    - spatial_config: dictionary with bbox, position, depth_estimate, 3d_size
    - navigation_tags: dictionary with relevant_to_instruction
    **Only output json in markdown codeblocks without explanations.**

### Example ###
[Instruction]: Proceed to the building with a glass entrance
[OUTPUT]:
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
[Instruction]: {instructions}

[Observation]: {observation}"""
        prompt = prompt.format(instructions=instructions, observation=observation, landmarks=landmarks)
        responses_raw = ''
        try:
            responses_raw = self.llm.request(prompt, model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            response = json_repair.loads(responses[-1])
            scene = response
            if log_dir is not None:
                with open(os.path.join(log_dir, 'parse_observation.txt'), 'w+') as f:
                    f.write(self.model_name)
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(responses_raw)
                    f.write("\n---\n")
                    f.write(response)
            return scene
        except Exception as e:
            logger.error(f"Failed to parse response: {responses_raw}")
            logger.error(f"{e}")
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

Now you are at a certain time step [step t],

the input for you includes:

### INPUT

current time step: step t

[whole instruction list]

[current instruction]

[current scene]

[history]: including [Executed actions] and [Visual memory]

######

Now, based on the above INPUT, plan your next action at this time step. 

******* IMPORTANT ********:

**Valid Actions** (0-7):
0: TASK_FINISH
1: MOVE_FORWARD (5 meters)
2: TURN_LEFT (45 degrees)
3: TURN_RIGHT (45 degrees)
4: GO_UP (2 meters)
5: GO_DOWN (2 meters)
6: MOVE_LEFT (5 meters)
7: MOVE_RIGHT (5 meters)

***********************

Your output should include:

### Output

[thought]: tell us why you choose this action, e.g. you can analyze the association between the current scene and the current instruction, consider the whole instruction list and history, etc.

[probabilities]: assign a probability distribution over the valid action list (0-7).

[selected_action]: Explicitly select the action with highest probability.

[execute_times]: How many times does the selected action should be executed.

[questions]: a string, questions you need help about current scene or instruction or planning.

#######

Note!! The Format: Strictly output **JSON.**

A valid output EXAMPLE:
```json
{{
  "thought": "The current instruction is to 'Turn right and check for a red building.' In the current scene, there's a red structure visible to the right. Since the historical actions show I previously moved forward and turned left, executing a right turn now aligns with the instruction to verify the red building's presence. The visual memory confirms no prior red structures were detected until now.",
  "probabilities": {{
    "0(TASK_FINISH)": 0.0,
    "1(MOVE_FORWARD)": 0.1,
    "2(TURN_LEFT)": 0.0,
    "3(TURN_RIGHT)": 0.0,
    "4(GO_UP)": 0.8,
    "5(GO_DOWN)": 0.0,
    "6(MOVE_LEFT)": 0.0,
    "7(MOVE_RIGHT)": 0.1
  }},
  "selected_action": 3,
  "execute_times": 2,
  "questions": "When the red building is aligned? How far is it from the drone?",
}}
```
#############

[More Constraints]

1. **Probability Rules**:
    - Output probabilities for **ALL 8 actions** (0-7)
    - Higher probability = stronger preference
    - Only if the Current Instruction is finished and it is the last instruction, can choose the TASK_FINISH action
2. **Important Note**:
    - When the instruction says **"turn right"** (or **"turn left"**) without a specified degree, it means a large turn, usually 90 degrees(about 2 times).
    - When the instruction says **"turn around"** without a specified degree, it means a large turn, usually 180 degrees(about 4 times).
    - When the valid action in the list says **"TURN_RIGHT"** (or **"TURN_LEFT"**), it refers to a small 15-degree turn. Be sure to distinguish between these cases.

############

EXAMPLE:

### INPUT

{{
  "current_time_step": "step 7",
  "whole_instruction_list": [
    {{
    "sub-instruction_1": "turn right and go down the road.",
    "landmark": ["road"]
    }},
    {{
    "sub-instruction_2": "turn left after the park and visit the park benches along the side.",
    "landmark": ["park", "park benches"]
    }},
    {{
    "sub-instruction_3": "stop by the trees near the benches on the other side of the park.",
    "landmark": ["trees", "benches", "park"]
    }}
  ],
  "current_instruction": "turn right and go down the road.",
  "current_scene":   {{
    "scene": {{
      "overall_scene_description": "The scene depicts a park with a road running through it, featuring trees and benches along the sides. The environment is characterized by a mix of open spaces and dense vegetation, with several obstacles present in the form of parked vehicles and pedestrians.",
      "objects": [
        {{
          "name": "park",
          "category": "vegetation",
          "relative_horizontal_position": "center",
          "distance": "50m",
          "status": "founded"
        }},
        {{
          "name": "road",
          "category": "road",
          "relative_horizontal_position": "left",
          "distance": "20m",
          "status": "approaching"
        }},
        {{
          "name": "park benches_1",
          "category": "furniture",
          "relative_horizontal_position": "lower right",
          "distance": "10m",
          "status": "being searched for"
        }},
        {{
          "name": "trees_1",
          "category": "vegetation",
          "relative_horizontal_position": "somewhat left",
          "distance": "30m",
          "status": "founded"
        }},
        {{
          "name": "vehicles_1",
          "category": "vehicle",
          "relative_horizontal_position": "especially right",
          "distance": "15m",
          "status": "moving away"
        }},
        {{
          "name": "pedestrians_1",
          "category": "person",
          "relative_horizontal_position": "somewhat left",
          "distance": "25m",
          "status": "not visible"
        }}
      ],
      "additional_guidance": "The scene presents several obstacles, including parked vehicles and pedestrians. To navigate safely, it is recommended to proceed with caution and avoid any potential hazards."
    }}
  }},
  "history": {{
    "executed_actions": [
      "4: GO_UP (2 meters)",
      "3: TURN_RIGHT (45 degrees)",
      "3: TURN_RIGHT (45 degrees)",
    ]
  }}
}}

### OUTPUT

```json
{{
  "thought": "The current instruction is to 'turn right and go down the road.' In the current scene, there's a road visible to the left. Since the historical actions show I previously ascended and turned right. The visual memory confirms there was a road at center before, and the road in my left now. I have turned right too much, I should turn left and move forward along the road. ",
  "probabilities": {{
    "0": 0.0,
    "1": 0.1,
    "2": 0.8,
    "3": 0.0,
    "4": 0.0,
    "5": 0.0,
    "6": 0.0,
    "7": 0.1
  }},
  "selected_action": 2,
  "execute_times": 1,
  "questions": "When should I finish this instruction?",
}}
```

############

### INPUT

{input}
"""

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
    def plan(self, navigation_instructions, scene_description, current_instruction, log_dir=None, step=0):
        history, plan = self.history_manager.get()
        input = {}
        input['current_time_step'] = f'step {step}'
        input['whole_instruction_list'] = navigation_instructions
        input['current_instruction'] = current_instruction
        input['current_scene'] = scene_description
        input['history'] = history
        # prompt = self.prompt.format(actions_description=actions_description, scene_description=scene_description, navigation_instructions=navigation_instructions, history=history)
        prompt = self.prompt.format(input=json.dumps(input))
        # system_prompt = self.system_prompt.format(actions_description=actions_description)
        responses_raw = self.llm.request(prompt, model_name=self.model_name)
        return responses_raw
        responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
        response = json_repair.loads(responses[-1])
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
    
    def plan_split(self, navigation_instructions, scene_description, current_instruction, log_dir=None, replan:bool=False, step=0):
        # history, plan = self.history_manager.get()
        # previous_instruction = navigation_instructions[index - 1]
        # current_instruction = navigation_instructions[index]
        # next_instruction = navigation_instructions[index + 1]
        # prompt = self.prompt.format(actions_description=actions_description, scene_description=scene_description, previous_instruction=previous_instruction, current_instruction=current_instruction, next_instruction=next_instruction, history=history)
        # # system_prompt = self.system_prompt.format(actions_description=actions_description)
        history, plan = self.history_manager.get()
        input = {}
        input['current_time_step'] = f'step {step}'
        input['whole_instruction_list'] = navigation_instructions
        input['current_instruction'] = current_instruction
        input['current_scene'] = scene_description
        input['history'] = history
        prompt = self.prompt.format(input=json.dumps(input))
        responses_raw = self.llm.request(prompt, model_name=self.model_name)
        responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
        response = json_repair.loads(responses[-1])
        thoughs = response['thought']
        probabilities = response['probabilities']
        action = response['selected_action']
        question = response['questions']
        
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
                f.write(str(question))
        return thoughs, probabilities, action
    
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
        return json_repair.loads(landmarks[-1])
    
    def finished_judge(self, current_instruction, next_instruction, scene, log_dir=None):
        prompt = """You are a drone navigation analysis expert. You are provided with the following inputs:
1. **Current Instruction**: The command that is currently being executed.
2. **Next Instruction**: The subsequent command that will be executed.
3. **Action History**: A list of actions that have been performed so far.
4. **Current Scene Description**: A detailed description of the current scene.

Your task is to determine whether the current instruction has been fully completed, partially completed, or not completed at all. To make this judgment, analyze the inputs as follows:
- Evaluate if the actions taken (from the action history) align with the directives in the current instruction.
- Consider the current scene description to verify if the expected outcomes of the current instruction are visible.
- Use the next instruction as a clue to see if it implies a transition from the current instruction.
- Summarize relevant evidence from the inputs to support your conclusion.

Output your analysis strictly in valid JSON format with the following structure:
{{
  "instruction_status": "<completed | partially_completed | not_completed>",
  "justification": "<A brief explanation of your decision>",
  "evidence": "<A summary of the relevant details from the current instruction, next instruction, action history, and current scene description that supports your decision>"
}}

Your output must be strictly in JSON codeblock with no additional commentary or explanation.

Current Instruction: {current_instruction}
Next Instruction: {next_instruction}
Action History: {action_history}
Current Scene Description: {scene}
"""
        prompt = prompt.format(current_instruction=current_instruction, next_instruction=next_instruction, action_history=self.history_manager.get_actions(), scene=scene)
        response_raw = self.llm.request(prompt, model_name=self.model_name)
        response = re.findall(r"```json(?:\w+)?\n(.*?)```", response_raw, re.DOTALL | re.IGNORECASE)
        if len(response) == 0:
            try: 
                judge = json_repair.loads(response_raw)
            except Exception as e:
                judge = response_raw
        else:
            judge = json_repair.loads(response[-1])
        if log_dir is not None:
            with open(os.path.join(log_dir, 'judge.txt'), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(response_raw)
                f.write("\n---\n")
                f.write(str(judge))
        return judge


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
        if self.manual_mode:
            with open(os.path.join(log_dir, 'instructions.txt'), 'w+') as f:
                f.write("\n".join(instructions))
            return
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
        depths = observations['depth']
        def get_suggestion(navigation_instructions, current_instruction, log_dir=None):
            prompt = scene_prompt_preprocess.format(navigation_instructions=navigation_instructions, current_instruction=current_instruction)
            response_raw = self.planner.llm.request(prompt, model_name=self.planner.model_name)
            response = re.findall(r"```json(?:\w+)?\n(.*?)```", response_raw, re.DOTALL | re.IGNORECASE)
            if len(response) == 0:
                try: 
                    suggestion = json_repair.loads(response_raw)
                except Exception as e:
                    suggestion = response_raw
            else:
                suggestion = json_repair.loads(response[-1])
            if log_dir is not None:
                with open(os.path.join(log_dir, 'suggestion.txt'), 'w+') as f:
                    f.write(self.planner.model_name)
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(response_raw)
                    f.write("\n---\n")
                    f.write(str(suggestion))
            return suggestion
        def get_scene_with_suggestion(navigation_instructions, current_instruction, rgb, landmark, log_dir=None):
            suggestion = get_suggestion(navigation_instructions, current_instruction, log_dir)
            prompt = scene_prompt_activate.format(navigation_instructions=current_instruction, suggestion=suggestion)
            observation_raw = self.vision.detect_capture(frame=rgb, prompt=prompt, save_path=img_path)
            observations = re.findall(r"```json(?:\w+)?\n(.*?)```", observation_raw, re.DOTALL | re.IGNORECASE)
            if len(observations) == 0:
                observation = observation_raw
                try:
                    observation = json_repair.loads(observation)
                except Exception as e:
                    observation = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
            else: 
                observation = json_repair.loads(observations[-1])
            # scene = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
            if log_dir is not None:
                with open(os.path.join(log_dir, 'scene.txt'), 'w+') as f:
                    f.write(self.vlm_model)
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(observation_raw)
                    f.write("\n---\n")
                    f.write(str(observation))
            return observation
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

3. Navigation-Relevant Tagging
   Relavant_to_instruction: confidence score from 0 to 1

4. Output Format
    JSON with the following keys:
    - object_id: unique identifier
    - primary_category: primary object category
    - functional_components: list of functional components
    - spatial_config: dictionary with bbox, position, depth_estimate, 3d_size
    - navigation_tags: dictionary with relevant_to_instruction
    **Only output json in markdown codeblocks without explanations.**

### Example ###
[Instruction]: Proceed to the building with a glass entrance
[OUTPUT]:
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
                    observation = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
                else: 
                    observation = json_repair.loads(observations[-1])
            # scene = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
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
            scene = observation
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
        def check_collision(depth_img, action, img_width=640, img_height=480, drone_width=0.20, drone_height=0.05, fov=90, distance=6.0):
            # print(depth_img.shape) # (480, 640, 1)
            pixel_angle = fov / img_width
            center_x = img_width // 2
            center_y = img_height // 2
            if action == 1:
                half_angle_x = np.arctan(drone_width / (2 * distance)) * (180 / np.pi)
                half_angle_y = np.arctan(drone_height / (2 * distance)) * (180 / np.pi)
                half_width = math.ceil(half_angle_x / pixel_angle)
                half_height = math.ceil(half_angle_y / pixel_angle)
                for dx in range(-half_width, half_width):
                    for dy in range(-half_height, half_height):
                        x = center_x + dx
                        y = center_y + dy
                        if x < 0 or x >= img_width or y < 0 or y >= img_height:
                            continue
                        if depth_img[y, x] < distance:
                            return True
                return False
            else:
                return False
        for i in range(len(instructions)):
            instruction = instructions[i]
            rgb = rgbs[i]
            depth = depths[i]
            index = self.instruction_indexes[i]
            prev_action = prev_actions[i]
            if prev_action is not None and prev_action[1] > 1:
                frame = Frame(rgb)
                image_to_base64(frame.image, os.path.join(log_dir, f'{step}.jpg'))
                action = prev_action[0]
                actions.append(action)
                prev_actions[i] = [action, prev_action[1] - 1]
                continue
            if self.manual_mode:
                depth_unit8 = (depth*255).astype(np.uint8)
                frame = Frame(rgb)
                cv2.imwrite(os.path.join(log_dir, f'{step}_depth.png'), depth_unit8)
                image_to_base64(frame.image, os.path.join(log_dir, f'{step}.jpg'))
                if check_collision(depth * 100, 1):
                    print('Collision Dangeroous')
                instruction = [None] + instruction.split('. ') + [None]
                action, finished = map(int, input('Enter action and finished: ').split())
                finished = finished == 1
                if finished:
                    self.instruction_indexes[i] = index + 1
                    print(f'Instruction {index} finished')
                with open(os.path.join(log_dir, 'action.txt'), 'w+') as f:
                    f.write("human")
                    f.write("\n---\n")
                    selectd_action = actions_description.split('\n')[action]
                    f.write(f"{selectd_action}")
                    if finished:
                        f.write("\n---\n")
                        f.write(f"{instruction[index]} finished")
                if action == 2 or action == 3:
                    prev_actions[i] = [action, 3]
                else:
                    prev_actions[i] = [action, 1]
                actions.append(action)
                continue
            else: 
                # instruction = [None] + instruction.split('. ') + [None]
                instruction = [None] + self.landmarks[i] + [None]
                current_instruction = self.landmarks[i][index - 1][f'sub-instruction_{index}']
                scene = get_scene_with_suggestion(navigation_instructions=self.landmarks[i], current_instruction=current_instruction, rgb=rgb, landmark=self.landmarks[i][index - 1]['landmark'], log_dir=log_dir)
                if step > 0: 
                    next_instruction = self.landmarks[i][index]
                    next_instruction = next_instruction[f'sub-instruction_{index + 1}'] if next_instruction is not None else None
                    judge = self.planner.finished_judge(current_instruction, next_instruction, scene, log_dir=log_dir)
                    if judge['instruction_status'] == 'completed':
                        self.instruction_indexes[i] = index + 1
                        print(f'Instruction {index} finished')
                        index = index + 1
                        if index + 1 == len(instruction):
                            action = 0
                            actions.append(action)
                            continue
                        current_instruction = self.landmarks[i][index - 1][f'sub-instruction_{index}']
                if self.planner.model_name == DEEPSEEKR1_32B:
                    response = self.planner.plan(navigation_instructions=self.landmarks[i], scene_description=scene, index = index, current_instruction=current_instruction, log_dir=log_dir,step=step)
                    thoughs, probabilities, action = self.parser.parse_response(response, log_dir=log_dir)
                else:
                    thoughs, probabilities, action = self.planner.plan_split(navigation_instructions=self.landmarks[i], current_instruction=current_instruction, scene_description=scene, log_dir=log_dir, step=step)
                if action == 2 or action == 3:
                    prev_actions[i] = [action, 3]
                else:
                    prev_actions[i] = [action, 1]
            # thoughs, plan, action = self.parser.parse_response(response, log_dir=log_dir)
            # self.history_manager.update_plan(plan)
            self.history_manager.update(None if prev_action is None else prev_action[0], scene, instructions=current_instruction, log_dir=log_dir)
            self.history_manager.history_thoughts = thoughs
            actions.append(action)
        print(f'Action: {actions}')
        return actions