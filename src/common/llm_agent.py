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

class HistoryManager():
    def __init__(self, model_name=GPT4O_MINI):
        self.model_name = model_name
        self.history_actions = []
        self.history_observations = []
        self.history_thoughts = None
        self.history = None
        self.llm = LLMWrapper()
        self.plan = None

    def update(self, action, observation, instructions, log_dir=None):
        self.history_observations.append(observation)
        actions = actions_description.split('\n')
        prompt = """[General Task Description]
You are an embodied drone that navigates in the real world. You need to based on the current scene and instruction, update the visual memory to track the status of objects relative to the drone's position. 

You should follow this steps:
- Track the status of objects relative to the drone's actions. For each object, determine if it is:
    - Approaching: Object is moving closer to the drone.
    - Moving away: Object is getting farther from the drone.
    - Searching for: Object is within the field of view but needs further analysis.
    - Founded: Object has been discovered or identified by the drone.
    - Not visible: Object is outside the current field of view.
- Include the approximate distance of each object from the drone (e.g., "10m", "20m", etc.).
- The location should reflect the relative position of the object, such as "center", "left", "right", etc.
- Update the visual memory after each action performed

#### Step 1: Track Object's Status
For each object, determine its current status based on its movement relative to the drone's position. Use the following statuses:
- **Approaching**: Object is getting closer to the drone.
- **Moving away**: Object is moving farther from the drone.
- **Being searched for**: Object is visible but the drone is not yet sure about its exact location.
- **Not visible**: Object is outside the drone's current field of view.
- **Founded**: Object has been discovered or recognized by the drone.

#### Step 2: Update Object's Distance
For each object, determine the approximate distance to the drone after the action is executed. Use the object's depth estimate or calculate based on the drone's movement.

#### Step 3: Determine Object's Location
For each object, track its location relative to the drone's position. Possible locations include:
- **Center**: The object is directly in front of or near the drone.
- **Left**: The object is on the left side of the drone.
- **Right**: The object is on the right side of the drone.
- **Behind**: The object is behind the drone.

#### Step 4: Update Visual Memory
After considering the status, distance, and location of each object, update the visual memory in the following format:

- **Object**: Object name (e.g., "building", "vehicle", etc.)
- **Status**: The determined status from Step 1 (e.g., "approaching", "moving away", "founded", etc.)
- **Location**: The determined location from Step 3 (e.g., "center", "left", "right", etc.)
- **Distance**: The updated distance to the drone after the action (e.g., "10m", "15m", etc.)

############

EXAMPLE:

### INPUT

Current Instruction: turn left after the park and visit the park benches along the side.

Current Scene: [
    {{
        "object_id": "building_01",
        "primary_category": "building",
        "functional_components": ["entrance"],
        "spatial_config": {{
        "bbox": [0.32, 0.15, 0.68, 0.83],
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
        "object_id": "road_01",
        "primary_category": "road",
        "spatial_config": {{
        "bbox": [0.12, 0.65, 0.23, 0.72],
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
    }}
]

Visual Memory: [
    {{
        "object": "road",
        "status": "Approaching",
        "location": "center",
        "distance": "12.4m"
    }},{{
        "object": "building",
        "status": "Moving away",
        "location": "right",
        "distance": "27.4m"
    }},{{
        "object": "park",
        "status": "Searching for",
        "location": "unknown",
        "distance": "unknown"
    }}
]

Action: 2: TURN_LEFT (15 degrees)

### OUTPUT

```json
{{
    "visual_memory": [
        {{
            "object": "road",
            "status": "Moving away",
            "location": "center",
            "distance": "8.7m"
        }},{{
            "object": "building",
            "status": "Approaching",
            "location": "center",
            "distance": "28.4m"
        }},{{
            "object": "park",
            "status": "Searching for",
            "location": "unknow",
            "distance": "unknow"
        }}
    ]
}}
```

############

### INPUT

Current Instruction: {navigation_instruction}

Current Scene: {scene_description}

Visual Memory: {visual_memory}

Action: {action}
"""
        responses_raw = ''
        try: 
            if action is not None:
                self.history_actions.append(actions[action])
                # if len(self.history_actions) > 20:
                #     self.history_actions.pop(0)
                return
                prompt = prompt.format(visual_memory=self.history, action=actions[action], scene_description=observation, navigation_instruction=instructions)
            else:
                return
                prompt = prompt.format(visual_memory=self.history, action=None, scene_description=observation, navigation_instructions=instructions)
            responses_raw = self.llm.request(prompt=prompt, model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            response = json5.loads(responses[-1])
            self.history = response['visual_memory']
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
        history['previous_thoughts'] = self.history_thoughts
        # history['visual_memory'] = self.history
        return history, self.plan

    def get_actions(self):
        return self.history_actions

    def clear(self):
        self.history_actions = []
        self.history_observations = []
        self.history = None
        self.history_thoughts = None
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
[Instruction]: {navigation_instructions}

[Observation]: {observation}"""
        prompt = prompt.format(instructions=instructions, observation=observation, landmarks=landmarks)
        responses_raw = ''
        try:
            responses_raw = self.llm.request(prompt, model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            response = json.loads(responses[-1])
            scene = re
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

[history]: including [Executed actions] and [Previous Thoughts]

######

Now, based on the above INPUT, plan your next action at this time step. 

******* IMPORTANT ********:

**Valid Actions** (0-7):
0: TASK_FINISH
1: MOVE_FORWARD (5 meters)
2: TURN_LEFT (15 degrees)
3: TURN_RIGHT (15 degrees)
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

[instruction_finished]: a bool value, identify if the current instruction has been finished.

#######

Note!! The Format: Strictly output **JSON.**

A valid output EXAMPLE:
```json
{{
  "thought": "The current instruction is to 'Turn right and check for a red building.' In the current scene, there's a red structure visible to the right. Since the historical actions show I previously moved forward and turned left, executing a right turn now aligns with the instruction to verify the red building's presence. The visual memory confirms no prior red structures were detected until now.",
  "probabilities": {{
    "0": 0.0,
    "1": 0.1,
    "2": 0.0,
    "3": 0.0,
    "4": 0.8,
    "5": 0.0,
    "6": 0.0,
    "7": 0.1
  }},
  "selected_action": 3,
  "instruction_finished": false
}}
```
#############

[More Constraints]

1. **Probability Rules**:
    - Output probabilities for **ALL 8 actions** (0-7)
    - Higher probability = stronger preference
    - Only if the Current Instruction is finished and it is the last instruction, can choose the TASK_FINISH action
2. **Instruction Finish Rule**:
    - If the Current Instruction is finished, set "instruction_finished" to true
3. **Important Note**:
    - When the instruction says **"turn right"** (or **"turn left"**) without a specified degree, it means a large turn, usually 90 degrees(about 6 times).
    - When the instruction says **"turn around"** without a specified degree, it means a large turn, usually 180 degrees(about 12 times).
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
  "current_scene": [
    {{
        "object_id": "building_01",
        "primary_category": "building",
        "functional_components": ["entrance"],
        "spatial_config": {{
        "bbox": [0.32, 0.15, 0.68, 0.83],
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
        "object_id": "road_01",
        "primary_category": "road",
        "spatial_config": {{
        "bbox": [0.12, 0.65, 0.23, 0.72],
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
    }}
  ],
  "history": {{
    "executed_actions": [
      "4: GO_UP (2 meters)",
      "3: TURN_RIGHT (15 degrees)",
      "3: TURN_RIGHT (15 degrees)",
      "3: TURN_RIGHT (15 degrees)",
      "3: TURN_RIGHT (15 degrees)",
      "3: TURN_RIGHT (15 degrees)",
      "3: TURN_RIGHT (15 degrees)"
    ],
    "visual_memory": [
      {{
        "object": "road",
        "status": "closing",
        "visible": true,
        "location": "center"
      }},{{
        "object": "building",
        "status": "closing",
        "visible": true,
        "location": "right"
      }}
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
  "instruction_finished": false
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
                    observation = json5.loads(observations[-1])
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
        for i in range(len(instructions)):
            instruction = instructions[i]
            rgb = rgbs[i]
            index = self.instruction_indexes[i]
            if self.manual_mode:
                frame = Frame(rgb)
                image_to_base64(frame.image, os.path.join(log_dir, f'{i}.jpg'))
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
                actions.append(action)
                continue
            else: 
                # instruction = [None] + instruction.split('. ') + [None]
                instruction = [None] + self.landmarks[i] + [None]
                current_instruction = self.landmarks[i][index - 1][f'sub-instruction_{index}']
                scene = get_scene(instruction=instruction[index], rgb=rgb, landmark=self.landmarks[i][index - 1]['landmark'], log_dir=log_dir)
                if self.planner.model_name == DEEPSEEKR1_32B:
                    response = self.planner.plan(navigation_instructions=self.landmarks[i], scene_description=scene, index = index, current_instruction=current_instruction, log_dir=log_dir,step=step)
                    thoughs, probabilities, action, finished = self.parser.parse_response(response, log_dir=log_dir)
                else:
                    thoughs, probabilities, action, finished = self.planner.plan_split(navigation_instructions=self.landmarks[i], current_instruction=current_instruction, scene_description=scene, log_dir=log_dir, step=step)
            if finished:
                self.instruction_indexes[i] = index + 1
                print(f'Instruction {index} finished')
                index = index + 1
                if index + 1 == len(instruction):
                    action = 0
                elif action == 0:
                    print(f'Wrong finished')
                    current_instruction = self.landmarks[i][index - 1][f'sub-instruction_{index}']
                    thoughs, probabilities, action, finished = self.planner.plan_split(navigation_instructions=self.landmarks[i], scene_description=scene, current_instruction=current_instruction, log_dir=log_dir, replan=True)
                self.history_manager.clear()
            # thoughs, plan, action = self.parser.parse_response(response, log_dir=log_dir)
            # self.history_manager.update_plan(plan)
            self.history_manager.update(action, scene, instructions=current_instruction, log_dir=log_dir)
            self.history_manager.history_thoughts = thoughs
            actions.append(action)
        print(f'Action: {actions}')
        return actions