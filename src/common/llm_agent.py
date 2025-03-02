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
  - "objects": A list (array) where each element is an object with the following properties:
      - "name": The name of object.
      - "question": A question regarding the object's status or its impact on the navigation task, guiding the expert to concentrate their analysis.

Your output must be strictly in JSON format, without any additional commentary or explanation.

### Input ###
[History]:
{history}

[Current Instruction]:
{current_instruction}

[Next Instruction]:
{next_instruction}"""

scene_prompt_activate = """[ROLE]  
You are an advanced multimodal perception system for a drone executing Vision-Language Navigation (VLN). Your task is to analyze first-person view RGB image and generate mission-aware environmental semantics for the given [Instruction].

The JSON must include the following information:
- "scene": A string describe the scene according to the image input, including its category, its object(including their size and relative position to you) in the form of "This is a scene of .., in the upper left there is a .., in the right there is a .., etc."

**Note: If multiple objects share the same "name", differentiate them by appending a unique number to their name (e.g., "vehicle_1", "vehicle_2").**
**Note: Only VISIBLE objects can be included in the output.**
**Note: It is crucial to think about each question in [Suggestion].**

Your output must strictly be valid JSON without any additional commentary or explanation. 

### Input ###
[Instruction]:
{navigation_instructions}

[Suggestion]:
{suggestion}"""

class HistoryManager():
    def __init__(self, model_name=GPT4O_MINI):
        self.model_name = model_name
        self.history_actions = []
        self.history_observations = []
        self.history_thoughts = []
        self.history_keyposes = []
        self.history = None
        self.history_raw = None
        self.llm = LLMWrapper()
        self.plan = None

    def update(self, log_dir=None):
        # self.history_observations.append(observation)
        actions = actions_description.split('\n')
        prompt = """You are a memory expert. Below is the structured description of a historical scene, the current scene, and the recent action performed. Please update the memory based on these descriptions. The updated memory should be concise, highlighting only key information while leaving out redundant details. Focus on condensing the history into a shorter version, in a short paragraph, preserving the essential context of past decisions, actions, and the environment.

the input for you includes:
[History]: the history of the previous actions and observations
[Thought]: Why you choose this action
[Action]: Which action you have performed
[Keypose]: Which sub-action in the current instruction are you operating
[Observation]: The current scene

You should:
1) evaluate the new observation and history.
2) update the history with the previous action and the new observation.
3) summarize the updated history in brief. 

Your output must strictly be valid JSON in markdown codeblock without any additional commentary or explanation. 
The JSON must include the following information:
- "history": the updated history after the new observation and previous action.

**Note: The VFOV is 50 degrees and the HFOV is 90 degrees. Think carefully about your position relative to objects.**

### Input ###

[History]:
{history}

[Thought]:
{thought}

[Observation]:
{observation}

[Action]:
{previous_action}

[Keypose]:
{keypose}"""

        responses_raw = ''
        try: 
            # if action is not None:
            #     self.history_actions.append(actions[action])
            #     # if len(self.history_actions) > 20:
            #     #     self.history_actions.pop(0)
            #     # return
            #     prompt = prompt.format(history=self.history, previous_action=actions[action], observation=observation)
            # else:
            #     # return
            #     prompt = prompt.format(history=self.history, previous_action=None, observation=observation)
            # responses_raw = self.llm.request_with_history(prompt=prompt, model_name=self.model_name, history_id='visual_memory')
            history = self.get_tuple(1)
            if len(history) == 0:
                history = {
                    "thought": None,
                    "observation": None,
                    "action": None,
                    "keypose": None
                }
            else:
                history = history[0]
            prompt = prompt.format(history=self.history, thought=history['thought'], observation=history['observation'], previous_action=history['action'], keypose=history['keypose'])
            responses_raw = self.llm.request(prompt, model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            if len(responses) == 0:
                response = json_repair.loads(responses_raw)
            else:
                response = json_repair.loads(responses[-1])
            # if self.history_raw is not None:
            #     self.llm.update_history('visual_memory', {"role": "assistant", "content": self.history_raw})
            self.history_raw = responses_raw
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
    
    def add_tuple(self, thought, observation, action, keypose):
        actions = actions_description.split('\n')
        self.history_thoughts.append(thought)
        self.history_observations.append(observation)
        self.history_actions.append(actions[action])
        self.history_keyposes.append(keypose)
    
    def get_tuple(self, length=3):
        history = []
        thoughts = self.history_thoughts[-length:]
        observations = self.history_observations[-length:]
        actions = self.history_actions[-length:]
        keyposes = self.history_keyposes[-length:]
        for thought, observation, action, keypose in zip(thoughts, observations, actions, keyposes):
            history.append({
                "thought": thought,
                "observation": observation,
                "action": action,
                "keypose": keypose
            })
        return history

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
        self.history_thoughts = []
        self.history_keyposes = []
        self.history_raw = None
        self.history = None
        self.plan = None
        self.llm.clear_history('visual_memory')

actions_description = """TASK_FINISH  
MOVE_FORWARD (5 meters)
TURN_LEFT (15 degrees)
TURN_RIGHT (15 degrees)
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

the input for you includes:

### INPUT

[current instruction]

[next instruction]

[Additional Guidance]: if there is any special attention to avoid collisions.

[history]

[current scene]

######

Now, based on the above INPUT, plan your next action at this time step. 

******* IMPORTANT ********:

**Valid Actions** (0-5):
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

[execute_times]: How many times does the selected action should be executed.

[keypose]: Mention which sub-action in the current instruction are you operating, and do you need another step to finish the sub-action.

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
    "3(TURN_RIGHT)": 0.1,
    "4(GO_UP)": 0.8,
    "5(GO_DOWN)": 0.0,
    "6(MOVE_LEFT)": 0.0,
    "7(MOVE_RIGHT)": 0.0
  }},
  "selected_action": 3,
  "execute_times": 2,
  "keypose": "I am executing the sub-action of turning right as per the initial part of the instruction.",
}}
```
#############

[More Constraints]

1. **Probability Rules**:
    - Output probabilities for **ALL 8 actions** (0-7)
    - Higher probability = stronger preference
    - Only if the Current Instruction is finished and it is the last instruction, can choose the TASK_FINISH action
2. **Important Note**:
    - When the instruction says **"turn right"** (or **"turn left"**) without a specified degree, it means a large turn, usually 90 degrees(about 6 times).
    - When the instruction says **"turn around"** without a specified degree, it means a large turn, usually 180 degrees(about 12 times).
    - If the additional guidance says some actions will collide with objects, you must select other safe action to avoid collision.
    - Do not skip any keypoint mentioned in the instruction.
    - One step may not be enough to finish an action. You can repeat the previous action if necessary.

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
    
    def plan_split(self, scene_description, current_instruction, next_instruction, attention, log_dir=None, replan:bool=False, step=0):
        # history, plan = self.history_manager.get()
        # previous_instruction = navigation_instructions[index - 1]
        # current_instruction = navigation_instructions[index]
        # next_instruction = navigation_instructions[index + 1]
        # prompt = self.prompt.format(actions_description=actions_description, scene_description=scene_description, previous_instruction=previous_instruction, current_instruction=current_instruction, next_instruction=next_instruction, history=history)
        # # system_prompt = self.system_prompt.format(actions_description=actions_description)
        # history, plan = self.history_manager.get()
        # history = self.history_manager.get_tuple(2)
        history = self.history_manager.history
        input = {}
        # input['current_time_step'] = f'step {step}'
        # input['whole_instruction_list'] = navigation_instructions
        input['current_instruction'] = current_instruction
        input['next_instruction'] = next_instruction
        input['history'] = history
        # input['history_actions'] = self.history_manager.get_actions()
        input['current_scene'] = scene_description
        input['additional_guidance'] = attention
        prompt = self.prompt.format(input=json.dumps(input))
        responses_raw = self.llm.request(prompt, model_name=self.model_name)
        responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
        response = json_repair.loads(responses[-1])
        thoughs = response['thought']
        probabilities = response['probabilities']
        action = response['selected_action']
        keypose = response['keypose']
        # question = response['questions']
        
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
                f.write(str(keypose))
                # f.write(str(question))
        return thoughs, keypose, action
    
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
    
    def finished_judge(self, current_instruction, next_instruction, scene, guidance, log_dir=None):
        prompt = """You are a drone navigation analysis expert. You are provided with the following inputs:
[Current Instruction]: The command that is currently being executed.
[Next Instruction]: The subsequent command that will be executed.
[History]: Including the history observations, thoughts, actions and keyposes.
[Current Scene Description]: A detailed description of the current scene.
[Additional Guidance]: Tips for avoiding collisions in the current scene.

Your task is to determine whether the current instruction has been fully completed, partially completed, or not completed at all. To make this judgment, analyze the inputs as follows:
- Consider the current scene description to verify if the expected outcomes of the current instruction or the start of the next instruction are visible.
- Consider the additional guidance to verify surrounding obstacles. 
- Use the next instruction as a clue to see if it implies a transition from the current instruction.
- Summarize relevant evidence from the inputs to support your conclusion.

Output your analysis strictly in valid JSON format with the following structure:
{{
  "instruction_status": "<completed | partially_completed | not_completed>",
  "justification": "<A brief explanation of your decision>",
  "evidence": "<A summary of the relevant details from the current instruction, next instruction, action history, and current scene description that supports your decision>"
}}

Your output must be strictly in JSON codeblock with no additional commentary or explanation.

[Current Instruction]:
{current_instruction}

[Next Instruction]:
{next_instruction}

[History]:
{history}

[Current Scene Description]:
{scene}

[Additional Guidance]:
{guidance}"""
        prompt = prompt.format(current_instruction=current_instruction, next_instruction=next_instruction, action_history=self.history_manager.get_actions(), scene=scene, history=self.history_manager.history, guidance=guidance)
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
        def get_suggestion(current_instruction, next_instruction, reget=False, log_dir=None):
            prompt = scene_prompt_preprocess.format(current_instruction=current_instruction, next_instruction=next_instruction, history=self.history_manager.history)
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
                file_name = "suggestion.txt" if not reget else "suggestion_reget.txt"
                with open(os.path.join(log_dir, file_name), 'w+') as f:
                    f.write(self.planner.model_name)
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(response_raw)
                    f.write("\n---\n")
                    f.write(str(suggestion))
            return suggestion
        def get_scene_with_suggestion(current_instruction, next_instruction, rgb, landmark, reget=False, log_dir=None):
            suggestion = get_suggestion(current_instruction, next_instruction, reget=reget, log_dir=log_dir)
            prompt = scene_prompt_activate.format(navigation_instructions=current_instruction, suggestion=suggestion)
            observation_raw = self.vision.detect_capture(frame=rgb, prompt=prompt, save_path=img_path)
            observations = re.findall(r"```json(?:\w+)?\n(.*?)```", observation_raw, re.DOTALL | re.IGNORECASE)
            if len(observations) == 0:
                observation = observation_raw
                try:
                    observation = json_repair.loads(observation)
                    observation = observation['scene']
                except Exception as e:
                    observation = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
            else: 
                observation = json_repair.loads(observations[-1])
                observation = observation['scene']
            # scene = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
            if log_dir is not None:
                file_name = "scene.txt" if not reget else "scene_reget.txt"
                with open(os.path.join(log_dir, file_name), 'w+') as f:
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
        def check_collision(depth_img, action, img_width=640, img_height=360, drone_width=1.0, drone_height=0.1, fov=90, distance=5.1):
            # print(depth_img.shape) # (360, 640, 1)
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
            elif action == 4:
                height_map = np.zeros_like(depth_img)
                for y in range(img_height):
                    angle_y_tan = np.tan(abs(y - center_y) * pixel_angle * (np.pi / 180))
                    height_map[y] = angle_y_tan * depth_img[y]
                half_angle_x = np.arctan(drone_width / (2 * distance)) * (180 / np.pi)
                half_angle_y = np.arctan(drone_height / (2 * distance)) * (180 / np.pi)
                half_width = math.ceil(half_angle_x / pixel_angle)
                half_width = 10
                height = math.ceil(img_height * 0.05)
                gradient_y = np.gradient(height_map, axis=0)
                # depth_gradient_y = np.gradient(depth_img, axis=0)
                gradient_threshold = 0.01
                for dx in range(-half_width, half_width):
                    x = center_x + dx
                    for dy in range(0, height):
                        y = img_height + dy
                        if x < 0 or x >= img_width or y < 0 or y >= img_height:
                            continue
                        gradient = abs(gradient_y[y, x])
                        # print(f"[{x}, {y}], depth: {depth_img[y, x]}, height: {height_map[y, x]}, gradient_height: {gradient_y[y, x]}, gradient_depth: {depth_gradient_y[y, x]}")
                        if height_map[y, x] < distance and gradient <= gradient_threshold:
                            return True
                return False
            elif action == 5:
                height_map = np.zeros_like(depth_img)
                for y in range(img_height):
                    angle_y_tan = np.tan(abs(y - center_y) * pixel_angle * (np.pi / 180))
                    height_map[y] = angle_y_tan * depth_img[y]
                half_angle_x = np.arctan(drone_width / (2 * distance)) * (180 / np.pi)
                half_angle_y = np.arctan(drone_height / (2 * distance)) * (180 / np.pi)
                half_width = math.ceil(half_angle_x / pixel_angle)
                half_width = 10
                height = math.ceil(img_height * 0.05)
                gradient_y = np.gradient(height_map, axis=0)
                # depth_gradient_y = np.gradient(depth_img, axis=0)
                gradient_threshold = 0.01
                for dx in range(-half_width, half_width):
                    x = center_x + dx
                    for dy in range(-height, 0):
                        y = img_height + dy
                        if x < 0 or x >= img_width or y < 0 or y >= img_height:
                            continue
                        gradient = abs(gradient_y[y, x])
                        # print(f"[{x}, {y}], depth: {depth_img[y, x]}, height: {height_map[y, x]}, gradient_height: {gradient_y[y, x]}, gradient_depth: {depth_gradient_y[y, x]}")
                        if height_map[y, x] < distance and gradient <= gradient_threshold:
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
                if log_dir is not None: 
                    frame = Frame(rgb)
                    image_to_base64(frame.image, os.path.join(log_dir, f'{step}.jpg'))
                    depth_unit8 = (depth*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(log_dir, f'{step}_depth.png'), depth_unit8)
                action = prev_action[0]
                actions.append(action)
                prev_actions[i] = [action, prev_action[1] - 1]
                continue
            if self.manual_mode:
                frame = Frame(rgb)
                depth_unit8 = (depth*255).astype(np.uint8)
                cv2.imwrite(os.path.join(log_dir, f'{step}_depth.png'), depth_unit8)
                image_to_base64(frame.image, os.path.join(log_dir, f'{step}.jpg'))
                if check_collision(depth * 100, 1):
                    print('MOVE_FORWARD Collision Dangeroous')
                if check_collision(depth * 100, 4, distance=2.1):
                    print('GO_UP Collision Dangeroous')
                if check_collision(depth * 100, 5, distance=2.1):
                    print('GO_DOWN Collision Dangeroous')
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
                    prev_actions[i] = [action, 1]
                else:
                    prev_actions[i] = [action, 1]
                actions.append(action)
                continue
            else: 
                if log_dir is not None: 
                    depth_unit8 = (depth*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(log_dir, f'{step}_depth.png'), depth_unit8)
                # instruction = [None] + instruction.split('. ') + [None]
                instruction = [None] + self.landmarks[i] + [None]
                current_instruction = self.landmarks[i][index - 1][f'sub-instruction_{index}']
                next_instruction = self.landmarks[i][index] if index < len(self.landmarks[i]) else None
                scene = get_scene_with_suggestion(current_instruction=current_instruction, next_instruction=next_instruction, rgb=rgb, landmark=self.landmarks[i][index - 1]['landmark'], log_dir=log_dir)
                attention = ""
                if check_collision(depth * 100, 1):
                    attention += "MOVE_FORWARD will collide with objects. "
                else:
                    attention += "MOVE_FORWARD is safe. "
                if check_collision(depth * 100, 4, distance=2.1):
                    attention += "GO_UP will collide with objects. "
                else:
                    attention += "GO_UP is safe. "
                if check_collision(depth * 100, 5, distance=2.1):
                    attention += "GO_DOWN will collide with objects. "
                else:
                    attention += "GO_DOWN is safe. "
                attention += "TURN_LEFT and TURN_RIGHT are safe. "
                if step > 0: 
                    next_instruction_text = next_instruction[f'sub-instruction_{index + 1}'] if next_instruction is not None else None
                    judge = self.planner.finished_judge(current_instruction, next_instruction_text, scene, guidance=attention, log_dir=log_dir)
                    if judge['instruction_status'] == 'completed':
                        self.instruction_indexes[i] = index + 1
                        print(f'Instruction {index} finished')
                        index = index + 1
                        if index + 1 == len(instruction):
                            action = 0
                            actions.append(action)
                            continue
                        current_instruction = self.landmarks[i][index - 1][f'sub-instruction_{index}']
                        scene = get_scene_with_suggestion(current_instruction=current_instruction, next_instruction=next_instruction, rgb=rgb, landmark=self.landmarks[i][index - 1]['landmark'], reget=True, log_dir=log_dir)
                if self.planner.model_name == DEEPSEEKR1_32B:
                    response = self.planner.plan(navigation_instructions=self.landmarks[i], scene_description=scene, index = index, current_instruction=current_instruction, log_dir=log_dir,step=step)
                    thoughs, probabilities, action = self.parser.parse_response(response, log_dir=log_dir)
                else:
                    thoughs, keypose, action = self.planner.plan_split(current_instruction=current_instruction, next_instruction=next_instruction, scene_description=scene, attention=attention, log_dir=log_dir, step=step)
                if action == 2 or action == 3:
                    prev_actions[i] = [action, 1]
                else:
                    prev_actions[i] = [action, 1]
            # thoughs, plan, action = self.parser.parse_response(response, log_dir=log_dir)
            # self.history_manager.update_plan(plan)
            # self.history_manager.history_thoughts = thoughs
            self.history_manager.add_tuple(thought=thoughs, action=action, keypose=keypose, observation=scene)
            self.history_manager.update(log_dir=log_dir)
            actions.append(action)
        print(f'Action: {actions}')
        return actions