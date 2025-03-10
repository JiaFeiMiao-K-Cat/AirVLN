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
from src.common.llm_wrapper import LLMWrapper, GPT3, GPT4, GPT4O_MINI, LLAMA3, RWKV, QWEN, INTERN, GEMMA2, DEEPSEEKR1_32B, DEEPSEEKR1_8B, QWQ_32B_LOCAL, QWEN_2_5_72B
from src.common.vlm_wrapper import VLMWrapper, LLAMA3V, image_to_base64


actions_description = """TASK_FINISH
MOVE_FORWARD (5 meters)
TURN_LEFT (15 degrees)
TURN_RIGHT (15 degrees)
ASCENT (2 meters)
DESCENT (2 meters)
MOVE_LEFT (5 meters)
MOVE_RIGHT (5 meters)"""


class AgentConfig:
    def __init__(self):
        self.instruction_splitter = {
            "enable": True,
            "model": QWEN_2_5_72B
        }
        self.judger = {
            "enable": True,
            "model": QWEN_2_5_72B,
            "use_observation": True,
            "use_guidance": True,
            "use_history": True,
            "use_action": True,
            "use_keypose": True,
        }
        self.history = {
            "enable": True,
            "model": QWEN_2_5_72B,
            "type": "list", # ["list", "tuple", "summarize"]
            "length": 3,
            "include_thought": True,
            "include_observation": True,
            "include_action": True,
            "include_keypose": True
        }
        self.perception = {
            "detector": "vlm",
            "vlm_model": LLAMA3V,
            "model": QWEN_2_5_72B,
            "suggestion": True,
            "collision_estimation": True
        }
        self.planner = {
            "model": QWEN_2_5_72B,
            "instruction_type": "split", #["split", "full"]
            "type": "next_action",
            "include_probabilities": True,
            "include_keypose": True,
            "include_thought": True,
            "include_execute_times": True,
            "use_history": True,
            "use_observation": True,
            "use_guidance": True
        }
        self.parser = {
            "model": QWEN_2_5_72B
        }
        self.manual_mode = False
    def check(self):
        if self.manual_mode:
            return
        if self.instruction_splitter['enable']:
            self.planner['instruction_type'] = "split"
        else:
            self.planner['instruction_type'] = "full"
            self.judger['enable'] = False
        if not self.history['enable']:
            self.planner['use_history'] = False
            self.judger['use_history'] = False
        if not self.perception['collision_estimation']:
            self.judger['use_guidance'] = False
            self.planner['use_guidance'] = False

class InstructionSplitter:
    def __init__(self, global_config: AgentConfig):
        self.global_config = global_config
        config = global_config.instruction_splitter
        self.llm = LLMWrapper()
        self.model_name = config['model']
        self.enable = config['enable']
    def split(self, navigation_instructions: str, log_dir=None):
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
        splited_instructions = re.findall(r"```json(?:\w+)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if len(splited_instructions) == 0:
            splited_instructions = [re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)]
        if log_dir is not None:
            with open(os.path.join(log_dir, 'extract_landmarks.txt'), 'w+') as f:
                f.write(self.model_name)
                f.write("\n---\n")
                f.write(prompt)
                f.write("\n---\n")
                f.write(response)
                f.write("\n---\n")
                f.write(splited_instructions[-1])
        splited_instructions = json_repair.loads(splited_instructions[-1])
        if self.enable:
            return splited_instructions
        else:
            landmarks = []
            for instruction in splited_instructions:
                landmarks.extend(instruction['landmark'])
            landmarks = list(dict.fromkeys(landmarks))
            return [{"sub-instruction_1": navigation_instructions, "landmark": landmarks}]
class HistoryManager:
    def __init__(self, global_config: AgentConfig):
        self.global_config = global_config
        config = global_config.history
        self.model_name = config['model']
        self.enable = config['enable']
        self.type = config['type']
        self.length = config['length']
        self.include_thought = config['include_thought']
        self.include_observation = config['include_observation']
        self.include_action = config['include_action']
        self.include_keypose = config['include_keypose']
        self.history_actions = []
        self.history_observations = []
        self.history_thoughts = []
        self.history_keyposes = []
        self.history = None
        self.history_raw = None
        self.llm = LLMWrapper()
        self.plan = None

    def update(self, log_dir=None):
        if self.type == "list": 
            self.history = self.history_actions
            return
        elif self.type == "tuple":
            self.history = self.get_tuple(self.length)
            return
        # self.history_observations.append(observation)
        actions = actions_description.split('\n')
        input_includes = """[History]: the history of the previous actions and observations
[Observation]: The scene after taken the actions in history
[Action]: The action you *PLAN* to do"""
        input_template = """[History]:
{history}

[Observation]:
{observation}

[Action]:
{action}"""
        if self.include_thought:
            input_includes += "\n[Thought]: Why you choose this action"
            input_template += "\n\n[Thought]:\n{thought}"
        if self.include_keypose:
            input_includes += "\n[KeyPose]: The sub-action in the CURRENT instruction you are operating"
            input_template += "\n\n[KeyPose]:\n{keypose}"
        prompt = """You are a memory expert. Below is the structured description of a historical scene, the current scene, and the recent action. Please update the memory based on these descriptions. The updated memory should be concise, highlighting only key information while leaving out redundant details. Focus on condensing the history into a shorter version, in a short paragraph, preserving the essential context of past decisions, actions, and the environment.

the input for you includes:
""" + input_includes + """

You should:
1) evaluate the new observation and history.
2) update the history with the action and observation.
3) summarize the updated history in brief. 

Your output must strictly be valid JSON codeblock without any additional commentary or explanation. 
The JSON must include the following information:
- "history": the updated history in brief words.

**Note: The VFOV and the HFOV are 90 degrees. Think carefully about your position relative to objects.**

### Input ###

""" + input_template

        responses_raw = ''
        try: 
            history = self.get_tuple(1)
            if len(history) == 0:
                history = {
                    "thought": None,
                    "observation": None,
                    "action": None,
                    "keypose": None
                }
            else:
                if self.include_thought:
                    thought = history[0]['thought']
                else:
                    thought = None
                if self.include_observation:
                    observation = history[0]['observation']
                else:
                    observation = None
                if self.include_action:
                    action = history[0]['action']
                else:
                    action = None
                if self.include_keypose:
                    keypose = history[0]['keypose']
                else:
                    keypose = None
                history = {
                    "thought": thought,
                    "observation": observation,
                    "action": action,
                    "keypose": keypose
                }
            prompt = prompt.format(history=self.history, thought=history['thought'], observation=history['observation'], action=history['action'], keypose=history['keypose'])
            responses_raw = self.llm.request(prompt, model_name=self.model_name)
            responses = re.findall(r"```json(?:\w+)?\n(.*?)```", responses_raw, re.DOTALL | re.IGNORECASE)
            if len(responses) == 0:
                response = json_repair.loads(responses_raw)
            else:
                response = json_repair.loads(responses[-1])
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
            item = {}
            if self.include_observation:
                item['observation'] = observation
            if self.include_action:
                item['action'] = action
            if self.include_thought:
                item['thought'] = thought
            if self.include_keypose:
                item['keypose'] = keypose
            history.append(item)
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

class Judger:
    def __init__(self, global_config: AgentConfig):
        self.global_config = global_config
        config = global_config.judger
        self.llm = LLMWrapper()
        self.model_name = config['model']
        self.enable = config['enable']
        self.use_observation = config['use_observation']
        self.use_guidance = config['use_guidance']
        self.use_history = config['use_history']
        self.use_action = config['use_action']
        self.use_keypose = config['use_keypose']
    def judge(self, current_instruction, next_instruction, scene, guidance, history, log_dir=None):
        input_includes = """[Current Instruction]: The command that is currently being executed
[Next Instruction]: The subsequent command that will be executed"""
        input_template = """[Current Instruction]:
{current_instruction}

[Next Instruction]:
{next_instruction}"""
        judge_steps = ""
        if self.use_history:
            judge_steps += "- Consider the history to verify if all the keypoint of the current instruction has been completed.\n"
            input_includes += "\n[History]: Including the history observations, thoughts, actions and keyposes"
            input_template += "\n\n[History]:\n{history}"
        if self.use_observation:
            judge_steps += "- Consider the current scene description to verify if the expected outcomes of the current instruction or the start of the next instruction are visible.\n"
            input_includes += "\n[Current Scene]: A detailed description of the current scene"
            input_template += "\n\n[Current Scene]:\n{scene}"
        if self.use_guidance:
            judge_steps += "- Consider the additional guidance to infer your relative position with surroundings.\n"
            input_includes += "\n[Additional Guidance]: Tips to avoid collisions and infer your relative position with surroundings"
            input_template += "\n\n[Additional Guidance]:\n{guidance}"
        judge_steps += """- Use the next instruction as a clue to see if the current instruction has been completed.
- Summarize relevant evidence from the inputs to support your conclusion."""
        prompt = """You are a drone navigation analysis expert. Your task is to estimate whether the current instruction has been fully completed, partially completed, or not completed at all. 

You are provided with the following inputs:
""" + input_includes + """

To make your judgment, analyze the inputs as follows:
""" + judge_steps + """

Output your analysis strictly in valid JSON format with the following structure:
{{
  "instruction_status": "<completed | partially_completed | not_completed>",
  "justification": "<A brief explanation of your decision>",
  "evidence": "<A summary of the relevant details from the current instruction, next instruction, action history, and current scene description that supports your decision>"
}}

Your output must be strictly in JSON codeblock with no additional commentary or explanation.

""" + input_template
        prompt = prompt.format(current_instruction=current_instruction, next_instruction=next_instruction, scene=scene, history=history, guidance=guidance)
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

class LLMParser:
    def __init__(self, global_config: AgentConfig):
        self.global_config = global_config
        config = global_config.parser
        pass


class LLMPlanner:
    def __init__(self, global_config: AgentConfig, history_manager: HistoryManager):
        self.global_config = global_config
        config = global_config.planner
        self.model_name = config['model']
        self.instruction_type = config['instruction_type']
        self.type = config['type']
        self.include_probabilities = config['include_probabilities']
        self.include_keypose = config['include_keypose']
        self.include_thought = config['include_thought']
        self.include_execute_times = config['include_execute_times']
        self.use_history = config['use_history']
        self.use_observation = config['use_observation']
        self.use_guidance = config['use_guidance']
        self.llm = LLMWrapper()
        self.history_manager = history_manager
    def plan(self, scene_description, current_instruction, next_instruction, attention, log_dir=None, replan:bool=False, step=0):
        if self.instruction_type == "split":
            input_includes = """[Current Instruction]
[Next Instruction]"""
            input_template = """[Current Instruction]:
{current_instruction}

[Next Instruction]:
{next_instruction}"""
            actions = """**Valid Actions** (1-7):
1: MOVE_FORWARD (5 meters)
2: TURN_LEFT (15 degrees)
3: TURN_RIGHT (15 degrees)
4: ASCENT (2 meters)
5: DESCENT (2 meters)
6: MOVE_LEFT (5 meters)
7: MOVE_RIGHT (5 meters)"""
            prob_rule = """    - Output probabilities for **ALL 7 actions** (1-7)"""
            prob_example = """  "probabilities": {{
    "1(MOVE_FORWARD)": 0.1,
    "2(TURN_LEFT)": 0.0,
    "3(TURN_RIGHT)": 0.1,
    "4(ASCENT)": 0.8,
    "5(DESCENT)": 0.0,
    "6(MOVE_LEFT)": 0.0,
    "7(MOVE_RIGHT)": 0.0
  }},"""
        else:
            input_includes = """[Instruction]"""
            input_template = """[Instruction]:
{current_instruction}"""
            actions = """**Valid Actions** (0-7):
0: TASK_FINISH
1: MOVE_FORWARD (5 meters)
2: TURN_LEFT (15 degrees)
3: TURN_RIGHT (15 degrees)
4: ASCENT (2 meters)
5: DESCENT (2 meters)
6: MOVE_LEFT (5 meters)
7: MOVE_RIGHT (5 meters)"""
            prob_rule = """    - Output probabilities for **ALL 8 actions** (0-7)"""
            prob_example = """  "probabilities": {{
    "0(TASK_FINISH)": 0.0,
    "1(MOVE_FORWARD)": 0.1,
    "2(TURN_LEFT)": 0.0,
    "3(TURN_RIGHT)": 0.1,
    "4(ASCENT)": 0.8,
    "5(DESCENT)": 0.0,
    "6(MOVE_LEFT)": 0.0,
    "7(MOVE_RIGHT)": 0.0
  }},"""
        if self.use_history:
            input_includes += "\n[History]"
            input_template += "\n\n[History]:\n{history}"
        input_includes += "\n[Observation]: The description of current scene."
        input_template += "\n\n[Observation]:\n{scene_description}"
        if self.use_observation:
            input_includes += "\n[Additional Guidance]: Tips to avoid collisions and infer your relative position with surroundings"
            input_template += "\n\n[Additional Guidance]:\n**{additional_guidance}**"
        prompt = """[General Task Description]
You are an embodied drone that navigates in the real world. You need to explore between some places marked and ultimately find the destination to stop. To finish the task, you need to follow the navigation instructions.

the input for you includes:

### INPUT

""" + input_includes + """

######

Now, based on the above INPUT, plan your next action at this time step. 

******* IMPORTANT ********:

""" + actions + """

***********************

Your output should include:

### Output

[thought]: tell us why you choose this action, e.g. you can analyze the association between the current scene and the current instruction, consider the whole instruction list and history, etc.

[probabilities]: assign a probability distribution over the valid action list (1-7).

[selected_action]: Explicitly select the action with highest probability.

[execute_times]: How many times does the selected action should be executed.

[keypose]: Mention which sub-action in the CURRENT instruction are you operating, and do you need another step to finish the sub-action.

#######

Note!! The Format: Strictly output **JSON.**

A valid output EXAMPLE:
```json
{{
  "thought": "...",
""" + prob_example + """
  "selected_action": 4,
  "execute_times": 1,
  "keypose": "I am executing the sub-action of ...",
}}
```
#############

[More Constraints]

1. **Probability Rules**:
""" + prob_rule + """
    - Higher probability = stronger preference
    - If the additional guidance shows some actions will collide with objects, the probabilities of these actions should be 0.
2. **Important Note**:
    - When the instruction says **turn right** (or **turn left**) without a specified degree or reference, it means a large turn, usually 90 degrees(about 6 times).
    - When the instruction says **turn around** without a specified degree or reference, it means a large turn, usually 180 degrees(about 12 times).
    - Do not skip any keypoint mentioned in the instruction.
    - One step may not be enough to finish an action. You can repeat the previous action if necessary.
    - To get a better view or align target, you can select the actions not in the instruction.

############

### INPUT

""" + input_template
        history = self.history_manager.history
        previous_action = self.history_manager.history_actions[-1] if len(self.history_manager.history_actions) > 0 else None
        prompt = prompt.format(current_instruction=current_instruction, next_instruction=next_instruction, scene_description=scene_description, additional_guidance=attention, action=previous_action, history=history)
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

class Perception:
    def __init__(self, global_config: AgentConfig):
        self.global_config = global_config
        config = global_config.perception
        self.llm = LLMWrapper()
        self.detector = config['detector']
        self.model_name = config['model']
        self.vlm_model = config['vlm_model']
        self.suggestion = config['suggestion']
        self.collision_estimation = config['collision_estimation']
        self.vision = VisionClient(self.detector, vlm_model=self.vlm_model)
        self.scene_prompt_preprocess = """You are an embodied drone that navigates in the real world. Your task is to generate a JSON request for a vision perception expert to help you plan next action.

The JSON must include the following information:
- "required_information": An object containing:
  - "objects": A list (array) where each element is an object with the following properties:
      - "name": The name of object.
      - "question": A question regarding the object's status or its impact on the navigation task, guiding the expert to concentrate their analysis.

Your output must be strictly in JSON codeblock, without any additional commentary or explanation.

### Input ###
[History]:
{history}

[Current Instruction]:
{current_instruction}

[Next Instruction]:
{next_instruction}"""

        self.scene_prompt_activate = """You are an advanced multimodal perception system for a drone that navigates in the real world. Your task is to analyze first-person view RGB image and generate mission-aware environmental semantics for the given [Instruction].

The JSON must include the following information:
- "scene": A string describe the scene according to the image input, in the form of "Overall: This is a scene of ... . In the left: ... . In the center: ... . etc."

**Note: Only VISIBLE objects can be included in the output.**
**Note: It is crucial to think about each question in [Suggestion].**

Your output must strictly be valid JSON without any additional commentary or explanation. 

### Input ###
[Instruction]:
{instruction}

[Suggestion]:
{suggestion}"""
        self.scene_prompt_passive = """You are an advanced multimodal perception system for a drone that navigates in the real world. Your task is to analyze first-person view RGB image and generate mission-aware environmental semantics for the given [Instruction].

The JSON must include the following information:
- "scene": A string describe the scene according to the image input, in the form of "Overall: This is a scene of ... . In the left: ... . In the center: ... . etc."

**Note: Only VISIBLE objects can be included in the output.**

Your output must strictly be valid JSON without any additional commentary or explanation. 

### Input ###
[Instruction]:
{instruction}"""
    def get_suggestion(self, current_instruction, next_instruction, history, prev_action=None, reget=False, log_dir=None):
            prompt = self.scene_prompt_preprocess.format(current_instruction=current_instruction, next_instruction=next_instruction, action=prev_action, history=history)
            response_raw = self.llm.request(prompt, model_name=self.model_name)
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
                    f.write(self.model_name)
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(response_raw)
                    f.write("\n---\n")
                    f.write(str(suggestion))
            return suggestion
    def get_scene_with_suggestion(self, current_instruction, next_instruction, rgb, landmark, history, reget=False, log_dir=None, img_path=None):
        if self.suggestion:
            suggestion = self.get_suggestion(current_instruction, next_instruction, history=history, reget=reget, log_dir=log_dir)
            prompt = self.scene_prompt_activate.format(instruction=current_instruction, suggestion=suggestion)
        else:
            prompt = self.scene_prompt_passive.format(instruction=current_instruction)
        observation_raw = self.vision.detect_capture(frame=rgb, prompt=prompt, save_path=img_path)
        observations = re.findall(r"```json(?:\w+)?\n(.*?)```", observation_raw, re.DOTALL | re.IGNORECASE)
        if len(observations) == 0:
            observation = observation_raw
            try:
                observation = json_repair.loads(observation)
                observation = observation['scene']
            except Exception as e:
                # observation = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
                observation = observation_raw
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
    def get_scene(self, instruction, rgb, landmark, log_dir=None, img_path=None):
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
                # observation = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
            else: 
                observation = json_repair.loads(observations[-1])
        # scene = self.parser.parse_observation(observation, instructions=instruction, landmarks=landmark, log_dir=log_dir)
        if log_dir is not None: 
            if self.detector == 'vlm':
                with open(os.path.join(log_dir, 'scene.txt'), 'w+') as f:
                    f.write(self.vlm_model)
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(observation_raw)
                    f.write("\n---\n")
                    f.write(str(observation))
            elif self.detector == 'dino':
                with open(os.path.join(log_dir, 'dino.txt'), 'w+') as f:
                    f.write("dino")
                    f.write("\n---\n")
                    f.write(prompt)
                    f.write("\n---\n")
                    f.write(str(observation))
            else:
                with open(os.path.join(log_dir, 'yolo.txt'), 'w+') as f:
                    f.write("yolo")
                    f.write("\n---\n")
                    f.write(str(observation))
        scene = observation
        return scene
    def _check_collision(self, depth_img, action, img_width=672, img_height=672, drone_width=1.0, drone_height=0.1, fov=90, distance=5.1):
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
            gradient_threshold = 0.02
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
            gradient_threshold = 0.02
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
    def check_collision(self, depth_img):
        attention = ""
        if self._check_collision(depth_img, 1):
            attention += "MOVE_FORWARD will collide with objects. "
        # else:
        #     attention += "MOVE_FORWARD is safe. "
        if self._check_collision(depth_img, 4, distance=2.2):
            attention += "ASCENT will collide with objects. "
        # else:
        #     attention += "ASCENT is safe. "
        if self._check_collision(depth_img, 5, distance=2.2):
            attention += "DESCENT will collide with objects. "
        # else:
        #     attention += "DESCENT is safe. "
        # attention += "TURN_LEFT and TURN_RIGHT are safe. "
        return attention

class AgentV2:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.history_manager = HistoryManager(config)
        self.detector = config.perception['detector']
        self.vlm_model = config.perception['vlm_model']
        self.manual_mode = config.manual_mode
        self.perception = Perception(config)
        self.instruction_splitter = InstructionSplitter(config)
        self.judger = Judger(config)
        self.parser = LLMParser(config)
        self.planner = LLMPlanner(config, self.history_manager)
        self.instruction_indexes = [1]
        if config.planner['type'] == 'action_list':
            self.action_list = []

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
                self.landmarks.append(self.instruction_splitter.split(instruction, log_dir=log_dir))
    
    def act(self, observations, prev_actions, step = 0, log_dir=None):
        if log_dir is not None:
            log_dir = os.path.join(log_dir, f'step_{step}')
            os.makedirs(log_dir, exist_ok=True)
            img_path = os.path.join(log_dir, f'{step}.jpg')
        else:
            img_path = None
        actions = []
        finisheds = []
        instructions = observations['instruction']
        rgbs = observations['rgb']
        depths = observations['depth']
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
                prev_actions[i] = [action, prev_action[1] - 1]
                finisheds.append(False)
                actions.append(action)
                continue
            if self.manual_mode:
                frame = Frame(rgb)
                depth_unit8 = (depth*255).astype(np.uint8)
                cv2.imwrite(os.path.join(log_dir, f'{step}_depth.png'), depth_unit8)
                image_to_base64(frame.image, os.path.join(log_dir, f'{step}.jpg'))
                print(self.perception.check_collision(depth * 100))
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
                finisheds.append(finished)
                actions.append(action)
                continue
            else: 
                finished = False
                if log_dir is not None: 
                    depth_unit8 = (depth*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(log_dir, f'{step}_depth.png'), depth_unit8)
                # instruction = [None] + instruction.split('. ') + [None]
                instruction = [None] + self.landmarks[i] + [None]
                current_instruction = self.landmarks[i][index - 1]
                current_instruction_text = current_instruction[f'sub-instruction_{index}']
                next_instruction = self.landmarks[i][index] if index < len(self.landmarks[i]) else None
                next_instruction_text = next_instruction[f'sub-instruction_{index + 1}'] if next_instruction is not None else None
                scene = self.perception.get_scene_with_suggestion(current_instruction=current_instruction, next_instruction=next_instruction, rgb=rgb, landmark=self.landmarks[i][index - 1]['landmark'], history=self.history_manager.history, log_dir=log_dir, img_path=img_path)
                attention = self.perception.check_collision(depth * 100)
                if step > 0: 
                    judge = self.judger.judge(current_instruction_text, next_instruction_text, scene, guidance=attention, history=self.history_manager.history, log_dir=log_dir)
                    if judge['instruction_status'] == 'completed':
                        self.instruction_indexes[i] = index + 1
                        finished = True
                        print(f'Instruction {index} finished')
                        index = index + 1
                        if index + 1 == len(instruction):
                            action = 0
                            finisheds.append(finished)
                            actions.append(action)
                            continue
                        current_instruction = self.landmarks[i][index - 1]
                        current_instruction_text = current_instruction[f'sub-instruction_{index}']
                        next_instruction = self.landmarks[i][index] if index < len(self.landmarks[i]) else None
                        next_instruction_text = next_instruction[f'sub-instruction_{index + 1}'] if next_instruction is not None else None
                        scene = self.perception.get_scene_with_suggestion(current_instruction=current_instruction, next_instruction=next_instruction, rgb=rgb, landmark=self.landmarks[i][index - 1]['landmark'], history=self.history_manager.history, reget=True, log_dir=log_dir)
                thoughs, keypose, action = self.planner.plan(current_instruction=current_instruction_text, next_instruction=next_instruction_text, scene_description=scene, attention=attention, log_dir=log_dir, step=step)
                if action == 2 or action == 3:
                    prev_actions[i] = [action, 1]
                else:
                    prev_actions[i] = [action, 1]
            # thoughs, plan, action = self.parser.parse_response(response, log_dir=log_dir)
            # self.history_manager.update_plan(plan)
            # self.history_manager.history_thoughts = thoughs
            self.history_manager.add_tuple(thought=thoughs, action=action, keypose=keypose, observation=scene)
            self.history_manager.update(log_dir=log_dir)
            finisheds.append(finished)
            actions.append(action)
        print(f'Action: {actions}')
        return actions, finisheds