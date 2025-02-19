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
from src.common.llm_wrapper import LLMWrapper, GPT3, GPT4, GPT4O, GPT4O_MINI, LLAMA3, RWKV, QWEN, INTERN, GEMMA2, DEEPSEEKR1_32B, DEEPSEEKR1_8B
from src.common.vlm_wrapper import MINICPM, LLAMA3V, GPT4O_V
from src.common.llm_agent import Agent

def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    llm: str,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        llm: llm for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-llm={llm}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", 1, images, fps=fps
        )

def observations_to_image(observation: Dict, draw_depth: bool = False) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb[...,[2,1,0]])

    # draw depth map if observation has depth info. resize to rgb size.
    if draw_depth and ("depth" in observation):
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    frame = egocentric_view

    return frame

def setup():
    init_distributed_mode()

    seed = 100 + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self

def batch_obs(
    observations: List[DictTree]
):
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(obs[sensor])

    return batch

def initialize_env(split='train'):
    train_env = AirVLNLLMENV(batch_size=args.batchSize, split=split)

    return train_env
    
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMPlanner():
    def __init__(self):
        self.llm = LLMWrapper()
        self.model_name = DEEPSEEKR1_32B

        # You are a robot pilot and you should follow the user's instructions to generate a plan to fulfill the task or give advice on user's input if it's not clear or not reasonable.
        self.prompt_plan = """You are a drone operator and you should generate a flight plan based on navigation instructions to complete the task.

Your response should carefully consider the 'actions description', the 'scene description', the 'navigation instructions', and the 'previous actions' if they are provided.

Here is the 'actions description':
{actions_description}

Here is the 'scene description':
{scene_description}

Here is the 'navigation instructions':
{navigation_instructions}

Here is the 'previous actions':
{prev_actions}

Please generate the response with detailed plans and the next action to execute them, following these steps:
1. First, summarize the plan in a clear and structured format, including key steps and their purposes.
2. Provide a detailed explanation of the plan, describing the logic and reasoning behind each step.
3. Generate the next action required to implement the plan, adhering to the following:
   - The action should be represented by an integer.
   - Output the action as a JSON array with a single element and wrap the entire JSON array in a markdown code block.
   - Do not include any additional text, explanations, or comments outside the markdown code block.

'response':"""

    def set_model(self, model_name):
        self.model_name = model_name

    def plan(self, task_description: str, scene_description: Optional[str] = None, prev_actions: Optional[str] = None):
        # by default, the task_description is an action
        if not task_description.startswith("["):
            task_description = "[A] " + task_description
        
        actions_description = """0: STOP, indicates that the task has been completed.
1: MOVE_FORWARD, move forward by 5 meters.
2: TURN_LEFT, turn left by 15 degrees.
3: TURN_RIGHT, turn right by 15 degrees.
4: GO_UP, take off / go up by 2 meters.
5: GO_DOWN, land / go down by 2 meters.
6: MOVE_LEFT, move left by 5 meters.
7: MOVE_RIGHT, move right by 5 meters."""

        prompt = self.prompt_plan.format(
            actions_description=actions_description,
            scene_description=scene_description,
            navigation_instructions=task_description,
            prev_actions=prev_actions)
        
        response = self.llm.request(prompt, model_name=self.model_name)
        code_blocks = re.findall(r"```json(?:\w+)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if (code_blocks is None) or (len(code_blocks) == 0):
            return '[0]'
        return code_blocks[-1]
    def extract_landmarks(self, navigation_instructions: str):
        prompt = f"""You are a drone operator. Based on the following navigation instructions, extract the key landmarks and provide them in a JSON string array format.

Here are the 'navigation instructions':
{navigation_instructions}

Please follow these steps:
1. Carefully analyze the navigation instructions to identify key landmarks mentioned.
2. List each key landmark as a string in the JSON array format.
3. Output the list of key landmarks as a JSON array, ensuring that the landmarks are enclosed in double quotes.

Example output:
```json
["landmark1", "landmark2", "landmark3"]
```"""
        response = self.llm.request(prompt, model_name=GPT4O_MINI)
        landmarks = re.findall(r"```json(?:\w+)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        return json.loads(landmarks[-1])

class LLMEvaluator():
    def __init__(self, llm: str, observation_space, action_space, detector: str = 'yolo'):
        self.observation_space = observation_space
        self.action_space = action_space
        self.detector = detector

        self.planner = LLMPlanner()
        self.planner.set_model(llm)

        self.vision = VisionClient(detector)

        self.landmarks = []

    def eval(self):
        pass

    def preprocess(self, observations, log_dir=None):
        instructions = observations['instruction']
        self.landmarks = []
        if self.detector == 'dino':
            for instruction in instructions:
                self.landmarks.append(self.planner.extract_landmarks(instruction))

    def act(self, observations, prev_actions, step=0):
        actions = []
        instructions = observations['instruction']
        rgbs = observations['rgb']
        if self.detector == 'yolo':
            for instruction, rgb, prev_action in zip(instructions, rgbs, prev_actions):
                # depth = observation['depth'].cpu().numpy()

                self.vision.detect_capture(frame=rgb)

                scene = self.vision.get_obj_list()

                action = self.planner.plan(task_description=instruction, scene_description=scene, prev_actions=prev_action)
                try: 
                    action = json.loads(action)
                    actions.append(int(action[0]))
                except Exception as e:
                    logger.error(f"Failed to parse actions: {action}")
                    actions.append(0)
        elif self.detector == 'dino': 
            for instruction, rgb, prev_action, landmark in zip(instructions, rgbs, prev_actions, self.landmarks):
                # depth = observation['depth'].cpu().numpy()

                self.vision.detect_capture(frame=rgb, prompt=" . ".join(landmark))

                scene = self.vision.get_obj_list()

                action = self.planner.plan(task_description=instruction, scene_description=scene, prev_actions=prev_action)
                try: 
                    action = json.loads(action)
                    actions.append(int(action[0]))
                except Exception as e:
                    logger.error(f"Failed to parse actions: {action}")
                    actions.append(0)
        else:
            raise NotImplementedError()
        print(f'Action: {actions}')
        return actions

def eval_vlnce():
    logger.info(args)

    writer = TensorboardWriter(
        str(Path(args.project_prefix) / 'DATA/output/{}/eval/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        flush_secs=30,
    )

    _eval_checkpoint(
        llm=args.EVAL_LLM,
        writer=writer,
    )

    logger.info("END evaluate")
    if writer is not None:
        try:
            writer.writer.close()
            del writer
        except Exception as e:
            logger.error(e)
    logger.info("END evaluate")


def _eval_checkpoint(
    llm: str,
    writer,
) -> None:
    logger.info(f"LLM: {llm}")


    if args.EVAL_DATASET == 'train':
        train_env = AirVLNLLMENV(batch_size=args.batchSize, split='train')
    elif args.EVAL_DATASET == 'val_seen':
        train_env = AirVLNLLMENV(batch_size=args.batchSize, split='val_seen')
    elif args.EVAL_DATASET == 'val_unseen':
        train_env = AirVLNLLMENV(batch_size=args.batchSize, split='val_unseen')
    elif args.EVAL_DATASET == 'test':
        train_env = AirVLNLLMENV(batch_size=args.batchSize, split='test')
    else:
        raise KeyError


    #
    EVAL_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/results/{}'.format(args.name, args.make_dir_time)
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{llm}_{train_env.split}.json",
    )
    if os.path.exists(fname):
        print("skipping -- evaluation exists.")
        return
    
    # detector = 'vlm'
    detector = 'dino'
    # detector = 'yolo'
    use_agent = True


    if use_agent: 
        # trainer = Agent(detector=detector, parser=GPT4O_MINI, planner=args.EVAL_LLM, history=GPT4O_MINI, vlm_model=LLAMA3V)
        trainer = Agent(detector=detector, parser=args.EVAL_LLM, planner=args.EVAL_LLM, history=args.EVAL_LLM, vlm_model=LLAMA3V)
    else:
        trainer = LLMEvaluator(
            llm=args.EVAL_LLM,
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            detector=detector
        )

    gc.collect()


    #
    stats_episodes = {}
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    with torch.no_grad():
        start_iter = 0
        end_iter = len(train_env.data)
        cnt = 0
        for idx in range(start_iter, end_iter, train_env.batch_size):
            

            if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break
            prev_actions = [[] for _ in range(train_env.batch_size)]

            rgb_frames = [[] for _ in range(train_env.batch_size)]

            episodes = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []

            outputs = train_env.reset()
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations)

            print(f'Batch: {batch["instruction"]}')
            
            if args.SAVE_IMAGE_LOG:
                SAVE_IMAGE_LOG_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/images/{}/{}'.format(args.name, args.make_dir_time, llm)

                print(f'Episode ID: {train_env.batch[0]["episode_id"]}')

                if not os.path.exists(str(SAVE_IMAGE_LOG_DIR / train_env.batch[0]['episode_id'])):
                    os.makedirs(str(SAVE_IMAGE_LOG_DIR / train_env.batch[0]['episode_id']), exist_ok=True)

                SAVE_IMAGE_LOG_FOLDER = Path(SAVE_IMAGE_LOG_DIR) / train_env.batch[0]['episode_id']
            else:
                SAVE_IMAGE_LOG_DIR = None
                SAVE_IMAGE_LOG_FOLDER = None

            ended = False

            if args.SAVE_IMAGE_LOG:
                trainer.preprocess(batch, SAVE_IMAGE_LOG_FOLDER)
            else:
                trainer.preprocess(batch)

            for t in range(int(args.maxAction)):
                logger.info('llm:{} \t {} - {} / {}'.format(llm, idx, t, end_iter, ))


                actions = trainer.act(
                    batch,
                    prev_actions,
                    step=t,
                    log_dir=SAVE_IMAGE_LOG_FOLDER
                )
                max_length = 10
                for i, action in enumerate(actions):
                    prev_actions[i].append(action)
                    if len(prev_actions[i]) > max_length:
                        prev_actions[i].pop(0)

                # Make action and get the new state
                # actions = [temp[0] for temp in actions.numpy()]
                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations)

                logger.info('action: {}'.format(actions))

                # reset envs and observations if necessary
                for i in range(train_env.batch_size):
                    if args.EVAL_GENERATE_VIDEO:
                        frame = observations_to_image(observations[i], infos[i])
                        frame = append_text_to_image(
                            frame, train_env.batch[i]['instruction']['instruction_text']
                        )
                        rgb_frames[i].append(frame)

                    if not dones[i] or skips[i]:
                        continue

                    skips[i] = True
                    pbar.update()

                if np.array(dones).all():
                    ended = True
                    break

            for t in range(int(train_env.batch_size)):
                stats_episodes[str(train_env.batch[t]['episode_id'])] = infos[t]

                EVAL_SAVE_EVERY_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results_every/{}'.format(args.name, args.make_dir_time)
                if not os.path.exists(str(EVAL_SAVE_EVERY_RESULTS_DIR / llm)):
                    os.makedirs(str(EVAL_SAVE_EVERY_RESULTS_DIR / llm), exist_ok=True)

                f_intermediate_result_name = os.path.join(
                    str(EVAL_SAVE_EVERY_RESULTS_DIR / llm),
                    f"{train_env.batch[t]['episode_id']}.json",
                )
                f_intermediate_trajectory = {**infos[t]}
                with open(f_intermediate_result_name, "w") as f:
                    json.dump(f_intermediate_trajectory, f)

                if args.EVAL_GENERATE_VIDEO:
                    EVAL_GENERATE_VIDEO_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/videos/{}'.format(args.name, args.make_dir_time)
                    generate_video(
                        video_option=["disk"],
                        video_dir=str(EVAL_GENERATE_VIDEO_DIR),
                        images=rgb_frames[t],
                        episode_id=train_env.batch[t]['episode_id'],
                        llm=llm,
                        metrics={
                            # "spl": infos[t]['spl'],
                            "ndtw": infos[t]['ndtw'],
                        },
                        tb_writer=writer,
                    )

                logger.info((
                    'result-{} \t' +
                    'distance_to_goal: {} \t' +
                    'success: {} \t' +
                    'ndtw: {} \t' +
                    'sdtw: {} \t' +
                    'path_length: {} \t' +
                    'oracle_success: {} \t' +
                    'steps_taken: {}'
                ).format(
                    t,
                    infos[t]['distance_to_goal'],
                    infos[t]['success'],
                    infos[t]['ndtw'],
                    infos[t]['sdtw'],
                    infos[t]['path_length'],
                    infos[t]['oracle_success'],
                    infos[t]['steps_taken']
                ))

    # end
    pbar.close()


    #
    EVAL_INTERMEDIATE_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results/{}'.format(args.name, args.make_dir_time)
    f_intermediate_name = os.path.join(
        EVAL_INTERMEDIATE_RESULTS_DIR,
        f"stats_llm_{llm}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_INTERMEDIATE_RESULTS_DIR):
        os.makedirs(EVAL_INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f)

    #
    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = {}
        temp_1 = j.copy()

        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items():
            if type(_j) == str or type(_j) == list or type(_j) == dict:
                del temp_1[_i]

        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = (
            sum(v[stat_key] for v in stats_episodes.values())
            / num_episodes
        )

    #
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_llm_{llm}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    checkpoint_num = 1
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")
        writer.add_scalar(f"eval_{train_env.split}_{k}", v, checkpoint_num)

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass


if __name__ == "__main__":
    setup()
    eval_vlnce()

# LLMPlanner().plan("taking off from the rooftops and going down. flying over the main street. and zoom in to the white building. rotating and going up.")