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
from src.common.vlm_wrapper import MINICPM, LLAMA3V, GPT4O_V, INTERN_VL, QWEN_VL_7B, QWEN_VL_72B
from src.common.agent import AgentConfig, AgentV2

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

    agent_config = AgentConfig()
    agent_config.history['model'] = args.EVAL_LLM
    agent_config.instruction_splitter['model'] = args.EVAL_LLM
    agent_config.judger['model'] = args.EVAL_LLM
    agent_config.parser['model'] = args.EVAL_LLM
    agent_config.planner['model'] = args.EVAL_LLM
    agent_config.perception['model'] = args.EVAL_LLM
    agent_config.perception['vlm_model'] = QWEN_VL_72B
    agent_config.history['type'] = 'summarize'
    agent_config.history['include_thought'] = True
    agent_config.history['include_keypose'] = False
    agent_config.instruction_splitter['enable'] = False
    agent_config.perception['collision_estimation'] = False
    agent_config.judger['use_guidance'] = False
    agent_config.manual_mode = False
    agent_config.check()
    trainer = AgentV2(agent_config)

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
            prev_actions = [None for _ in range(train_env.batch_size)]
            finisheds = [[] for _ in range(train_env.batch_size)]

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


                actions, finished = trainer.act(
                    batch,
                    prev_actions,
                    step=t,
                    log_dir=SAVE_IMAGE_LOG_FOLDER
                )
                for i, finish in enumerate(finished):
                    finisheds[i].append(finish)

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
                infos[t]['finished'] = finisheds[t]
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