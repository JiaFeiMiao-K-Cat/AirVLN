from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, Union
from numpy.typing import NDArray
import numpy as np
from contextlib import asynccontextmanager
import time, math
import cv2
from filterpy.kalman import KalmanFilter

import json, os
import requests
import queue
import asyncio, aiohttp
import threading
from src.common.vlm_wrapper import VLMWrapper, LLAMA3V


class Frame():
    def __init__(self, image: Image.Image | NDArray[np.uint8]=None, depth: Optional[NDArray[np.int16]]=None):
        if image is None:
            self._image_buffer = np.zeros((352, 640, 3), dtype=np.uint8)
            self._image = Image.fromarray(self._image_buffer)
        if isinstance(image, np.ndarray):
            self._image_buffer = image
            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._image = Image.fromarray(im_rgb)
        elif isinstance(image, Image.Image):
            self._image = image
            self._image_buffer = np.array(image)
        self._depth = depth
    
    @property
    def image(self) -> Image.Image:
        return self._image
    
    @property
    def depth(self) -> Optional[NDArray[np.int16]]:
        return self._depth
    
    @image.setter
    def image(self, image: Image.Image):
        self._image = image
        self._image_buffer = np.array(image)

    @depth.setter
    def depth(self, depth: Optional[NDArray[np.int16]]):
        self._depth = depth

    @property
    def image_buffer(self) -> NDArray[np.uint8]:
        return self._image_buffer
    
    @image_buffer.setter
    def image_buffer(self, image_buffer: NDArray[np.uint8]):
        self._image_buffer = image_buffer
        self._image = Image.fromarray(image_buffer)

class SharedFrame():
    def __init__(self):
        self.timestamp = 0
        self.frame = Frame()
        self.yolo_result = {}
        self.lock = threading.Lock()

    def get_image(self) -> Optional[Image.Image]:
        with self.lock:
            return self.frame.image
    
    def get_yolo_result(self) -> dict:
        with self.lock:
            return self.yolo_result
    
    def get_depth(self) -> Optional[NDArray[np.int16]]:
        with self.lock:
            return self.frame.depth
        
    def set(self, frame: Frame, yolo_result: dict):
        with self.lock:
            self.frame = frame
            self.timestamp = time.time()
            self.yolo_result = yolo_result

from io import BytesIO
from PIL import Image
from typing import Optional, List

import json, sys, os
import queue
import grpc
import asyncio

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_YOLO_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

VISION_SERVICE_IP = os.environ.get("VISION_SERVICE_IP", "localhost")
YOLO_SERVICE_PORT = os.environ.get("YOLO_SERVICE_PORT", "50050").split(",")[0]

'''
Access the YOLO service through gRPC.
'''
class YoloGRPCClient():
    def __init__(self, shared_frame: SharedFrame=None):
        channel = grpc.insecure_channel(f'{VISION_SERVICE_IP}:{YOLO_SERVICE_PORT}')
        self.stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)
        self.is_async_inited = False
        self.image_size = (640, 352)
        self.frame_queue = queue.Queue()
        self.shared_frame = shared_frame
        self.frame_id_lock = asyncio.Lock()
        self.frame_id = 0

    def init_async_channel(self):
        channel_async = grpc.aio.insecure_channel(f'{VISION_SERVICE_IP}:{YOLO_SERVICE_PORT}')
        self.stub_async = hyrch_serving_pb2_grpc.YoloServiceStub(channel_async)
        self.is_async_inited = True

    def is_local_service(self):
        return VISION_SERVICE_IP == 'localhost'

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()

    def retrieve(self) -> Optional[SharedFrame]:
        return self.shared_frame
    
    def detect_local(self, frame: Frame, conf=0.2):
        image = frame.image
        image_bytes = YoloGRPCClient.image_to_bytes(image.resize(self.image_size))
        self.frame_queue.put(frame)

        detect_request = hyrch_serving_pb2.DetectRequest(image_data=image_bytes, conf=conf)
        response = self.stub.DetectStream(detect_request)
        
        json_results = json.loads(response.json_data)
        # {'image_id': 0, 'result': [{'name': 'train_1', 'confidence': 0.31, 'box': {'x1': 0.42, 'y1': 0.42, 'x2': 1.0, 'y2': 0.78}}]}
        if self.shared_frame is not None:
            self.shared_frame.set(self.frame_queue.get(), json_results)

    async def detect(self, frame: Frame, conf=0.1):
        if not self.is_async_inited:
            self.init_async_channel()

        if self.is_local_service():
            self.detect_local(frame, conf)
            return

        image = frame.image
        # do not resize for demo
        image_bytes = YoloGRPCClient.image_to_bytes(image)
        async with self.frame_id_lock:
            image_id = self.frame_id
            self.frame_queue.put((self.frame_id, frame))
            self.frame_id += 1

        detect_request = hyrch_serving_pb2.DetectRequest(image_id=image_id, image_data=image_bytes, conf=conf)
        response = await self.stub_async.Detect(detect_request)
    
        json_results = json.loads(response.json_data)
        if self.frame_queue.empty():
            return
        # discard old images
        while self.frame_queue.queue[0][0] < json_results['image_id']:
            self.frame_queue.get()
        # discard old results
        if self.frame_queue.queue[0][0] > json_results['image_id']:
            return
        if self.shared_frame is not None:
            self.shared_frame.set(self.frame_queue.get()[1], json_results)


def iou(boxA, boxB):
    # Calculate the intersection over union (IoU) of two bounding boxes
    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x2'], boxB['x2'])
    yB = min(boxA['y2'], boxB['y2'])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
    boxBArea = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def euclidean_distance(boxA, boxB):
    centerA = ((boxA['x1'] + boxA['x2']) / 2, (boxA['y1'] + boxA['y2']) / 2)
    centerB = ((boxB['x1'] + boxB['x2']) / 2, (boxB['y1'] + boxB['y2']) / 2)
    return math.sqrt((centerA[0] - centerB[0])**2 + (centerA[1] - centerB[1])**2)


class ObjectInfo:
    def __init__(self, name, x, y, w, h) -> None:
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)

    def __str__(self) -> str:
        return f"{self.name} x:{self.x:.2f} y:{self.y:.2f} width:{self.w:.2f} height:{self.h:.2f}"

class ObjectTracker:
    def __init__(self, name, x, y, w, h) -> None:
        self.name = name
        self.kf_pos = self.init_filter()
        self.kf_siz = self.init_filter()
        self.timestamp = 0
        self.size = None
        self.update(x, y, w, h)

    def update(self, x, y, w, h):
        self.kf_pos.update((x, y))
        self.kf_siz.update((w, h))
        self.timestamp = time.time()

    def predict(self) -> Optional[ObjectInfo]:
        # if no update in 2 seconds, return None
        if time.time() - self.timestamp > 0.5:
            return None
        self.kf_pos.predict()
        self.kf_siz.predict()
        return ObjectInfo(self.name, self.kf_pos.x[0][0], self.kf_pos.x[1][0], self.kf_siz.x[0][0], self.kf_siz.x[1][0])

    def init_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state dimensions (x, y, vx, vy), 2 measurement dimensions (x, y)
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                        [0, 1, 0, 0]])
        kf.R *= 1  # Measurement uncertainty
        kf.P *= 1000  # Initial uncertainty
        kf.Q *= 0.01  # Process uncertainty
        return kf

class VisionSkillWrapper():
    def __init__(self, shared_frame: SharedFrame):
        self.shared_frame = shared_frame
        self.last_update = 0
        self.object_trackers: dict[str, ObjectTracker] = {}
        self.object_list = []
        self.aruco_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
            cv2.aruco.DetectorParameters())
        
    def update(self):
        if self.shared_frame.timestamp == self.last_update:
            return
        self.last_update = self.shared_frame.timestamp
        self.object_list = []
        objs = self.shared_frame.get_yolo_result()['result']
        for obj in objs:
            name = obj['name']
            box = obj['box']
            x = (box['x1'] + box['x2']) / 2
            y = (box['y1'] + box['y2']) / 2
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']
            self.object_list.append(ObjectInfo(name, x, y, w, h))
    def _update(self):
        if self.shared_frame.timestamp == self.last_update:
            return
        self.last_update = self.shared_frame.timestamp

        objs = self.shared_frame.get_yolo_result()['result']

        updated_trackers = {}

        for obj in objs:
            name = obj['name']
            box = obj['box']
            x = (box['x1'] + box['x2']) / 2
            y = (box['y1'] + box['y2']) / 2
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']

            best_match_key = None
            best_match_distance = float('inf')
            
            # Find the best matching tracker
            for key, tracker in self.object_trackers.items():
                if tracker.name == name:
                    existing_box = {
                        'x1': tracker.kf_pos.x[0][0] - tracker.kf_siz.x[0][0] / 2,
                        'y1': tracker.kf_pos.x[1][0] - tracker.kf_siz.x[1][0] / 2,
                        'x2': tracker.kf_pos.x[0][0] + tracker.kf_siz.x[0][0] / 2,
                        'y2': tracker.kf_pos.x[1][0] + tracker.kf_siz.x[1][0] / 2,
                    }
                    distance = euclidean_distance(existing_box, box)
                    if distance < best_match_distance:
                        best_match_distance = distance
                        best_match_key = key

            # Update the best matching tracker or create a new one
            if best_match_key is not None and best_match_distance < 50:  # Threshold can be adjusted
                self.object_trackers[best_match_key].update(x, y, w, h)
                updated_trackers[best_match_key] = self.object_trackers[best_match_key]
            else:
                new_key = f"{name}_{len(self.object_trackers)}"  # Create a unique key
                updated_trackers[new_key] = ObjectTracker(name, x, y, w, h)

        # Replace the old trackers with the updated ones
        self.object_trackers = updated_trackers

        # Create the list of current objects
        self.object_list = []
        to_delete = []
        for key, tracker in self.object_trackers.items():
            obj = tracker.predict()
            if obj is not None:
                self.object_list.append(obj)
            else:
                to_delete.append(key)
        
        # Remove trackers that should be deleted
        for key in to_delete:
            del self.object_trackers[key]
    # def update(self):
    #     if self.shared_frame.timestamp == self.last_update:
    #         return
    #     self.last_update = self.shared_frame.timestamp
    #     objs = self.shared_frame.get_yolo_result()['result'] + self.shared_frame.get_yolo_result()['result_custom']
    #     for obj in objs:
    #         name = obj['name']
    #         box = obj['box']
    #         x = (box['x1'] + box['x2']) / 2
    #         y = (box['y1'] + box['y2']) / 2
    #         w = box['x2'] - box['x1']
    #         h = box['y2'] - box['y1']
    #         if name not in self.object_trackers:
    #             self.object_trackers[name] = ObjectTracker(name, x, y, w, h)
    #         else:
    #             self.object_trackers[name].update(x, y, w, h)
        
    #     self.object_list = []
    #     to_delete = []
    #     for name, tracker in self.object_trackers.items():
    #         obj = tracker.predict()
    #         if obj is not None:
    #             self.object_list.append(obj)
    #         else:
    #             to_delete.append(name)
    #     for name in to_delete:
    #         del self.object_trackers[name]

    def get_obj_list(self) -> str:
        self.update()
        str_list = []
        for obj in self.object_list:
            str_list.append(str(obj))
        return str(str_list).replace("'", '')

    def get_obj_info(self, object_name: str) -> ObjectInfo:
        for _ in range(10):
            self.update()
            for obj in self.object_list:
                if obj.name.startswith(object_name):
                    return obj
            time.sleep(0.2)
        return None

    def is_visible(self, object_name: str) -> Tuple[bool, bool]:
        return self.get_obj_info(object_name) is not None, False

    def object_x(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_x: {object_name} is not in sight', True
        return info.x, False
    
    def object_y(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_y: {object_name} is not in sight', True
        return info.y, False
    
    def object_width(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_width: {object_name} not in sight', True
        return info.w, False
    
    def object_height(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_height: {object_name} not in sight', True
        return info.h, False
    
    def object_distance(self, object_name: str) -> Tuple[Union[int, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_distance: {object_name} not in sight', True
        mid_point = (info.x, info.y)
        FOV_X = 0.42
        FOV_Y = 0.55
        if mid_point[0] < 0.5 - FOV_X / 2 or mid_point[0] > 0.5 + FOV_X / 2 \
        or mid_point[1] < 0.5 - FOV_Y / 2 or mid_point[1] > 0.5 + FOV_Y / 2:
            return 30, False
        depth = self.shared_frame.get_depth().data
        start_x = 0.5 - FOV_X / 2
        start_y = 0.5 - FOV_Y / 2
        index_x = (mid_point[0] - start_x) / FOV_X * (depth.shape[1] - 1)
        index_y = (mid_point[1] - start_y) / FOV_Y * (depth.shape[0] - 1)
        return int(depth[int(index_y), int(index_x)] / 10), False
    
class VisionClient():
    def __init__(self, detector: str = 'yolo', vlm_model: str = LLAMA3V):
        self.shared_frame = SharedFrame()
        self.detector = detector
        self.yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)
        self.dino_client = GroundingDINOClient(shared_frame=self.shared_frame)
        self.vlm_model = vlm_model
        self.vlm_client = VLMWrapper()
        self.vision = VisionSkillWrapper(self.shared_frame)
        self.latest_frame = None
    
    def set_vlm(self, model_name: str):
        self.vlm_model = model_name

    def get_latest_frame(self, plot=False):
        image = self.shared_frame.get_image()
        if plot and image:
            self.vision.update()
            # YoloClient.plot_results_oi(image, self.vision.object_list)
        return image
    
    def detect_capture(self, frame, depth=None, prompt: str = None, save_path = None):
        self.latest_frame = frame
        frame = Frame(frame, depth)
        if self.detector == 'yolo':
            self.yolo_client.detect_local(frame)
            # asyncio.run(self.yolo_client.detect(frame))
        elif self.detector == 'dino': 
            self.dino_client.detect(frame, prompt)
        else:
            return self.vlm_client.request(prompt, model_name=self.vlm_model, image=frame.image, save_path=save_path)
    
    def get_obj_list(self) -> str:
        return self.vision.get_obj_list()

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GroundingDINO'))
from groundingdino.util.inference import load_model, load_image, predict, annotate, box_convert
import cv2
import torch
import groundingdino.datasets.transforms as T

class GroundingDINOClient():
    def __init__(self, shared_frame: SharedFrame=None):
        self.shared_frame = shared_frame
        self.frame_queue = queue.Queue()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_config = os.path.join(base_dir, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
        pretrained_path = os.path.join(base_dir, "pretrained", "groundingdino_swint_ogc.pth")
        self.model = load_model(model_config, pretrained_path)
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.25
    def detect(self, frame: Frame, text_prompt: str):
        image = frame.image
        image.save('image.jpg', 'JPEG')
        image_source, image = load_image('image.jpg')
        self.frame_queue.put(frame)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=self.BOX_TRESHOLD,
            text_threshold=self.TEXT_TRESHOLD
        )
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        logits = logits.numpy()
        # print(boxes)
        # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # cv2.imwrite("annotated_image.jpg", annotated_frame)
        
        if self.shared_frame is not None:
            # {'image_id': 0, 'result': [{'name': 'train_1', 'confidence': 0.31, 'box': {'x1': 0.42, 'y1': 0.42, 'x2': 1.0, 'y2': 0.78}}]}
            result = {}
            result['image_id'] = 0
            result['result'] = []
            for box, logit, phrase in zip(boxes, logits, phrases):
                result['result'].append({
                    'name': phrase,
                    'confidence': logit,
                    'box': {
                        'x1': box[0],
                        'y1': box[1],
                        'x2': box[2],
                        'y2': box[3]
                    }
                })
            # print(result)
            self.shared_frame.set(self.frame_queue.get(), result)

