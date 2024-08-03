import os
import json
import pickle
import glob
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # yolov5 strongsort root directory
ROOT = Path("/projectnb/cs585bp/hsharma/MaskDINO/yolov8_tracking")
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
#if str(ROOT / 'yolov8') not in sys.path:
#    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import sys
sys.path.append('/projectnb/cs585bp/hsharma/MaskDINO')

from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import cv2
import torch
import numpy as np
from tqdm import tqdm

from mmdet.core import encode_mask_results

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskdino_config(cfg)
cfg.merge_from_file("../configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_bdd100k.yaml")
cfg.merge_from_list(["MODEL.WEIGHTS","/projectnb/cs585bp/hsharma/MaskDINO/output_trainbdd100k/model_0039999.pth"])
cfg.freeze()
from demo.predictor import VisualizationDemo
demo = VisualizationDemo(cfg)

from trackers.multi_tracker_zoo import create_tracker

metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

#@torch.no_grad()

source='/projectnb/cs585bp/hsharma/bdd100k_converted/bdd/images/seg_track_20/train/000d4f89-3bcbe37a'
yolo_weights=WEIGHTS / 'yolov8n-seg.pt'  # model.pt path(s),
reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt'  # model.pt path,
tracking_method = "deepocsort"
tracking_config = ROOT / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')
imgsz=(720, 1280)  # inference size (height, width)
conf_thres=0.5  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
device='0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
show_vid=True  # show results
save_txt=True  # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
save_crop=False  # save cropped prediction boxes
save_trajectories=False  # save trajectories for each track
save_vid=False  # save confidences in --save-txt labels
nosave=False  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
update=False  # update all models
project=ROOT / 'runs' / 'track'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok do not increment
line_thickness=2  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
hide_class=False  # hide IDs
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference
vid_stride=1  # video frame-rate stride
retina_masks=False

with torch.no_grad():
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    device = select_device(device)

    is_seg=True ## segmentation is true if using MaskDINO
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size
    stride=16
    bs = 1
    from collections import defaultdict
    import pycocotools.mask as mask_util
    results=defaultdict(list)

    IMG_ROOT = "/projectnb/cs585bp/hsharma/bdd100k_converted/bdd/images/seg_track_20/val"
    IMG_ROOT = Path(IMG_ROOT)
    # source_cfg_path = "/projectnb/cs585bp/hsharma/bdd100k_converted/bdd/labels/seg_track_20/seg_track_val_cocoformat.json"
    source_cfg_path = "/projectnb/cs585bp/hsharma/bdd100k_converted/bdd/labels/seg_track_20/seg_track_val_cocoformat.json"

    source_cfg = json.load(open(source_cfg_path))

    mask_to_bdd = defaultdict(lambda:0) ## default convert
    mask_to_bdd.update({1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7}) ## converting COCO label to closest BDD100k label


    for video in tqdm(source_cfg['videos']):
    #for video in source_cfg_path:
        print(video['name'])
        source_path = IMG_ROOT / video['name'] # relative path

        dataset = LoadImages(
            source_path, # source
            imgsz=imgsz,
            stride=stride,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs

        tracker_list = []
        for i in range(bs):
            tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
            tracker_list.append(tracker, )
            if hasattr(tracker_list[i], 'model'):
                if hasattr(tracker_list[i].model, 'warmup'):
                    tracker_list[i].model.warmup()
        outputs = [None] * bs

    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        curr_frames, prev_frames = [None] * bs, [None] * bs



        for frame_idx, batch in enumerate(dataset):
            path, im1, im0s, vid_cap, s = batch
        #print(im1.shape)
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            with dt[0]:
                im = im1.transpose((2, 0, 1))[::-1]
                im = np.ascontiguousarray(im)
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

        # Inference
            with dt[1]:

                list1=[]
                predictor = DefaultPredictor(cfg)

                predictions = predictor(im1)

                tensor_list = []
                masks=[]
                for i in range(100):
                    list1= [predictions['instances'].pred_boxes[i].tensor.cpu().numpy()[0][0],predictions['instances'].pred_boxes[i].tensor.cpu().numpy()[0][1],predictions['instances'].pred_boxes[i].tensor.cpu().numpy()[0][2],predictions['instances'].pred_boxes[i].tensor.cpu().numpy()[0][3]]
                    list2=np.append(list1,predictions['instances'].scores[i].cpu().numpy())
                    list3=np.append(list2,predictions['instances'].pred_classes[i].cpu().numpy())
                    if(list3[4]>=0.5): # higher confidence
                        tensor_list.append(list3)
                        masks.append(predictions['instances'].pred_masks[i].cpu().numpy())
                        #print(predictions['instances'].pred_masks[i].cpu().numpy().shape)
            # pdb.set_trace()
                masks = torch.tensor(masks)
                p=torch.Tensor(tensor_list).cpu()
            # results['segm_result'].append(masks)
                results['bbox'].append(p)
                class_wise = [[] for _ in range(9)]
            
                for box,seg in zip(p, masks): # convert to BDD100k format before converting
                    class_wise[mask_to_bdd[int(box[-1])]].append(seg)
                    #print(seg.shape)
                results['segm_result'].append(encode_mask_results(class_wise))
                p = [p]
                # Process detections
            for i, det in enumerate(p):  # detections per image ## enumerate because of batch size
            #print(i)
            #print(det)
                #condition = (det[:, 1] != det[:, 3]) | (det[:, 0] != det[:, 2])
                #det1 = det[condition]
                #print(len(det1))
                #print(det1)
                seen+=1
                print(path)
                _, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                curr_frames[i] = im0

            
                if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                    if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                        tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])


                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    condition = (det[:, 1] != det[:, 3]) & (det[:, 0] != det[:, 2])
                    det = det[condition]
                # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                    with dt[2]:
                    #print(type(det))
                        print(len(det))
                        print("Hi")
                        print(det)
                        outputs[i] = tracker_list[i].update(det.cpu(), im0)
                    # outputs[i][:,4] Tracking the id's, det[:, 5] Label
                        #print(outputs)
                    track_result={}
                    #print(outputs[i])
                    if(len(outputs[i])>0):
                        for loc,j in enumerate(outputs[i][:,4].astype(int)): ## list of id's found this time
                            if int(det[:, 5][loc]) in mask_to_bdd.keys():
                                ans={'bbox':det[loc,:4].detach().numpy(), ## add confidence here
                                    'label':int(det[:, 5][loc]),
                                    'segm':masks[loc]}
                                ans['segm'] = mask_util.encode(
                                    np.array(ans['segm'][:, :, np.newaxis], order='F',
                                    dtype='uint8'))[0]
                                track_result[j] = ans
                results['track_result'].append(track_result)
                prev_frames[i] = curr_frames[i]
                #print(results)

        # break
    # break
### collect data for all track together, /projectnb/cs585bp/hsharma/pcan/pcan/core/mask/utils.py has mask compression



with open("scoring_metric_MOTS_maskdino_val.pkl","wb") as file: ## scoring metric
    pickle.dump(results, file)




# BDD100K categories
# {"supercategory": "human", "id": 1, "name": "pedestrian"}, 
# {"supercategory": "human", "id": 2, "name": "rider"}, 
# {"supercategory": "vehicle", "id": 3, "name": "car"}, 
# {"supercategory": "vehicle", "id": 4, "name": "truck"},
#  {"supercategory": "vehicle", "id": 5, "name": "bus"}, 
# {"supercategory": "vehicle", "id": 6, "name": "train"},
# {"supercategory": "bike", "id": 7, "name": "motorcycle"}, 
# {"supercategory": "bike", "id": 8, "name": "bicycle"}
# MaskDINO categories: 
# {0: 'person',
#  1: 'bicycle',
#  2: 'car',
#  3: 'motorcycle',
#  4: 'airplane',
#  5: 'bus',
#  6: 'train',
#  7: 'truck',
#  8: 'boat',
#  9: 'traffic light',
#  10: 'fire hydrant',
#  11: 'stop sign',
## there is no rider 

# import pickle
# import json

# output = pickle.load(open("/projectnb/cs585bp/hsharma/MaskDINO/yolov8_tracking/scoring_metric.pkl", "rb"))
# bbox = output['bbox'][0]
# seg_mask = output['segm_result'][0]

# class_wise = [[] for _ in range(9)]

# for box,seg in zip(bbox, seg_mask): # convert to BDD100k format before converting
#    class_wise[mask_to_bdd[int(box[-1])]].append(seg)