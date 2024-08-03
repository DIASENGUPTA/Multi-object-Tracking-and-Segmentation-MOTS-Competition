import argparse
import cv2
import os
import pdb
import pickle
import json
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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
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

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskdino_config(cfg)
cfg.merge_from_file("../configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml")
cfg.merge_from_list(["MODEL.WEIGHTS","weights/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"])
cfg.freeze()
from demo.predictor import VisualizationDemo
demo = VisualizationDemo(cfg)

from trackers.multi_tracker_zoo import create_tracker

metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )




@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(720, 1280),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg=True ## segmentation is true if using MaskDINO
    #is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    bs = 1
    from collections import defaultdict
    from mmdet.core import encode_mask_results
    import pycocotools.mask as mask_util
    results=defaultdict(list)

    IMG_ROOT = "/projectnb/cs585bp/hsharma/bdd100k_converted/bdd/images/seg_track_20/val"
    IMG_ROOT = Path(IMG_ROOT)
    # source_cfg_path = "/projectnb/cs585bp/hsharma/bdd100k_converted/bdd/labels/seg_track_20/seg_track_val_cocoformat.json"
    source_cfg_path = "/projectnb/cs585bp/hsharma/bdd100k_converted/bdd/labels/seg_track_20/seg_track_new_test_cocoformat.json"
    source_cfg = json.load(open(source_cfg_path))

    mask_to_bdd = defaultdict(lambda:0) ## default convert
    mask_to_bdd.update({0:1, 1:8, 2:3, 3:7, 5:5, 6:6, 7:4}) ## converting COCO label to closest BDD100k label


    for video in source_cfg['videos']:
        source_path = IMG_ROOT / video['name'] # relative path

        dataset = LoadImages(
            source_path, # source
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
        #model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

        # Create as many strong sort instances as there are video sources
        tracker_list = []
        for i in range(bs):
            tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
            tracker_list.append(tracker, )
            if hasattr(tracker_list[i], 'model'):
                if hasattr(tracker_list[i].model, 'warmup'):
                    tracker_list[i].model.warmup()
        outputs = [None] * bs

        # Run tracking
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
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
                # pdb.set_trace()
                masks = torch.tensor(masks)
                p=torch.Tensor(tensor_list)
                # results['segm_result'].append(masks)
                # results['bbox_result'].append(p)
                class_wise_bbox = [[] for _ in range(8)]
                class_wise = [[] for _ in range(8)]

                for box,seg in zip(p, masks): # convert to BDD100k format before converting
                    class_wise[mask_to_bdd[int(box[-1])]-1].append(seg)
                    class_wise_bbox[mask_to_bdd[int(box[-1])]-1].append(box[:-1])
                results['bbox_result'].append(class_wise_bbox)
                results['segm_result'].append(encode_mask_results(class_wise))
                p = [p]
                    # Process detections
            for i, det in enumerate(p):  # detections per image ## enumerate because of batch size
                #print(i)
                #print(det)
                seen += 1

                _, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                curr_frames[i] = im0

                # txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                # s += '%gx%g ' % im.shape[2:]  # print string
                # imc = im0.copy() if save_crop else im0  # for save_crop

                # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                
                if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                    if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                        tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
                track_result={}
                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # pass detections to strongsort
                    with dt[2]:
                        #print(det)
                        outputs[i] = tracker_list[i].update(det.cpu(), im0)
                        # outputs[i][:,4] Tracking the id's, det[:, 5] Label

                    for loc,j in enumerate(outputs[i][:,4].astype(int)): ## list of id's found this time
                        if int(det[:, 5][loc]) in mask_to_bdd.keys():
                            ans={'bbox':det[loc,:5].detach().numpy(),
                                'label':int(det[:, 5][loc])-1,
                                'segm':masks[loc]}
                            ans['segm'] = mask_util.encode(
                                np.array(ans['segm'][:, :, np.newaxis], order='F',
                            dtype='uint8'))[0]
                            track_result[j] = ans
                results['track_result'].append(track_result)

                    #tracker_list[i].tracker.pred_n_update_all_tracks()
 
                prev_frames[i] = curr_frames[i]
        # break
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    #if update:
    #    strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
    with open("scoring_metric_test.pkl","wb") as file: ## scoring metric
        pickle.dump(results, file)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8n-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[720, 1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
