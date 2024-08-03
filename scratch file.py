# import json
# import os

# def load_bdd100k_coco_format(json_file, image_root):
#     with open(json_file, "r") as f:
#         data = json.load(f)

#     dataset_dicts = []
#     for img_ann in data['images']:
#         record = {}
#         record['file_name'] = os.path.join(image_root, img_ann['file_name'])
#         record['height'] = img_ann['height']
#         record['width'] = img_ann['width']
#         record['image_id'] = img_ann['id']
        
#         annotations = []
#         for ann in data['annotations']:
#             if ann['image_id'] == img_ann['id']:
#                 annotation = {}
#                 annotation['bbox'] = ann['bbox']
#                 annotation['bbox_mode'] = BoxMode.XYWH_ABS
#                 annotation['category_id'] = ann['category_id']
#                 if 'segmentation' in ann:
#                     annotation['segmentation'] = ann['segmentation']
#                 annotations.append(annotation)
                
#         record['annotations'] = annotations
#         dataset_dicts.append(record)
        
#     return dataset_dicts

# from detectron2.data.datasets import register_coco_instances

# register_coco_instances("bdd100k_seg_train", {}, "bdd100k_converted/bdd/labels/seg_track_20/seg_track_train_cocoformat.json","bdd100k_converted/bdd/images/seg_track_20/train")
# register_coco_instances("bdd100k_seg_val", {}, "bdd100k_converted/bdd/labels/seg_track_20/seg_track_val_cocoformat.json","bdd100k_converted/bdd/images/seg_track_20/val")

# from detectron2.config import get_cfg

# cfg = get_cfg()
# cfg.DATASETS.TRAIN = ("bdd100k_mots",)
# cfg.DATASETS.TEST = ("bdd100k_mots",)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Set the number of classes for your dataset

# python demo.py --config-file ../configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml  --input ../../bdd100k_converted/bdd/images/seg_track_20/train/000d4f89-3bcbe37a/000d4f89-3bcbe37a-0000001.jpg --opts MODEL.WEIGHTS ../weights/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth
import sys
sys.path.insert(0, '/projectnb/cs585bp/hsharma/MaskDINO')
#sys.path.insert(1, '/projectnb/cs585bp/hsharma/detectron2')
#sys.path.insert(2, '/projectnb/cs585bp/hsharma/maskdino')
#sys.path.insert(2, '/projectnb/cs585bp/hsharma/detectron2/detectron2')
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from ..maskdino import add_maskdino_config
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import cv2
import torch
import numpy as np

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskdino_config(cfg)
cfg.merge_from_file("configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml")
cfg.merge_from_list(["MODEL.WEIGHTS","weights/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"])
cfg.freeze()
from demo.predictor import VisualizationDemo
demo = VisualizationDemo(cfg)

img = read_image("../bdd100k_converted/bdd/images/seg_track_20/train/000d4f89-3bcbe37a/000d4f89-3bcbe37a-0000001.jpg", format="BGR")
list1=[]
predictor = DefaultPredictor(cfg)
predictions = predictor(img)
#print(predictions)
list1= [predictions['instances'].pred_boxes[0].tensor.cpu().numpy()[0][0],predictions['instances'].pred_boxes[0].tensor.cpu().numpy()[0][2],predictions['instances'].pred_boxes[0].tensor.cpu().numpy()[0][1],predictions['instances'].pred_boxes[0].tensor.cpu().numpy()[0][3]]
list2=np.append(list1,predictions['instances'].scores[0].cpu().numpy())
list3=np.append(list2,predictions['instances'].pred_classes[0].cpu().numpy())
list4=predictions['instances'].pred_masks[0].cpu().numpy()
print(list3)

tensor_list = []
#tensor_list.append(torch.zeros((100, 6)))
for i in range(100):
    list1= [predictions['instances'].pred_boxes[i].tensor.cpu().numpy()[0][0],predictions['instances'].pred_boxes[i].tensor.cpu().numpy()[0][2],predictions['instances'].pred_boxes[i].tensor.cpu().numpy()[0][1],predictions['instances'].pred_boxes[i].tensor.cpu().numpy()[0][3]]
    list2=np.append(list1,predictions['instances'].scores[i].cpu().numpy())
    list3=np.append(list2,predictions['instances'].pred_classes[i].cpu().numpy())
    print(predictions['instances'].scores[i].cpu().numpy())
    if list3[4]>=0.5:
        tensor_list.append(list3)
print(torch.Tensor(tensor_list))

metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

image = img[:, :, ::-1]
visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
instances = predictions["instances"].to("cpu")
visualized_output = visualizer.draw_instance_predictions(predictions=instances)
visualized_output.save('output_new') ## put output file location here
#WINDOW_NAME = "mask2former demo"
#cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
# if cv2.waitKey(0) == 27:
#     break

# modify code based on /projectnb/cs585bp/hsharma/MaskDINO/demo/demo.py
