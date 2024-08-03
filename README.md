# Multi-object-Tracking-and-Segmentation-Competition
Pcan is the current best approach for bdd100k dataset tracking algorithm. In our project, we have integrated the YOLOv8 tracking algorithm, utilizing MaskDINO as the vehicle segmentation algorithm in place of YOLOv8's segmentation capabilities. This approach allows us to use MaskDINO's advanced segmentation performance while maintaining the robust tracking capabilities of YOLOv8_tracking.

## Clean Data

BDD100k dataset uses rider and bicycle as separate classes while MaskDINO uses them as same, so make sure in the config json files that you replace them to be same classes. Merging similar classes
```
python pcan/seg_track_modified.py
```

Remove intersecting masks
```
python remove_intersection_scalable.py
```

## Annotate bdd100k labels
```
python annotate_RLE_images.py
python cocoRLE_to_coco_polygon.py
```
## Visualisation

```
python sratch_file.py
python MaskDINO/yolov8_tracking/bdd_output.py
```

### Track
```
python track_test.py
```

### Score according to MOTS tracking challenge
```
python MaskDINO/yolov8_tracking/scoring_bdd100k.py
```


Run `test_maskdino.ipynb' to test our maskdino based tracking algorithm.


