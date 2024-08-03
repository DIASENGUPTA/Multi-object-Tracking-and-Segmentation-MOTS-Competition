import argparse
import pycocotools.mask as mask
import json
import cv2
from tqdm import tqdm

def polygonFromMask(maskedArr):
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        # if contour.size >= 6:
        if len(contour) > 4:
            segmentation.append(contour.flatten().tolist())
    # RLEs = mask.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    # RLE = mask.merge(RLEs)
    # area = mask.area(RLE)
    # [x, y, w, h] = cv2.boundingRect(maskedArr)
    return segmentation

def main(input_file, output_file):
    with open(input_file, "r") as f:
        coco_gt = json.load(f)

    coco_new_gt = coco_gt.copy()
    coco_new_gt['annotations']=[]

    for i in range(len(coco_gt['annotations'])):
        try:
            maskedArr = mask.decode(coco_gt['annotations'][i].get('segmentation'))
            res = polygonFromMask(maskedArr)
            if len(res)==0: ## if encoding is 0, nothing gets appended
                continue
            coco_gt['annotations'][i]['segmentation'] = res
            coco_new_gt['annotations'].append(coco_gt['annotations'][i])
        except Exception as e:
            print("Row:", i, ":", e)
            # coco_new_gt['annotations'][i]['segmentation']=[]
    print("Removed:", len(coco_gt['annotations']) - len(coco_new_gt['annotations'])," Annotations because of errors")

    for i in range(len(coco_new_gt['images'])):
        coco_new_gt['images'][i]["width"]= 1280
        coco_new_gt['images'][i]["height"]= 720

    print("Done")
    print("Writing to Disk...")
    with open(output_file, "w") as f:
        json.dump(coco_new_gt, f)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO segmentation masks to polygons")
    parser.add_argument("-i", "--input", type=str, help="Input file path")
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    main(input_file, output_file)
