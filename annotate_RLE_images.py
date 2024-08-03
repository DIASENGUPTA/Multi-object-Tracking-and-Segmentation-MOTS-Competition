import os
import json
import cv2
import pycocotools.mask as mask
import random
import argparse

def annotate_image(input_file, output_location, image_dir, ground_truth):
    with open(input_file) as f:
        coco_res = json.load(f)

    frame = coco_res[random.randint(0, len(coco_res) - 1)]
    

    image_path = os.path.join(image_dir, frame['videoName'], frame['name'])
    image = cv2.imread(image_path)

    for obj in frame['labels']:
        category = obj['category']
        maskedArr = mask.decode(obj['rle'])
        contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        cv2.putText(image, category, (contours[0][0][0][0], contours[0][0][0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if ground_truth is not None:
        with open(ground_truth+"/"+frame['videoName']+".json") as f:
            ground_file = json.loads(f.read())
        # print(ground_file)
        ground_frame = [ground_frame for ground_frame in ground_file['frames'] if ground_frame['name'] == frame['name']]
        ground_frame = ground_frame[0]
        for obj in ground_frame['labels']:
            category = obj['category']
            maskedArr = mask.decode(obj['rle'])
            contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (255, 0, 255), 2)
            cv2.putText(image, "True:"+category, (contours[0][0][0][0], contours[0][0][0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)



    output_file = os.path.join(output_location, frame['name'])
    cv2.imwrite(output_file, image)
    print(f"Annotated image saved at: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate image with objects from input file")
    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument("--output_location", type=str, default="./",
                        help="Path to the output location (default: current directory)")
    parser.add_argument("--image_dir", type=str, default="/projectnb/cs585bp/hsharma/bdd100k_converted/bdd/images/seg_track_20/val",
                        help="Path to the image directory (default: /projectnb/cs585bp/hsharma/bdd100k_converted/bdd/images/seg_track_20/val)")
    parser.add_argument("--ground_truth", type=str, default=None, 
                        help="Path to Ground truth if it has to be overlayed on top of the image")
    args = parser.parse_args()

    input_file = args.input_file
    output_location = args.output_location
    image_dir = args.image_dir
    ground_truth = args.ground_truth

    annotate_image(input_file, output_location, image_dir, ground_truth)

# Generate with ground truth python /projectnb/cs585bp/hsharma/annotate_RLE_images.py --input_file /projectnb/cs585bp/hsharma/pcan/output_MOTS_val_maskdino.json --ground_truth /projectnb/cs585bp/hsharma/bdd100k_converted/bdd/labels/seg_track_20/rles/val
# Generate without ground truth python /projectnb/cs585bp/hsharma/annotate_RLE_images.py --input_file /projectnb/cs585bp/hsharma/pcan/output_MOTS_val_maskdino.json