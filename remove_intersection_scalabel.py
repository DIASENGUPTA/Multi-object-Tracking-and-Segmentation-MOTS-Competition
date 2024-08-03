import argparse
import numpy as np
import json
import pycocotools.mask as mask
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Process JSON file.')
    parser.add_argument('--input_file', type=str, help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, help='Path to the output JSON file')
    return parser.parse_args()


def main(input_file, output_file):
    with open(input_file) as f:
        coco_res = json.load(f)

    for frame in tqdm(coco_res):
        total_masked = np.zeros((720, 1280))
        for obj in frame['labels']:
            category = obj['category']
            maskedArr = mask.decode(obj['rle'])
            contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            total_masked += maskedArr

        total_mask = total_masked > 1

        for obj in frame['labels']:
            maskedArr = mask.decode(obj['rle'])
            maskedArr[total_mask] = 0  # Set maskedArr values to 0 where total_masked is True
            obj['rle'] = mask.encode(np.array(maskedArr, order='F', dtype='uint8'))
            obj['rle']['counts'] = obj['rle']['counts'].decode('utf-8')

    with open(output_file, "w") as f:
        json.dump(coco_res, f)


if __name__ == '__main__':
    args = parse_args()
    main(args.input_file, args.output_file)

# python remove_intersection_scalabel.py --input_file /projectnb/cs585bp/hsharma/pcan/output_MOTS_test_pcan.json --output_file /projectnb/cs585bp/hsharma/pcan/output_MOTS_test_pcan_cleaned.json