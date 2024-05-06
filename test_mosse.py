"""
Created by: Gustav Häger
Updated by: Johan Edstedt (2021)
Updated MOSSE Tracker: Bao-Long (2024)
"""


import argparse
import cv2

# import numpy as np
from tqdm import tqdm
from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker, MoSSETracker, MoSSETrackerDeepFeature, MoSSETrackerManual, MoSSETrackerColor



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Args for the tracker')
    parser.add_argument('--sequences',nargs="+",default=[4],type=int)   #3,4,5 # takes more than 1 sequences
    parser.add_argument('--dataset_path',type=str,default="./otb_mini")
    parser.add_argument('--show_tracking',action='store_true')
    parser.add_argument('--tracker', choices=['grey', 'RGB', 'Deep', 'HOG', 'color'], help='Select a tracker with different features')
    args = parser.parse_args()

    dataset_path,SHOW_TRACKING,sequences = args.dataset_path,args.show_tracking,args.sequences

    dataset = OnlineTrackingBenchmark(dataset_path)
    results = []
    for sequence_idx in tqdm(sequences):
        a_seq = dataset[sequence_idx]

        if SHOW_TRACKING:
            cv2.namedWindow("tracker")
        
        if args.tracker == 'grey':
            tracker = NCCTracker()
        elif args.tracker == 'RGB':
            tracker = MoSSETracker()
        elif args.tracker == 'Deep':
            tracker = MoSSETrackerDeepFeature()
        elif args.tracker == 'HOG':
            tracker = MoSSETrackerManual()
        else:
            tracker = MoSSETrackerColor()
            pass
        
        pred_bbs = []
        for frame_idx, frame in tqdm(enumerate(a_seq), leave=False):
            image_color = frame['image']
            image = image_color                       # R,G,B (multi channel)                
            # image = np.sum(image_color, 2) / 3      # sum over axis 2 (RGB) = (R+G+B)/3 (single channel)
            # image = image[:,:,None]
            if frame_idx == 0:
                bbox = frame['bounding_box'] # GND
                if bbox.width % 2 == 0:
                    bbox.width += 1

                if bbox.height % 2 == 0:
                    bbox.height += 1

                current_position = bbox             # Position information: xpos, ypos, xcen, ycen, width, height,...
                tracker.start(image, bbox)
                frame['bounding_box']
            else:
                tracker.detect(image)
                tracker.update()
            pred_bbs.append(tracker.get_region())

            if SHOW_TRACKING:
            # if  frame_idx == 0 or frame_idx==10 or frame_idx==50 or frame_idx==100 or frame_idx==150 or frame_idx==200:
                bbox = tracker.get_region()
                pt0 = (bbox.xpos, bbox.ypos)
                pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
                image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
                image_color = cv2.putText(image_color, 'frame '+str(frame_idx), (5,30), 
                                          fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                          color=(255,255,255), thickness=3)
                cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
                # cv2.imwrite('./result_vis/data_%d_frame_%d.png' %(sequence_idx, frame_idx), image_color)
                cv2.imshow("tracker", image_color)
                cv2.waitKey(0)
        sequence_ious = dataset.calculate_per_frame_iou(sequence_idx, pred_bbs)
        results.append(sequence_ious)
    overlap_thresholds, success_rate = dataset.success_rate(results)
    auc = dataset.auc(success_rate)
    print(f'Tracker AUC: {auc}')
