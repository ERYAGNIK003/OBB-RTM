from collections import defaultdict
import cv2
import numpy as np
import torchvision
import torch
import os 
import csv
from ultralytics import YOLO
import time
from boxmot.trackers.bbox import BotSort
from boxmot.trackers import OccluBoost

from ultralytics.utils.nms import non_max_suppression
from torchvision.ops import nms
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import read_image
from pathlib import Path
import argparse
import shutil


YOLODIR=os.path.join(".","Weights_Drashti_HaOBB")
yoloes=["yolo11n-obb","yolo11s-obb","yolo11m-obb","yolo11l-obb","yolo11x-obb"]

def cxcywh2xy_top_wh(cx,cy,w,h):
    x=cx-w//2
    y=cy-h//2
    w=w
    h=h
    return x,y,w,h

def xyxy2xy_top_wh(xyxy):
    x1=xyxy[0]
    y1=xyxy[1]
    x2=xyxy[2]
    y2=xyxy[3]

    w=int(x2-x1)
    h=int(y2-y1)
    #xc=int(x1+(w/2))
    #yc=int(y1+(h/2))
    if w<0 or h<0:
        print("Error in BBox")
    return x1,y1,w,h

def nms_from_result(result0):
    #print(result0)
    boxes = result0.boxes.xyxy   # tensor of shape (N,4) in x1,y1,x2,y2
    scores = result0.boxes.conf   # tensor of shape (N,) confidence scores
    class_ids = result0.boxes.cls    # tensor of shape (N,) class indices

    # Class-agnostic NMS
    iou_threshold = 0.5
    keep = nms(boxes, scores, iou_threshold)

    # Filtered results
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    filtered_class_ids = class_ids[keep]

    dets = torch.cat([
    filtered_boxes,
    filtered_scores.unsqueeze(1),
    filtered_class_ids.unsqueeze(1)], dim=1)

    # Convert to NumPy for tracker
    dets_np = dets.cpu().numpy()
    return dets_np

   

trk_type="botsort" 
avgTime_inf=0
avgTime_trk=0
avgDets=0
FPS=0
img_sz=(1280, 720)



def getLogo(main_image):
    logo = cv2.imread('AU.png', cv2.IMREAD_UNCHANGED)  # Load the image with alpha channel
    # Determine the position to place the logo at the right corner
    # Extract the alpha channel from the logo
    alpha_channel = logo[:, :, 3]
    # Convert the logo to BGR (3 channels) for blending
    logo_bgr = logo[:, :, :3]
    # Determine the position to place the logo at the right corner
    logo_height, logo_width = logo_bgr.shape[:2]
    position_y = main_image.shape[0] - logo_height - 10  # 10 pixels margin from bottom
    position_x = main_image.shape[1] - logo_width - 10   # 10 pixels margin from right

    # Overlay the logo onto the main image using the alpha channel for transparency
    for c in range(3):
        main_image[position_y:position_y+logo_height, position_x:position_x+logo_width, c] = (
            alpha_channel / 255.0 * logo_bgr[:, :, c] +
            (1.0 - alpha_channel / 255.0) * main_image[position_y:position_y+logo_height, position_x:position_x+logo_width, c]
        )
    return main_image

def do_Inference_On_Vid(input_path,output_dir,trackerName,annFlg):
    global YOLODIR,yoloes
    if trackerName not in ["botsort","occluboost"]:
        return
    for modelFile in yoloes:
        weight=os.path.join(YOLODIR,modelFile,"weights","best.pt")
        model = YOLO(weight)
        model_out=os.path.join(output_dir,modelFile)
        os.mkdir(model_out)

        out_vid=cv2.VideoWriter(os.path.join(model_out,"annotated.mp4"),cv2.VideoWriter_fourcc(*'MP4V'),30,img_sz)

        if trackerName=="botsort":
            tracker = BotSort(reid_weights=None, device=torch.device('cpu'),half=False,with_reid=False,
                              track_high_thresh=0.25,track_low_thresh=0.1,new_track_thresh=0.25,min_hits=5,
                              nr_classes=14,cmc_method="sof",is_obb=True)#gmc_method="sparseOptFlow"
        elif trackerName=="occluboost":
            tracker = OccluBoost(reid_model=None, device=torch.device("cpu"),half=False,with_reid=False,det_thresh=0.30,
                                 max_age=30,min_hits=5,iou_threshold=0.30,per_class=False,use_cmc=True,cmc_method="sof",
                                 lambda_iou=0.5,lambda_mhd=0.25,lambda_shape=0.25,use_dlo_boost=False,use_duo_boost=False,
                                 s_sim_corr=False,use_rich_s=False,use_sb=False,use_vt=False,adaptive_kf=True,track_high_thresh=0.25,
                                 nr_classes=14,is_obb=True,)#gmc_method="sparseOptFlow"
        else:
            return
        avgTime_inf=0
        avgTime_trk=0
        avgDets=0
        FPS=0
        unqTrk=[]
        
        # Create overall file 
        f_trk=open(os.path.join(model_out,f"{modelFile}_BotSort_withoutReID.csv"),"w")
        logger_trk = csv.writer(f_trk)
        #logger_trk.writerow([])#frm,id,xtop,top,w,h,conf,x,y,z

        # Create DET overall file 
        f_det=open(os.path.join(model_out,f"{modelFile}_Frmwise_Dets.csv"),"w")
        logger_det = csv.writer(f_det)
        #logger_det.writerow(["frm","clsid","x1","y1","x2","y2","x3","y3","x4","y4","conf"])
        logger_det.writerow(["frm","clsid","cx","cy","w","h","angle_rad","conf"])
                                           
        # Open the video file
        cap = cv2.VideoCapture(input_path)

        # Store the track history
        track_history = defaultdict(lambda: [])
        frm=0
        frm1000_flg=False
        frmCopy=None
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                frm=frm+1
                frm1000_flg=False
                if frm%500==0:
                    print("Working on frame ",frm)
                    frm1000_flg=True
                    frmCopy=frame.copy()
                    
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                start = time.time()
                results=model.predict(frame,agnostic_nms=True,device=0,conf=0.1, iou=0.5,verbose=False) 
                end = time.time()
                t = end - start

                names=None
                
                num_dets=0
                for result in results:
                    scores = result.obb.conf.cpu().numpy()
                    classes = result.obb.cls.cpu().numpy()
                    #boxes = result.obb.xyxyxyxy.cpu().numpy()
                    xywhr = result.obb.xywhr.cpu().numpy()  # center-x, center-y, width, height, angle (radians)
                    names=results[0].names
                    detections = []
                    
                    for score, cls, (cx, cy, w, h, angle) in zip(scores, classes, xywhr):
                        num_dets+=1
                        #x1, y1, x2, y2, x3, y3, x4, y4 = bbox.reshape(-1).astype(int)
                        #bbox = bbox.astype(int)
                        # OBB corners: (4, 2)
                        ##x = bbox[:, 0]
                        #y = bbox[:, 1]

                        # Convert OBB → HBB
                        #x1 = x.min()
                        #y1 = y.min()
                        #x2 = x.max()
                        #y2 = y.max()

                        detections.append([float(cx),
                                           float(cy),
                                           float(w),
                                           float(h),
                                           float(angle),
                                           float(score),
                                           int(cls)])
                        #pts = bbox.reshape((-1, 1, 2))

                        #cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                        
                        #logger_det.writerow([frm,cls, x1, y1, x2, y2, x3, y3, x4, y4, score])#["frm","clsid","x1","y1","x2","y2","x3","y3","x4","y4","conf"])
                        logger_det.writerow([frm,int(cls), float(cx), float(cy), float(w), float(h), float(angle),
                                             float(score)])#["frm","clsid","cx","cy","w","h","angle_rad","conf"]
                detections = np.asarray(detections, dtype=np.float32)
                #
                t_inf=t
                avgTime_inf += (t - avgTime_inf) / frm
                ##########
                
                start = time.time()
                ts = tracker.update(detections,frame)
                end = time.time()
                t = end - start
                
                t_trk=t
                avgTime_trk += (t - avgTime_trk) / frm
                
                '''
                xyxys = ts[:,0:4].astype('int') # float64 to int
                ids = ts[:, 4].astype('int') # float64 to int 
                confs = ts[:, 5]
                clss = ts[:, 6]
                '''
                #OBB  columns (9): cx, cy, w, h, angle, id, conf, cls, det_ind
                for track in ts:
                    # -----------------------------
                    # Read BoxMOT OBB output
                    # -----------------------------
                    cx, cy, w, h, angle = track[:5]

                    track_id = int(track[5])
                    conf = float(track[6])
                    cls = int(track[7])

                    #x,y,w,h=cxcywh2xy_top_wh(cx,cy,w,h)
                    #classname=names[int(cls)]
                    
                    if track_id not in unqTrk:
                        unqTrk.append(track_id)
                            
                    # -----------------------------
                    # Convert angle to degrees
                    # -----------------------------
                    angle_deg = np.degrees(angle)

                    # -----------------------------
                    # Create rotated rectangle
                    # -----------------------------
                    rect = (
                        (float(cx), float(cy)),
                        (float(w), float(h)),
                        float(angle_deg)
                    )

                    # Get 4 corner points
                    corners = cv2.boxPoints(rect)
                    corners = np.int32(corners)

                    # Convert OBB -> enclosing AABB
                    x_min = corners[:, 0].min()
                    y_min = corners[:, 1].min()
                    x_max = corners[:, 0].max()
                    y_max = corners[:, 1].max()

                    # MOT: left, top, width, height
                    left = float(x_min)
                    top = float(y_min)
                    mot_w = float(x_max - x_min)
                    mot_h = float(y_max - y_min)
                    #logger_trk.writerow([])#frm,id,xtop,top,w,h,conf,x,y,z
                    logger_trk.writerow([frm,track_id,left,top,mot_w,mot_h,1,-1,-1,-1])

                    if frm1000_flg:
                        #frm1000_flg=False
                        # -----------------------------
                        # Draw MOT HBB
                        # -----------------------------
                        cv2.rectangle(frmCopy, (int(left), int(top)), (int(left + mot_w), int(top + mot_h)), (0, 255, 0), 2)
                    # -----------------------------
                    # Draw OBB
                    # -----------------------------
                    cv2.polylines(
                        frame,
                        [corners],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2
                    )

                    # -----------------------------
                    # Label
                    # -----------------------------
                    label = f"ID:{track_id} C:{cls} {conf:.2f}"

                    # Position label near top-left corner
                    x, y = corners[0]

                    cv2.putText(
                        frame,
                        label,
                        (int(x), int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
        
                avgDets+=(num_dets-avgDets)/frm

                FPS=round(1/(t_inf+t_trk),2)
                                
                #write timings
                cv2.putText(frame, "No. Dets: "+str(num_dets), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
                cv2.putText(frame, "Inference: "+str(round(t_inf*1000,2)), (1000,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
                cv2.putText(frame, "Track: "+str(round(t_trk*1000,2)), (2000,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
                cv2.putText(frame, "FPS: "+str(FPS), (3000,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)

                ann_frm=getLogo(frame)
                resized = cv2.resize(ann_frm, img_sz, interpolation=cv2.INTER_AREA)
                #cv2.imwrite(os.path.join(out_img_dir,str(frm)+".jpg"),resized)
                out_vid.write(resized)
                if frm1000_flg:
                    cv2.imwrite(os.path.join(model_out,str(frm)+".jpg"),frmCopy)
            else:
                # Break the loop if the end of the video is reached
                break
                
        cap.release()
        out_vid.release()
        f_trk.close()
        f_det.close()

        print(f"{modelFile} got {len(unqTrk)} tracks")
        
        del track_history
        del tracker                  
        del model
        
    return



def do_Inference_On_Imgs(input_path,output_dir,annFlg):
    global YOLODIR,yoloes
    #iterate over each YOLO model and run inference
    frame=cv2.imread(input_path)
    for modelFile in yoloes:
        weight=os.path.join(YOLODIR,modelFile,"weights","best.pt")
        model = YOLO(weight)
        model_out=os.path.join(output_dir,modelFile)
        os.mkdir(model_out)

        frm=frame.copy()
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        start = time.time()
        results = model.predict(frm,agnostic_nms=True,device=0,conf=0.25, iou=0.5,verbose=False) 
        end = time.time()
        t = end - start

        num_dets=0        
        for result in results:
            scores = result.obb.conf.cpu().numpy()
            classes = result.obb.cls.cpu().numpy()
            boxes = result.obb.xyxyxyxy.cpu().numpy()
    
            for score, cls, bbox in zip(scores, classes, boxes):
                num_dets+=1
                bbox = bbox.astype(int)

                # bbox = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

                pts = bbox.reshape((-1, 1, 2))

                cv2.polylines(
                    frm,
                    [pts],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2
                )

                cv2.putText(
                    frm,
                    str(int(cls)),
                    tuple(bbox[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3
                )
                
        t_inf=round(t*1000,0) #ms
        FPS=int(round(1/t,0)) 
        cv2.putText(frm, "No. Dets: "+str(num_dets), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5)
        cv2.putText(frm, "Inference: "+str(t_inf), (1000,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5)
        #cv2.putText(ann_frm, "Track: "+str(round(t_trk*1000,2)), (2000,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        cv2.putText(frm, "FPS: "+str(FPS), (3000,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5)
        ann_frm=getLogo(frm)
        resized = cv2.resize(ann_frm, img_sz, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(model_out,"output.jpg"),resized)
        
        del frm
        del ann_frm
        del resized
        del model
    
    return

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11-OBB image inference and video tracking"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image (.jpg) or video (.mp4/.avi/.mov/.mkv)"
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Tracker name for video input, e.g. botsort, bytetrack, ocsort"
    )

    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Generate annotated output"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    input_path = Path(args.input)

    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")

    extension = input_path.suffix.lower()

    # --------------------------------------------------------
    # Determine input type
    # --------------------------------------------------------

    if extension == ".jpg":
        input_type = "image"

        if args.tracker is not None:
            parser.error(
                "--tracker should only be used with video input."
            )

    elif extension in [".mp4", ".avi", ".MP4", ".mkv"]:
        input_type = "video"

        if args.tracker is None:
            parser.error(
                "--tracker is required when the input is a video."
            )

    else:
        parser.error(
            f"Unsupported file format: {extension}. "
            "Use .jpg for images or .mp4/.avi/.mov/.mkv for videos."
        )

    return args, input_type

if __name__ == "__main__":

    args, input_type = parse_args()

    annFlg=args.annotate
    

    ######
    input_path = Path(args.input)
    output_dir = Path(f"OBB_inference_{input_path.stem}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    ########
    
    # --------------------------------------------------------
    # Your inference code goes here
    # --------------------------------------------------------

    if input_type == "image":

        print("Running YOLO11-OBB image inference...")

        # YOLO11-OBB inference here
        do_Inference_On_Imgs(input_path,output_dir,annFlg)        

    elif input_type == "video":

        print(
            f"Running YOLO11-OBB + {args.tracker} tracking..."
        )

        # YOLO11-OBB + BoxMOT tracking here
        do_Inference_On_Vid(input_path,output_dir,args.tracker,annFlg)
        
