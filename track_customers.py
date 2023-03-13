import argparse
import cv2
import os
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
ROOT = FILE.parents[0] / 'yolov8_tracking'  # yolov8 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8_tracking.yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8_tracking.yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8_tracking.yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8_tracking.yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8_tracking.yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8_tracking.yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8_tracking.yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8_tracking.yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8_tracking.yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from yolov8_tracking.trackers.multi_tracker_zoo import create_tracker

from path_prediction.perspective_transformation.transform_perspektive import load_projection_matrix, load_zones, estimate_floor_position

from path_prediction.transition_net.transition_data import TransitionData
from path_prediction.transition_net.transition_net import TransitionNet

from path_prediction.utils.plots import draw_icon


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov8n.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='bytetrack', # tracking method
        tracking_config=None, # tracking config
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save annotated video
        nosave=False,  # do not save images/videos
        floor_plan_projection=None,  # camera projection on floor plan
        save_floor_positions=False,  # save floor positions
        predict_customer_path=False,  # predict customer path
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

    save_projection = floor_plan_projection is not None
    if save_projection:
        projection_project = FILE.parents[0] / "inference_data" / "projection_matrix" / floor_plan_projection
        floor_plan_im = cv2.imread(str(projection_project / "images" / "floor_plan.png"))
        P, P_inv, camera_origin = load_projection_matrix(projection_project)
        floor_redzones, image_greenzones = load_zones(projection_project / "zones")
        print(P, P_inv, camera_origin, floor_plan_im.shape)
        print(floor_redzones, image_greenzones)
        icon_man = cv2.imread(str(FILE.parents[0] / "inference_data" / "icons" / "icon_man.png"))
        icon_cart = cv2.imread(str(FILE.parents[0] / "inference_data" / "icons" / "icon_cart.png"))

    if predict_customer_path:
        transition_data_dir = FILE.parents[0] / "inference_data" / "transition_data"
        transition_data = TransitionData()
        transition_data.load_floorplan_folder(str(transition_data_dir), source_resolution=floor_plan_im.shape, cam_id=4)
        print(transition_data.raw_data.head(30))
        tn = TransitionNet(transition_data, grid_dimensions=(10, 15), state_length=1, state_scaler=1)
        print(tn)


    if not tracking_config:
        tracking_config = ROOT / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')
    imgsz *= 2 if len(imgsz) == 1 else 1

    source = str(source)
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
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

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
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        imfp = floor_plan_im.copy() # clean floor plan image for each frame
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    
                    for j, (output) in enumerate(outputs[i]):
                        
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)

                            if save_projection and c==0 :
                                floor_position = tuple(estimate_floor_position(P_inv, camera_origin, bbox, floor_redzones, image_greenzones))
                                try:
                                    draw_icon(imfp, icon_man, floor_position)
                                except:
                                    print("!!!!!!!!!!!!!!!!!!!!!!", floor_position)
                                #cv2.circle(imfp, floor_position, line_thickness+6, color, -1)

                                if save_floor_positions:
                                    with open(txt_path + '_floor_positions.txt', 'a') as f:
                                        f.write(('%g ' * 4 + '\n') % (frame_idx + 1, id, floor_position[0], floor_position[1]))

                            if save_projection and c==1 :
                                floor_position = tuple(estimate_floor_position(P_inv, camera_origin, bbox, floor_redzones, image_greenzones, avg_heigth=50))
                                draw_icon(imfp, icon_cart, floor_position)
                                #cv2.circle(imfp, floor_position, line_thickness+6, color, -1)

                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                            
            else:
                pass
                
            # Stream results
            im0 = annotator.result()
            if save_projection:
                assert im0.shape[0] == imfp.shape[0], "Floorplan does not has the same heigth as the image"
                im0 = np.concatenate((im0, imfp), axis=1)

            
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        if save_projection:
                            w += int(floor_plan_im.shape[1])
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


if __name__ == "__main__":
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(
        source='inference_data/videos/demo_clean.mp4', # file to predict
        yolo_weights=WEIGHTS / 'yolov8s_SD_EDEKA_Cheat_960.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='bytetrack',
        imgsz=(960, 960),
        conf_thres=0.81, #conf thres for detection model
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=True,  # save results to *.txt
        save_vid=True,  # save the annotated video
        floor_plan_projection="Ch4_960",  # project the annotated video onto the floor plan
        save_floor_positions=True,  # save the floor positions of the annotated video to .txt
        predict_customer_path=False,  # predict the path of the customer
        classes=[0,1],  # filter by class: --class 0, or --class 0 2 3
        project=Path.cwd() / 'inference_data' / 'runs',  # save results to project/name
    )
