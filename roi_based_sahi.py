import os
import numpy as np
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import matplotlib.pyplot as plt
import glob
from sahi.prediction import ObjectPrediction, PredictionResult
import json
import argparse

# SAHI를 위한 YOLOv11 모델 경로
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='sample_data/images')
parser.add_argument('--roi_mask_dir', type=str, default='sample_data/sahi_tiling/roi_masks2')
parser.add_argument('--output_dir', type=str, default='outputs/selective_video')
parser.add_argument('--ground_truth_dir', type=str, default='sample_data/labels')
parser.add_argument('--model_path', type=str, default='models/yolo11n.pt')
args = parser.parse_args()

# 기존 하드코딩 경로를 모두 args로 변경
detection_model_path = args.model_path
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=detection_model_path,
    confidence_threshold=0.2,  # 신뢰도 임계값 더 낮춤
    device="cuda"  # 또는 "cpu"
)

# 디렉토리 설정
input_dir = args.input_dir
mask_dir = args.roi_mask_dir
output_dir = args.output_dir
ground_truth_dir = args.ground_truth_dir
os.makedirs(output_dir, exist_ok=True)

def load_road_mask(frame_num):
    """마스크 로드 및 전처리"""
    mask_path = os.path.join(mask_dir, f"frame_{frame_num}_road_mask.png")
    print(f"\n[INFO] Looking for mask file: {mask_path}")
    
    if os.path.exists(mask_path):
        print(f"[INFO] Found mask file for frame {frame_num}")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        return mask
    else:
        print(f"[WARNING] Mask not found for frame {frame_num}")
        return None

def extract_road_area(image, mask, frame_num=None, debug=False):
    image_np = np.array(image)

    if mask.shape != image_np.shape[:2]:
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((7, 7), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=3)

    # 마스크 3채널 확장
    mask_3channel = np.stack([mask_dilated] * 3, axis=-1) * 255

    # 도로 영역만 추출
    road_image = cv2.bitwise_and(image_np, mask_3channel.astype(np.uint8))

    # 디버깅 이미지 저장
    if debug and frame_num is not None:
        debug_path = f"/tmp/frame_{frame_num}_masked_debug.jpg"
        cv2.imwrite(debug_path, road_image)
        print(f"[DEBUG] Saved masked image: {debug_path}")

    return road_image, mask_dilated


    # 디버깅용: road_only_image 저장
    debug_path = os.path.join(output_dir, f"frame_{frame_num}_road_only_debug.jpg")
    print(f"[DEBUG] Saved masked road image to: {debug_path}")



from sahi.prediction import PredictionResult

def apply_sahi_slicing_with_roi_optimized_final(image_np, mask, detection_model, frame_num=None):
    height, width = image_np.shape[:2]
    slice_height, slice_width = min(512, height), min(512, width)
    stride_y = int(slice_height * 0.5)
    stride_x = int(slice_width * 0.5)

    object_predictions = []
    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(mask)
    roi_margin = 200

    slice_count = 0

    for y in range(0, height, stride_y):
        for x in range(0, width, stride_x):
            x_end, y_end = min(x + slice_width, width), min(y + slice_height, height)
            mask_crop = mask[y:y_end, x:x_end]
            image_crop = image_np[y:y_end, x:x_end]

            roi_ratio = np.sum(mask_crop) / mask_crop.size
            if roi_ratio < 0.1:  # 🔥 ROI 비율 강화
                continue

            if np.mean(image_crop) < 3 and np.std(image_crop) < 8:
                continue

            slice_count += 1
            result = get_sliced_prediction(
                image=image_crop,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                auto_slice_resolution=False,
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=0.1,
            )

            for pred in result.object_prediction_list:
                if pred.score.value < 0.25:
                    continue

                box_w = pred.bbox.maxx - pred.bbox.minx
                box_h = pred.bbox.maxy - pred.bbox.miny
                if box_w < 5 or box_h < 5 or box_w > 400 or box_h > 400:
                    continue

                pred.bbox.minx += x
                pred.bbox.maxx += x
                pred.bbox.miny += y
                pred.bbox.maxy += y

                if (pred.bbox.minx < x_roi - roi_margin or pred.bbox.maxx > x_roi + w_roi + roi_margin or
                    pred.bbox.miny < y_roi - roi_margin or pred.bbox.maxy > y_roi + h_roi + roi_margin):
                    continue

                object_predictions.append(pred)

    print(f"[INFO] Frame {frame_num if frame_num else ''} - Performed prediction on {slice_count} valid slices.")
    return PredictionResult(object_prediction_list=object_predictions, image=np.ascontiguousarray(image_np))

def filter_car_and_truck(prediction_result):
    """차량 관련 클래스만 필터링"""
    allowed_classes = ['car', 'truck']
    filtered = []
    for pred in prediction_result.object_prediction_list:
        if pred.category.name in allowed_classes:
            filtered.append(pred)
    prediction_result.object_prediction_list = filtered
    return prediction_result

def load_ground_truth(frame_num):
    """Ground Truth 라벨 파일을 로드"""
    # 파일 패턴 출력
    label_path = os.path.join(ground_truth_dir, f"frame{frame_num}_jpg.rf.*.txt")
    label_files = glob.glob(label_path)
    
    print(f"\nLooking for ground truth file: {label_path}")
    print(f"Found files: {label_files}")
    
    if not label_files:
        print(f"No ground truth file found for frame {frame_num}")
        return None
    
    boxes = []
    with open(label_files[0], 'r') as f:
        lines = f.readlines()
    
    # 실제 이미지 크기 기준으로 좌표 변환
    image_width = 1920  # 실제 이미지 너비
    image_height = 1080  # 실제 이미지 높이
    
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        # YOLO 형식의 상대 좌표를 절대 좌표로 변환 (실제 이미지 크기 기준)
        x1 = int((x_center - width/2) * image_width)
        y1 = int((y_center - height/2) * image_height)
        x2 = int((x_center + width/2) * image_width)
        y2 = int((y_center + height/2) * image_height)
        boxes.append([x1, y1, x2, y2])
    
    return boxes

def calculate_iou(box1, box2):
    """두 바운딩 박스 간의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_detection_accuracy(prediction_result, ground_truth_boxes, mask):
    """마스크 영역 내의 객체들에 대해서만 정확도 평가"""
    if not ground_truth_boxes:
        return 0, 0, 0
    
    # 디버깅: 예측 결과 출력
    print(f"\nTotal predictions before filtering: {len(prediction_result.object_prediction_list)}")
    
    true_positives = 0
    false_positives = 0
    
    # 예측된 바운딩 박스들 필터링
    pred_boxes = []
    for pred in prediction_result.object_prediction_list:
        # 예측 객체의 바운딩 박스 좌표 가져오기
        box = [int(pred.bbox.minx), int(pred.bbox.miny), int(pred.bbox.maxx), int(pred.bbox.maxy)]
        
        # 디버깅: 모든 예측 박스 출력
        print(f"Raw prediction: {box}, Size: {box[2]-box[0]}x{box[3]-box[1]}, Confidence: {pred.score.value:.3f}, Category: {pred.category.name}")
        
        # 비정상적으로 큰 박스 필터링
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        if box_width > 300 or box_height > 300:  # 크기 제한 완화
            print(f"Filtered out: Box too large ({box_width}x{box_height})")
            continue
            
        # 너무 작은 박스도 필터링
        if box_width < 5 or box_height < 5:  # 크기 제한 완화
            print(f"Filtered out: Box too small ({box_width}x{box_height})")
            continue
        
        # 마스크 영역 확인을 위한 디버깅
        box_mask = np.zeros_like(mask)
        cv2.rectangle(box_mask, (box[0], box[1]), (box[2], box[3]), 1, -1)
        intersection = np.logical_and(mask, box_mask)
        
        # 마스크 영역과의 교차 확인
        if np.any(intersection):
            print(f"Box intersects with mask: {box}")
            pred_boxes.append((box, pred.score.value))
        else:
            print(f"Box does not intersect with mask: {box}")
    
    # Ground Truth 박스들 (마스크 영역 내에 있는 것만)
    masked_gt_boxes = []
    for gt_box in ground_truth_boxes:
        gt_box = [int(x) for x in gt_box]
        box_mask = np.zeros_like(mask)
        cv2.rectangle(box_mask, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), 1, -1)
        intersection = np.logical_and(mask, box_mask)
        
        if np.any(intersection):
            print(f"GT box intersects with mask: {gt_box}")
            masked_gt_boxes.append(gt_box)
        else:
            print(f"GT box does not intersect with mask: {gt_box}")
    
    print(f"\nPredicted boxes coordinates:")
    for box, conf in pred_boxes:
        print(f"Pred: {box}, Size: {box[2]-box[0]}x{box[3]-box[1]}, Confidence: {conf:.3f}")
    
    print(f"\nGround truth boxes coordinates:")
    for box in masked_gt_boxes:
        print(f"GT: {box}, Size: {box[2]-box[0]}x{box[3]-box[1]}")
    
    # 매칭되지 않은 GT 박스 추적
    matched_gt = [False] * len(masked_gt_boxes)
    
    # 각 예측에 대해 가장 높은 IoU를 가진 GT 박스 찾기
    for pred_box, conf in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(masked_gt_boxes):
            if not matched_gt[i]:
                iou = calculate_iou(pred_box, gt_box)
                print(f"IoU between pred {pred_box} and GT {gt_box}: {iou}")
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        # IoU 임계값을 0.05로 낮춤 (좌표계 차이를 고려)
        if best_iou >= 0.05:  
            true_positives += 1
            matched_gt[best_gt_idx] = True
            print(f"Matched pred {pred_box} with GT {masked_gt_boxes[best_gt_idx]}, IoU: {best_iou}")
        else:
            false_positives += 1
            print(f"No match for pred {pred_box}, best IoU: {best_iou}")
    
    false_negatives = len(masked_gt_boxes) - sum(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMasked GT boxes: {len(masked_gt_boxes)}")
    print(f"Predicted boxes in mask: {len(pred_boxes)}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    
    return precision, recall, f1_score




def save_result(result, frame, frame_num, ground_truth_boxes=None, mask=None):
    """결과를 시각화하여 저장"""
    # 차량 및 트럭만 필터링
    filtered_result = filter_car_and_truck(result)

    # 필터링된 예측 결과 시각화
    filtered_result.export_visuals(export_dir=output_dir)

    # 출력된 이미지 파일 이름을 지정 (고유한 이름으로 수정)
    filename = f"frame_{frame_num}_prediction_visual.png"  # 고유한 파일 이름 생성
    
    print(f"[✔] Saved: {filename}")
    
    # 결과 이미지의 이름을 지정하여 저장된 결과 파일을 덮어쓰지 않도록 함
    os.rename(os.path.join(output_dir, "prediction_visual.png"),
              os.path.join(output_dir, filename))

def save_json_result(result, frame_num, save_dir, processing_time=None):
    """결과를 JSON 형식으로 저장"""
    predictions = []
    
    # 처리 시간 정보를 첫 번째 항목으로 추가
    if processing_time is not None:
        predictions.append({"processing_time": processing_time})
    
    # 객체 감지 결과 추가
    for pred in result.object_prediction_list:
        predictions.append({
            "bbox": [
                int(pred.bbox.minx),
                int(pred.bbox.miny),
                int(pred.bbox.maxx),
                int(pred.bbox.maxy)
            ],
            "category": pred.category.name,
            "score": float(pred.score.value)
        })
    
    # JSON 파일로 저장
    output_path = os.path.join(save_dir, f"frame_{frame_num}_predictions.json")
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=4)

def evaluate_performance(cap, detection_model):
    """비디오의 각 프레임에 대해 객체 감지 수행"""
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metrics_data = []
    total_processing_time = 0
    
    print(f"\n[INFO] Starting video processing - Total frames: {total_frames}")
    
    # 비디오 저장을 위한 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(
        os.path.join(output_dir, 'detection_results.mp4'),
        fourcc, 30.0, (int(cap.get(3)), int(cap.get(4)))
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"\n[INFO] Processing frame {frame_count}/{total_frames}")
        
        # BGR에서 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # PIL 이미지로 변환
        image = Image.fromarray(frame_rgb)
        
        # 마스크 로드
        mask = load_road_mask(frame_count)
        if mask is None:
            print(f"[WARNING] Skipping frame {frame_count} due to missing mask")
            frame_count += 1
            continue
        
        # 처리 시작 시간 기록
        start_time = time.time()
        
        # 도로 영역 추출
        road_image, dilated_mask = extract_road_area(image, mask, frame_count)
        
        # NumPy 배열로 변환 (SAHI 함수는 NumPy 배열을 입력으로 받음)
        road_image_np = np.array(road_image)
        
        # SAHI 슬라이싱 적용
        result = apply_sahi_slicing_with_roi_optimized_final(road_image_np, dilated_mask, detection_model, frame_count)
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        total_processing_time += processing_time
        print(f"[INFO] Frame processing time: {processing_time:.3f} seconds")
        
        # Ground Truth 로드 (있는 경우)
        ground_truth_boxes = load_ground_truth(frame_count)
        
        # 결과 저장
        filtered_result = filter_car_and_truck(result)
        save_result(filtered_result, frame, frame_count, ground_truth_boxes, dilated_mask)
        save_json_result(filtered_result, frame_count, output_dir, processing_time)
        
        # 시각화된 프레임을 비디오에 추가
        vis_frame_path = os.path.join(output_dir, f"frame_{frame_count}_prediction_visual.png")
        if os.path.exists(vis_frame_path):
            vis_frame = cv2.imread(vis_frame_path)
            if vis_frame is not None:
                # 이미 BGR 형식으로 로드되었으므로 그대로 사용
                video_out.write(vis_frame)
                print(f"[INFO] Added frame {frame_count} to video output")
            else:
                print(f"[WARNING] Could not read visualization file for frame {frame_count}, using original frame")
                # 원본 프레임은 이미 BGR 형식이므로 그대로 사용
                video_out.write(frame)
        else:
            print(f"[WARNING] Visualization file not found for frame {frame_count}, using original frame")
            # 원본 프레임은 이미 BGR 형식이므로 그대로 사용
            video_out.write(frame)
        
        frame_count += 1
    
    # 비디오 저장 종료
    video_out.release()
    
    # 성능 요약 정보 출력
    avg_time = total_processing_time / frame_count if frame_count > 0 else 0
    fps = 1 / avg_time if avg_time > 0 else 0
    
    print("\nPerformance Summary:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average processing time: {avg_time:.3f} seconds per frame")
    print(f"Average FPS: {fps:.2f}")
    
    print(f"\n[✔] Completed processing all {frame_count} frames")
    print(f"[✔] Saved detection results video to: {os.path.join(output_dir, 'detection_results.mp4')}")
    return metrics_data


# 메인 실행
if __name__ == "__main__":
    # 디렉토리 설정
    video_path = '/root/jcci2025/input_video/final/jcci_input_video_final.mp4'
    output_dir = "/root/jcci2025/evaluate/250420_outputs/selective_video/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 성능 평가 실행
    evaluate_performance(cap, detection_model)
    
    # 자원 해제
    cap.release()
