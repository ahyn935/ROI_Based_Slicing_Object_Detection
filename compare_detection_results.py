import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
from matplotlib import pyplot as plt
import glob
import pandas as pd
import time
import argparse

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def load_detection(file_path):
    """JSON 파일에서 감지 결과 로드"""
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return [], None
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        predictions = []
        processing_time = None
        
        # JSON 구조에 따라 processing_time 추출 방식 수정
        if isinstance(data, list):
            # 첫 번째 항목이 processing_time을 포함하는지 확인
            if data and isinstance(data[0], dict) and 'processing_time' in data[0]:
                processing_time = float(data[0]['processing_time'])
            
            # 나머지 항목들이 detection 결과인지 확인
            for item in data:
                if isinstance(item, dict) and 'bbox' in item:
                    predictions.append(item)
        
        print(f"[DEBUG] Loaded {len(predictions)} predictions, processing_time: {processing_time} from {file_path}")
        return predictions, processing_time
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return [], None

def compare_detections(roi_preds, plain_preds, iou_threshold=0.3):
    """두 감지 결과 비교"""
    matched_roi = set()
    matched_plain = set()

    # ROI 기반 감지에 대해 매칭
    for i, roi in enumerate(roi_preds):
        best_iou = 0
        best_j = -1
        
        for j, plain in enumerate(plain_preds):
            if j in matched_plain:  # 이미 매칭된 SAHI 감지는 건너뛰기
                continue
            
            iou = calculate_iou(roi['bbox'], plain['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        
        # IoU 임계값 이상인 경우에만 매칭
        if best_iou >= iou_threshold and best_j != -1:
            matched_roi.add(i)
            matched_plain.add(best_j)

    # 매칭되지 않은 SAHI 감지 추가
    for j in range(len(plain_preds)):
        if j not in matched_plain:
            matched_plain.add(j)

    inter = [roi_preds[i] for i in matched_roi]
    roi_only = [roi_preds[i] for i in range(len(roi_preds)) if i not in matched_roi]
    plain_only = [plain_preds[j] for j in range(len(plain_preds)) if j not in matched_plain]

    return inter, roi_only, plain_only

def draw_boxes(image_path, inter, roi_only, plain_only, output_path):
    """감지 결과 시각화"""
    if not os.path.exists(image_path):
        print(f"[WARNING] Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return

    # 교집합 (흰색)
    for obj in inter:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,255), 2)
        cv2.putText(image, f"{obj['category']}: {obj['score']:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # ROI 전용 (녹색)
    for obj in roi_only:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, f"{obj['category']}: {obj['score']:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # SAHI 전용 (파란색)
    for obj in plain_only:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(image, f"{obj['category']}: {obj['score']:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    # 범례 추가
    legend_y = 30
    cv2.putText(image, "Green: ROI-based SAHI", (10, legend_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(image, "Blue: SAHI", (10, legend_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    cv2.imwrite(output_path, image)
    print(f"[✔] Saved comparison result to: {output_path}")

def process_frame(frame_id, roi_dir, plain_dir, image_dir, output_dir):
    """단일 프레임 처리"""
    # 파일 경로 설정
    roi_path = os.path.join(roi_dir, f"frame_{frame_id}_predictions.json")
    plain_path = os.path.join(plain_dir, f"frame_{frame_id}_predictions.json")
    image_path = os.path.join(image_dir, f"frame_{frame_id}.jpg")
    output_path = os.path.join(output_dir, f"compare_{frame_id}.jpg")
    
    # 감지 결과 로드
    roi_preds, roi_time = load_detection(roi_path)
    plain_preds, plain_time = load_detection(plain_path)

    if not roi_preds and not plain_preds:
        print(f"[WARNING] No detection results found for frame {frame_id}")
        return

    # 감지 결과 비교
    inter, roi_only, plain_only = compare_detections(roi_preds, plain_preds)

    print(f"\n[Frame {frame_id} Result]")
    print(f"교집합 (IoU ≥ 0.5): {len(inter)}")
    print(f"ROI-based SAHI 감지: {len(roi_only)}")
    print(f"SAHI 감지: {len(plain_only)}")
    if roi_time is not None and plain_time is not None:
        print(f"처리 시간 - ROI: {roi_time:.3f}초, SAHI: {plain_time:.3f}초")

    # 결과 시각화
    if os.path.exists(image_path):
        draw_boxes(image_path, inter, roi_only, plain_only, output_path)
    else:
        print(f"[WARNING] Image not found: {image_path}")
        print(f"Visualization skipped for frame {frame_id}")

def process_all_frames(roi_dir, plain_dir, image_dir, output_dir, start_frame=0, end_frame=None):
    """모든 프레임 처리"""
    os.makedirs(output_dir, exist_ok=True)
    
    # ROI 디렉토리에서 모든 JSON 파일 찾기
    roi_files = glob.glob(os.path.join(roi_dir, "frame_*_predictions.json"))
    
    # 프레임 번호 추출
    frame_ids = []
    for file in roi_files:
        try:
            frame_id = int(file.split("frame_")[1].split("_predictions")[0])
            frame_ids.append(frame_id)
        except:
            continue
    
    # 프레임 범위 필터링
    if end_frame is None:
        end_frame = max(frame_ids) if frame_ids else 0
    
    frame_ids = [f for f in frame_ids if start_frame <= f <= end_frame]
    frame_ids.sort()
    
    print(f"Processing {len(frame_ids)} frames from {start_frame} to {end_frame}")
    
    # 각 프레임 처리
    for frame_id in frame_ids:
        process_frame(frame_id, roi_dir, plain_dir, image_dir, output_dir)
    
    print(f"Completed processing {len(frame_ids)} frames")

def is_in_road_area(bbox, frame_id):
    """도로 영역 내에 bbox가 있는지 확인하는 함수"""
    try:
        mask_path = os.path.join(args.roi_mask_dir, f"frame_{frame_id}_road_mask.png")
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            return False
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {mask_path}")
            return False
            
        # bbox 영역 내의 마스크 값 확인
        x1, y1, x2, y2 = map(int, bbox)
        
        # bbox 좌표가 이미지 범위를 벗어나지 않도록 조정
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[1], x2)
        y2 = min(mask.shape[0], y2)
        
        if x1 >= x2 or y1 >= y2:
            return False
            
        roi_mask = mask[y1:y2, x1:x2]
        
        if roi_mask.size == 0:
            return False
            
        # bbox 영역의 30% 이상이 도로 영역에 있으면 도로 영역으로 판단
        road_pixels = np.sum(roi_mask > 0)
        total_pixels = roi_mask.size
        road_ratio = road_pixels / total_pixels
        
        return road_ratio > 0.3
        
    except Exception as e:
        print(f"Error checking road area for frame {frame_id}: {e}")
        return False

def calculate_performance_metrics(roi_dir, plain_dir, output_dir, start_frame=0, end_frame=None):
    """Calculate and visualize key performance metrics for ROI-based SAHI vs SAHI comparison"""
    roi_files = glob.glob(os.path.join(roi_dir, "frame_*_predictions.json"))
    frame_ids = []
    for file in roi_files:
        try:
            frame_id = int(file.split("frame_")[1].split("_predictions")[0])
            frame_ids.append(frame_id)
        except:
            continue
    if end_frame is None:
        end_frame = max(frame_ids) if frame_ids else 0
    frame_ids = [f for f in frame_ids if start_frame <= f <= end_frame]
    frame_ids.sort()
    
    total_frames = 0
    roi_processing_times = []
    plain_processing_times = []
    roi_in_road = 0
    roi_outside_road = 0
    plain_in_road = 0
    plain_outside_road = 0
    
    # 신뢰도 점수 저장
    roi_scores_road = []
    plain_scores_road = []
    
    # 프레임별 통계
    roi_road_counts = []
    plain_road_counts = []
    frame_numbers = []
    
    # 메모리 사용량 추적
    roi_memory = []
    plain_memory = []

    for frame_id in frame_ids:
        roi_path = os.path.join(roi_dir, f"frame_{frame_id}_predictions.json")
        plain_path = os.path.join(plain_dir, f"frame_{frame_id}_predictions.json")
        roi_preds, roi_time = load_detection(roi_path)
        plain_preds, plain_time = load_detection(plain_path)
        
        # 중복 제거
        roi_preds = remove_duplicates([item for item in roi_preds if 'bbox' in item])
        plain_preds = remove_duplicates([item for item in plain_preds if 'bbox' in item])
        
        if not roi_preds and not plain_preds:
            continue
        
        total_frames += 1
        
        # 처리 시간 저장
        if roi_time is not None:
            roi_processing_times.append(roi_time)
        if plain_time is not None:
            plain_processing_times.append(plain_time)
        
        # 현재 프레임의 도로 영역 내 감지 수
        roi_road_in_frame = 0
        plain_road_in_frame = 0
            
        # ROI-based SAHI 결과 분석
        for pred in roi_preds:
            if is_in_road_area(pred['bbox'], frame_id):
                roi_in_road += 1
                roi_road_in_frame += 1
                roi_scores_road.append(pred['score'])
            else:
                roi_outside_road += 1
        
        # 일반 SAHI 결과 분석
        for pred in plain_preds:
            if is_in_road_area(pred['bbox'], frame_id):
                plain_in_road += 1
                plain_road_in_frame += 1
                plain_scores_road.append(pred['score'])
            else:
                plain_outside_road += 1
        
        # 프레임별 통계 저장
        roi_road_counts.append(roi_road_in_frame)
        plain_road_counts.append(plain_road_in_frame)
        frame_numbers.append(frame_id)
        
        # 메모리 사용량 추정 (검출 객체 수 기반)
        roi_memory.append(len(roi_preds))
        plain_memory.append(len(plain_preds))

    roi_total = roi_in_road + roi_outside_road
    plain_total = plain_in_road + plain_outside_road
    
    # 효율성 지표 계산
    memory_saving = (sum(plain_memory) - sum(roi_memory)) / sum(plain_memory) * 100
    processing_efficiency = ((np.mean(plain_processing_times) - np.mean(roi_processing_times)) / np.mean(plain_processing_times)) * 100
    roi_precision = np.mean(roi_scores_road) if roi_scores_road else 0
    plain_precision = np.mean(plain_scores_road) if plain_scores_road else 0
    precision_improvement = ((roi_precision - plain_precision) / plain_precision) * 100 if plain_precision > 0 else 0
    
    # 결과 출력
    print("\n[ROI-based SAHI Improvements Analysis]")
    print(f"Total frames analyzed: {total_frames}")
    
    print(f"\n1. Resource Efficiency:")
    print(f"- Memory usage reduction: {memory_saving:.1f}%")
    print(f"- Processing time improvement: {processing_efficiency:.1f}%")
    
    print(f"\n2. Detection Quality:")
    print(f"- Road area detection ratio: {roi_in_road/roi_total*100:.1f}% vs {plain_in_road/plain_total*100:.1f}%")
    print(f"- Average confidence in road area: {roi_precision:.3f} vs {plain_precision:.3f}")
    if precision_improvement > 0:
        print(f"- Confidence improvement: +{precision_improvement:.1f}%")
    
    print(f"\n3. Detection Statistics:")
    print(f"ROI-based SAHI: {roi_total:,} total, {roi_in_road:,} in road area")
    print(f"Standard SAHI: {plain_total:,} total, {plain_in_road:,} in road area")

    # 전체 프로세싱 타임 계산 (NameError 방지)
    total_roi_time = sum(roi_processing_times) if roi_processing_times else 0
    total_plain_time = sum(plain_processing_times) if plain_processing_times else 0
    avg_roi_time = np.mean(roi_processing_times) if roi_processing_times else 0
    avg_plain_time = np.mean(plain_processing_times) if plain_processing_times else 0

    # 시각화
    plt.style.use('seaborn-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    COLORS = {
        'roi': '#2ecc71',
        'sahi': '#95a5a6',
        'highlight': '#e74c3c',
        'blue': '#3498db',
        'orange': '#e67e22'
    }
    width = 0.5

    # (1,1) Unnecessary detections (non-road) reduction & processing time improvement
    unnecessary_reduction = (plain_outside_road - roi_outside_road) / plain_outside_road * 100 if plain_outside_road > 0 else 0
    time_improvement = ((np.mean(plain_processing_times) - np.mean(roi_processing_times)) / np.mean(plain_processing_times)) * 100 if plain_processing_times and roi_processing_times else 0
    improvements = [unnecessary_reduction, time_improvement]
    labels = ['Unnecessary\nReduction', 'Processing Time\nImprovement']
    bars = axs[0,0].bar([0,1], improvements, width, color=[COLORS['highlight'], COLORS['blue']])
    axs[0,0].set_title('Unnecessary Detections & Processing Time', fontsize=14, pad=20)
    axs[0,0].set_xticks([0,1])
    axs[0,0].set_xticklabels(labels)
    for i, v in enumerate(improvements):
        axs[0,0].text(i, v/2, f'{v:.1f}%', ha='center', va='center', color='white', fontsize=13, fontweight='bold')

    # (1,2) Road area detection ratio (precision-like)
    roi_road_ratio = roi_in_road / roi_total * 100 if roi_total > 0 else 0
    plain_road_ratio = plain_in_road / plain_total * 100 if plain_total > 0 else 0
    bars = axs[0,1].bar([0,1], [roi_road_ratio, plain_road_ratio], width, color=[COLORS['roi'], COLORS['sahi']])
    axs[0,1].set_title('Road Area Detection Ratio', fontsize=14, pad=20)
    axs[0,1].set_xticks([0,1])
    axs[0,1].set_xticklabels(['ROI-based\nSAHI', 'SAHI'])
    for i, v in enumerate([roi_road_ratio, plain_road_ratio]):
        axs[0,1].text(i, v/2, f'{v:.1f}%\n({[roi_in_road, plain_in_road][i]:,})', ha='center', va='center', color='white', fontsize=13, fontweight='bold')
    axs[0,1].annotate(f'+{roi_road_ratio-plain_road_ratio:.1f}p ↑',
                  xy=(1, plain_road_ratio + 0.7), ha='center', color=COLORS['blue'],
                  fontsize=13, fontweight='bold')
    axs[0,1].spines['top'].set_visible(False)
    axs[0,1].spines['right'].set_visible(False)

    # (2,1) Frame-wise road detection ratio
    axs[1,0].plot(frame_ids, plain_road_ratio_per_frame, color=COLORS['sahi'], label='SAHI', linewidth=2)
    axs[1,0].plot(frame_ids, roi_road_ratio_per_frame, color=COLORS['roi'], label='ROI-based SAHI', linewidth=2)
    axs[1,0].set_title('Road Detection Ratio per Frame', fontsize=14, pad=20)
    axs[1,0].set_xlabel('Frame')
    axs[1,0].set_ylabel('Road Detection Ratio (%)')
    axs[1,0].legend()
    axs[1,0].spines['top'].set_visible(False)
    axs[1,0].spines['right'].set_visible(False)

    # (2,2) Unnecessary reduction vs. road detection loss
    road_loss = (plain_in_road - roi_in_road)
    bars = axs[1,1].bar([0,1], [plain_out_road - roi_out_road, road_loss], width, color=[COLORS['highlight'], COLORS['orange']])
    axs[1,1].set_title('Unnecessary Reduced vs. Road Lost', fontsize=14, pad=20)
    axs[1,1].set_xticks([0,1])
    axs[1,1].set_xticklabels(['Unnecessary\nReduced', 'Road\nLost'])
    for i, v in enumerate([plain_out_road - roi_out_road, road_loss]):
        axs[1,1].text(i, v/2, f'{v:,}', ha='center', va='center', color='white', fontsize=13, fontweight='bold')
    axs[1,1].spines['top'].set_visible(False)
    axs[1,1].spines['right'].set_visible(False)
    axs[1,1].text(0.5, max(plain_out_road - roi_out_road, road_loss) + 100, f'Unnecessary {plain_out_road - roi_out_road:,}↓, Road {road_loss:,}↓',
                  ha='center', va='bottom', color=COLORS['highlight'], fontsize=13, fontweight='bold')

    # 전체 프로세싱 타임 정보 하단에 표기 (NameError 방지)
    fig.text(0.5, 0.01, f'Total Processing Time (ROI-based SAHI): {total_roi_time:.2f}s (avg: {avg_roi_time:.3f}s/frame)   |   '
                            f'Total Processing Time (SAHI): {total_plain_time:.2f}s (avg: {avg_plain_time:.3f}s/frame)',
             ha='center', va='bottom', fontsize=13, color='#222', fontweight='bold')

    # 간단한 제목 및 안내
    fig.suptitle('ROI-based SAHI vs. SAHI Comparison', fontsize=16, y=0.97, color=COLORS['roi'], fontweight='bold')
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "roi_performance_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Dashboard saved to {output_dir}/roi_performance_dashboard.png")

def is_almost_same(box1, box2, pixel_thr=5):
    return all(abs(a-b) <= pixel_thr for a, b in zip(box1, box2))

def remove_duplicates(preds, iou_thr=0.3, pixel_thr=5):
    unique = []
    used = [False] * len(preds)
    for i, a in enumerate(preds):
        if used[i]:
            continue
        unique.append(a)
        for j in range(i+1, len(preds)):
            if used[j]:
                continue
            iou = calculate_iou(a['bbox'], preds[j]['bbox'])
            if iou >= iou_thr or is_almost_same(a['bbox'], preds[j]['bbox'], pixel_thr):
                used[j] = True
    return unique

def count_total_detections(directory, remove_dupes=False):
    total = 0
    json_files = glob.glob(os.path.join(directory, "frame_*_predictions.json"))
    for file in json_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                preds = [item for item in data if isinstance(item, dict) and 'bbox' in item]
                if remove_dupes:
                    preds = remove_duplicates(preds)
                total += len(preds)
        except Exception as e:
            print(f"[ERROR] {file}: {e}")
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_dir', type=str, default='outputs/selective_video')
    parser.add_argument('--plain_dir', type=str, default='outputs/full_sahi')
    parser.add_argument('--image_dir', type=str, default='sample_data/images')
    parser.add_argument('--output_dir', type=str, default='outputs/compare_results')
    parser.add_argument('--roi_mask_dir', type=str, default='sample_data/sahi_tiling/roi_masks2')
    args = parser.parse_args()

    # 변수 초기화
    roi_total = count_total_detections(args.roi_dir, remove_dupes=True)
    plain_total = count_total_detections(args.plain_dir, remove_dupes=True)
    roi_processing_times = []
    plain_processing_times = []
    json_files = glob.glob(os.path.join(args.roi_dir, "frame_*_predictions.json"))
    json_files.sort(key=lambda x: int(x.split("frame_")[1].split("_predictions")[0]))

    # 도로/비도로 감지 수, 프레임별 도로 감지 비율
    roi_in_road = 0
    plain_in_road = 0
    roi_out_road = 0
    plain_out_road = 0
    roi_road_ratio_per_frame = []
    plain_road_ratio_per_frame = []
    frame_ids = []

    for file in json_files:
        frame_id = int(file.split("frame_")[1].split("_predictions")[0])
        roi_path = os.path.join(args.roi_dir, f"frame_{frame_id}_predictions.json")
        plain_path = os.path.join(args.plain_dir, f"frame_{frame_id}_predictions.json")
        _, roi_time = load_detection(roi_path)
        _, plain_time = load_detection(plain_path)
        if roi_time is not None:
            roi_processing_times.append(roi_time)
        if plain_time is not None:
            plain_processing_times.append(plain_time)
        roi_preds, _ = load_detection(roi_path)
        plain_preds, _ = load_detection(plain_path)
        roi_preds = remove_duplicates([item for item in roi_preds if 'bbox' in item])
        plain_preds = remove_duplicates([item for item in plain_preds if 'bbox' in item])
        roi_road, roi_nonroad = 0, 0
        plain_road, plain_nonroad = 0, 0
        for pred in roi_preds:
            if is_in_road_area(pred['bbox'], frame_id):
                roi_in_road += 1
                roi_road += 1
            else:
                roi_out_road += 1
                roi_nonroad += 1
        for pred in plain_preds:
            if is_in_road_area(pred['bbox'], frame_id):
                plain_in_road += 1
                plain_road += 1
            else:
                plain_out_road += 1
                plain_nonroad += 1
        # 프레임별 도로 감지 비율
        roi_ratio = roi_road / (roi_road + roi_nonroad) * 100 if (roi_road + roi_nonroad) > 0 else 0
        plain_ratio = plain_road / (plain_road + plain_nonroad) * 100 if (plain_road + plain_nonroad) > 0 else 0
        roi_road_ratio_per_frame.append(roi_ratio)
        plain_road_ratio_per_frame.append(plain_ratio)
        frame_ids.append(frame_id)

    # 불필요 감지(비도로) 감소율
    unnecessary_reduction = (plain_out_road - roi_out_road) / plain_out_road * 100 if plain_out_road > 0 else 0
    # 도로 감지 비율
    roi_road_ratio = roi_in_road / roi_total * 100 if roi_total > 0 else 0
    plain_road_ratio = plain_in_road / plain_total * 100 if plain_total > 0 else 0
    # 도로 감지 손실률
    road_loss = (plain_in_road - roi_in_road)
    # 전체 감지 감소율
    total_reduction = (plain_total - roi_total) / plain_total * 100 if plain_total > 0 else 0
    # 처리 시간 개선율
    time_improvement = ((np.mean(plain_processing_times) - np.mean(roi_processing_times)) / np.mean(plain_processing_times)) * 100 if plain_processing_times and roi_processing_times else 0

    # 전체 프로세싱 타임 계산 (NameError 방지)
    total_roi_time = sum(roi_processing_times) if roi_processing_times else 0
    total_plain_time = sum(plain_processing_times) if plain_processing_times else 0
    avg_roi_time = np.mean(roi_processing_times) if roi_processing_times else 0
    avg_plain_time = np.mean(plain_processing_times) if plain_processing_times else 0

    # 시각화
    plt.style.use('seaborn-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    COLORS = {
        'roi': '#2ecc71',
        'sahi': '#95a5a6',
        'highlight': '#e74c3c',
        'blue': '#3498db',
        'orange': '#e67e22'
    }
    width = 0.5

    # (1,1) Unnecessary detections (non-road) reduction & processing time improvement
    improvements = [unnecessary_reduction, time_improvement]
    labels = ['Unnecessary\nReduction', 'Processing Time\nImprovement']
    bars = axs[0,0].bar([0,1], improvements, width, color=[COLORS['highlight'], COLORS['blue']])
    axs[0,0].set_title('Unnecessary Detections & Processing Time', fontsize=14, pad=20)
    axs[0,0].set_xticks([0,1])
    axs[0,0].set_xticklabels(labels)
    for i, v in enumerate(improvements):
        axs[0,0].text(i, v/2, f'{v:.1f}%', ha='center', va='center', color='white', fontsize=13, fontweight='bold')

    # (1,2) Road area detection ratio (precision-like)
    bars = axs[0,1].bar([0,1], [roi_road_ratio, plain_road_ratio], width, color=[COLORS['roi'], COLORS['sahi']])
    axs[0,1].set_title('Road Area Detection Ratio', fontsize=14, pad=20)
    axs[0,1].set_xticks([0,1])
    axs[0,1].set_xticklabels(['ROI-based\nSAHI', 'SAHI'])
    for i, v in enumerate([roi_road_ratio, plain_road_ratio]):
        axs[0,1].text(i, v/2, f'{v:.1f}%\n({[roi_in_road, plain_in_road][i]:,})', ha='center', va='center', color='white', fontsize=13, fontweight='bold')
    axs[0,1].annotate(f'+{roi_road_ratio-plain_road_ratio:.1f}p ↑',
                  xy=(1, plain_road_ratio + 0.7), ha='center', color=COLORS['blue'],
                  fontsize=13, fontweight='bold')
    axs[0,1].spines['top'].set_visible(False)
    axs[0,1].spines['right'].set_visible(False)

    # (2,1) Frame-wise road detection ratio
    axs[1,0].plot(frame_ids, plain_road_ratio_per_frame, color=COLORS['sahi'], label='SAHI', linewidth=2)
    axs[1,0].plot(frame_ids, roi_road_ratio_per_frame, color=COLORS['roi'], label='ROI-based SAHI', linewidth=2)
    axs[1,0].set_title('Road Detection Ratio per Frame', fontsize=14, pad=20)
    axs[1,0].set_xlabel('Frame')
    axs[1,0].set_ylabel('Road Detection Ratio (%)')
    axs[1,0].legend()
    axs[1,0].spines['top'].set_visible(False)
    axs[1,0].spines['right'].set_visible(False)

    # (2,2) Unnecessary reduction vs. road detection loss
    bars = axs[1,1].bar([0,1], [plain_out_road - roi_out_road, road_loss], width, color=[COLORS['highlight'], COLORS['orange']])
    axs[1,1].set_title('Unnecessary Reduced vs. Road Lost', fontsize=14, pad=20)
    axs[1,1].set_xticks([0,1])
    axs[1,1].set_xticklabels(['Unnecessary\nReduced', 'Road\nLost'])
    for i, v in enumerate([plain_out_road - roi_out_road, road_loss]):
        axs[1,1].text(i, v/2, f'{v:,}', ha='center', va='center', color='white', fontsize=13, fontweight='bold')
    axs[1,1].spines['top'].set_visible(False)
    axs[1,1].spines['right'].set_visible(False)
    axs[1,1].text(0.5, max(plain_out_road - roi_out_road, road_loss) + 100, f'Unnecessary {plain_out_road - roi_out_road:,}↓, Road {road_loss:,}↓',
                  ha='center', va='bottom', color=COLORS['highlight'], fontsize=13, fontweight='bold')

    # 전체 프로세싱 타임 정보 하단에 표기 (NameError 방지)
    fig.text(0.5, 0.01, f'Total Processing Time (ROI-based SAHI): {total_roi_time:.2f}s (avg: {avg_roi_time:.3f}s/frame)   |   '
                            f'Total Processing Time (SAHI): {total_plain_time:.2f}s (avg: {avg_plain_time:.3f}s/frame)',
             ha='center', va='bottom', fontsize=13, color='#222', fontweight='bold')

    # 간단한 제목 및 안내
    fig.suptitle('ROI-based SAHI vs. SAHI Comparison', fontsize=16, y=0.97, color=COLORS['roi'], fontweight='bold')
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    plt.savefig(os.path.join(args.output_dir, "roi_performance_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Dashboard saved to {args.output_dir}/roi_performance_dashboard.png")


