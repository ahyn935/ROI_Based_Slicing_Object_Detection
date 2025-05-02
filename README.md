# ROI_Based_Slicing_Object_Detection

**"Efficient Object Recognition Technique Using ROI-Based Slicing Inference"**

I propose a novel object detection pipeline based on the SAHI (Slicing Aided Hyper Inference) framework, which leverages ROI (Region of Interest) masks to perform efficient and accurate inference using a slicing-based approach. By focusing computation on relevant regions through ROI-guided slicing, my method significantly reduces unnecessary detections and improves both speed and accuracy, especially in resource-constrained or real-time environments. This approach extends the standard SAHI pipeline by integrating ROI masks, enabling more selective and efficient object detection. 

---

<p align="center">
  <img src="https://github.com/ahyn935/ROI_Based_Slicing_Object_Detection/blob/main/outputs/basic_sahi/frame_241_prediction_visual.png?raw=true" alt="Basic SAHI" width="45%"/>
  &nbsp;&nbsp;
  <img src="https://github.com/ahyn935/ROI_Based_Slicing_Object_Detection/blob/main/outputs/roi_based_sahi/frame_241_prediction_visual.png?raw=true" alt="ROI-Based SAHI" width="45%"/>
</p>

<p align="center">
  <b>Left: SAHI result | Right: ROI-Based SAHI result</b>
</p>

---

## Features

- ROI mask-based slicing inference for efficient object detection
- Direct comparison with standard SAHI (Slicing Aided Hyper Inference)
- Performance dashboard and statistics
- Sample data and easy-to-run scripts

## Folder Structure

```
roi-based-slicing-object-detection/
├── basic_sahi.py
├── roi_based_sahi.py
├── compare_detection_results.py
├── sample_data/
├── outputs/
├── models/
├── requirements.txt
├── run_all.sh
└── README.md
```

## How to Run

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run detection (sample frame)**
   ```bash
   python basic_sahi.py --input_dir sample_data/images --output_dir outputs/full_sahi
   python roi_based_sahi.py --input_dir sample_data/images --roi_mask_dir sample_data/sahi_tiling/roi_masks2 --output_dir outputs/selective_video
   ```

3. **Compare and visualize**
   ```bash
   python compare_detection_results.py
   ```
   - Dashboard: `outputs/compare_results/roi_performance_dashboard.png`

4. **Run all at once**
   ```bash
   bash run_all.sh
   ```

---

## Output Example

- `outputs/compare_results/roi_performance_dashboard.png`

---

## Full Experiment Results

- (You can add a screenshot of the full dashboard or key statistics here)
- (If needed, provide a Google Drive or other link for the full dataset/results)

---

## Requirements

- Python 3.8+
- numpy
- opencv-python
- matplotlib
- pandas

---

## Notes

- Sample data is for demonstration only.
- For full experiment, use your own dataset or contact me for access. 
