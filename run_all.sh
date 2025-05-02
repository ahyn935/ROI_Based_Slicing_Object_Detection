#!/bin/bash
python basic_sahi.py --input_dir sample_data/images --output_dir outputs/full_sahi
python roi_based_sahi.py --input_dir sample_data/images --roi_mask_dir sample_data/sahi_tiling/roi_masks2 --output_dir outputs/selective_video
python compare_detection_results.py 