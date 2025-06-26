#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_process_bags.py - The main script to process all rosbags.

Discovers rosbag directories, splits them by scenario, processes each bag
by calling a processor function, and aggregates the results into two final CSVs
(training and evaluation), adding a 'session_id' to each recording.
"""
import os
import shutil
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d

# Assuming these are installed in your ROS 2 Python environment
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

# ======================  1. BATCH CONFIGURATION  =============================
# The main directory containing subfolders like 'left_turn', 'right_turn', etc.
DATA_DIR = '/home/sitong/Desktop/Digital_Twin-Residual_Learning/data'

# Define which scenarios go into which dataset.
TRAINING_SCENARIOS = ['left_turn', 'right_turn', 'o']
EVALUATION_SCENARIOS = ['zz'] # Using zigzag for evaluation is a great test of generalization

# Define output directory and final filenames
OUTPUT_DIR = '/home/sitong/Desktop/Digital_Twin-Residual_Learning/processed_datasets'
TRAINING_CSV_NAME = 'training_data_all_sessions.csv'
EVALUATION_CSV_NAME = 'evaluation_data_all_sessions.csv'
# ===========================================================================

# ======================  2. SINGLE BAG PROCESSOR  ==========================
# This section contains the logic from your original script, refactored into a
# single, reusable function.

# --- Constants for the processor ---
ODOM_TOPIC     = '/odometry/filtered'
STEERING_TOPIC = '/rover/radio/steering'
THROTTLE_TOPIC = '/rover/radio/throttle'
STEER_LIMIT_DEG = 43.0
TARGET_HZ       = 200.0

def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

def resample(times, values, new_times, kind='linear'):
    if not isinstance(times, np.ndarray) or not isinstance(values, np.ndarray) or not isinstance(new_times, np.ndarray):
        raise TypeError("Inputs 'times', 'values', and 'new_times' must be numpy arrays.")
    if times.size == 0 or values.size == 0:
        if values.ndim == 1: return np.full_like(new_times, np.nan, dtype=float)
        else: return np.full((len(new_times), values.shape[1] if values.ndim > 1 else 1), np.nan, dtype=float)
    if values.ndim == 1:
        if len(times) == 1: return np.full_like(new_times, values[0], dtype=values.dtype)
        f = interp1d(times, values, bounds_error=False, fill_value=(values[0], values[-1]), kind=kind)
        return f(new_times)
    else:
        resampled_values = np.zeros((len(new_times), values.shape[1]), dtype=values.dtype)
        for i in range(values.shape[1]):
            if len(times) == 1:
                resampled_values[:, i] = np.full_like(new_times, values[0,i], dtype=values.dtype)
            else:
                f = interp1d(times, values[:, i], bounds_error=False, fill_value=(values[0, i], values[-1, i]), kind=kind)
                resampled_values[:, i] = f(new_times)
        return resampled_values

def read_bag_series(bag_path, topic_name_requested, extractor_callback):
    ts_list, data_list = [], []
    reader = SequentialReader()
    try:
        reader.open(StorageOptions(uri=bag_path, storage_id='sqlite3'),
                    ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    except Exception as e:
        # print(f"❌ Error opening bag file '{bag_path}': {e}")
        return np.array([]), np.array([])
    topic_types = reader.get_all_topics_and_types()
    type_map = {entry.name: entry.type for entry in topic_types}
    if topic_name_requested not in type_map:
        # print(f"❌ Topic '{topic_name_requested}' not found in '{bag_path}'.")
        return np.array([]), np.array([])
    msg_type = get_message(type_map[topic_name_requested])
    while reader.has_next():
        try:
            (topic_read, raw_data, timestamp_ns) = reader.read_next()
            if topic_read == topic_name_requested:
                msg = deserialize_message(raw_data, msg_type)
                current_timestamp_s = timestamp_ns / 1e9
                if hasattr(msg, 'header') and msg.header.stamp.sec > 0:
                    current_timestamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                ts_list.append(current_timestamp_s)
                data_list.append(extractor_callback(msg))
        except Exception: continue
    return np.array(ts_list), np.array(data_list)

def odom_extractor(msg):
    pos = msg.pose.pose.position; twist = msg.twist.twist; ori = msg.pose.pose.orientation
    yaw = quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
    return np.array([pos.x, pos.y, yaw, np.hypot(twist.linear.x, twist.linear.y)])

def float_extractor(msg): return np.array([msg.data])

def process_bag_to_df(bag_path: str):
    """Processes a single rosbag and returns a pandas DataFrame, or None on failure."""
    odom_t, odom_data = read_bag_series(bag_path, ODOM_TOPIC, odom_extractor)
    if odom_t.size < 2: return None
    steer_t, steer_val = read_bag_series(bag_path, STEERING_TOPIC, float_extractor)
    if steer_t.size < 2: return None
    throt_t, throt_val = read_bag_series(bag_path, THROTTLE_TOPIC, float_extractor)
    if throt_t.size < 2: return None
    
    t0 = max(odom_t[0], steer_t[0], throt_t[0])
    t_end = min(odom_t[-1], steer_t[-1], throt_t[-1])
    if t0 >= t_end: return None

    new_times = np.arange(t0, t_end, 1.0 / TARGET_HZ)
    if new_times.size < 20: return None

    odom_interp = resample(odom_t, odom_data, new_times, kind='linear')
    steer_interp  = resample(steer_t, steer_val.flatten(), new_times, kind='previous')
    
    dt = 1.0 / TARGET_HZ
    speed = odom_interp[:, 3]
    acc = np.gradient(speed, dt)
    
    data = {
        "time": new_times,
        "pos_x": odom_interp[:, 0],
        "pos_y": odom_interp[:, 1],
        "yaw": odom_interp[:, 2],
        "speed": speed,
        "acceleration": acc,
        "steer_deg": np.clip(steer_interp, -STEER_LIMIT_DEG, STEER_LIMIT_DEG)
    }
    return pd.DataFrame(data)
# ===========================================================================


# ======================  3. BATCH ORCHESTRATOR  ============================
def find_rosbags(base_dir, scenarios):
    bag_paths = []
    print(f"Searching for rosbags in scenarios: {scenarios}")
    for scenario in scenarios:
        scenario_path = Path(base_dir) / scenario
        if not scenario_path.is_dir(): continue
        for metadata_file in sorted(scenario_path.rglob('metadata.yaml')):
            bag_paths.append(str(metadata_file.parent))
    return bag_paths

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # --- Find and split bags ---
    train_bags = find_rosbags(DATA_DIR, TRAINING_SCENARIOS)
    eval_bags = find_rosbags(DATA_DIR, EVALUATION_SCENARIOS)

    # --- Process and Aggregate ---
    for description, bag_list, filename in [
        ("Training Set", train_bags, TRAINING_CSV_NAME),
        ("Evaluation Set", eval_bags, EVALUATION_CSV_NAME)
    ]:
        if not bag_list:
            print(f"No bags found for {description}. Skipping.")
            continue
        
        print(f"\n🚀 Processing {len(bag_list)} bags for: {description}")
        
        all_dfs = []
        for i, bag_path in enumerate(tqdm(bag_list, desc=f"Processing {description}")):
            df = process_bag_to_df(bag_path)
            if df is not None and not df.empty:
                df['session_id'] = i  # Add the unique session ID
                all_dfs.append(df)
        
        if not all_dfs:
            print(f"❌ No bags were successfully processed for {description}.")
            continue
            
        print(f"\n🔄 Aggregating {len(all_dfs)} processed sessions...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        final_output_path = Path(OUTPUT_DIR) / filename
        combined_df.to_csv(final_output_path, index=False)
        print(f"✅ Successfully created aggregated file: {final_output_path}")

    print("\n🎉 Batch processing complete!")

if __name__ == "__main__":
    # You MUST source ROS 2 before running this script
    # e.g., source /opt/ros/humble/setup.bash
    main()
# ===========================================================================