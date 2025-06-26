#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rosbag_processor.py - A reusable module to process a single rosbag.
"""

import os
import csv
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from transforms3d.euler import quat2euler

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

# ======================  1. 配置  =============================
ODOM_TOPIC     = '/odometry/filtered'
STEERING_TOPIC = '/rover/radio/steering'
THROTTLE_TOPIC = '/rover/radio/throttle'

STEER_LIMIT_DEG = 43.0
TARGET_HZ       = 200.0

THROTTLE_INPUT_DELAY_S = 0.0
STEERING_INPUT_DELAY_S = 0.0
# =============================================================

# ---------- 工具函数 ----------
def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

def resample(times, values, new_times, kind='linear'):
    if not isinstance(times, np.ndarray) or not isinstance(values, np.ndarray) or not isinstance(new_times, np.ndarray):
        raise TypeError("Inputs 'times', 'values', and 'new_times' must be numpy arrays.")
    if times.size == 0 or values.size == 0:
        if values.ndim == 1:
            return np.full_like(new_times, np.nan, dtype=float)
        else:
            return np.full((len(new_times), values.shape[1] if values.ndim > 1 else 1), np.nan, dtype=float)
    if values.ndim == 1:
        if len(times) == 1:
             return np.full_like(new_times, values[0], dtype=values.dtype)
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
        print(f"❌ Error opening bag file '{bag_path}': {e}")
        return np.array([]), np.array([])

    topic_types = reader.get_all_topics_and_types()
    type_map = {entry.name: entry.type for entry in topic_types}
    
    if topic_name_requested not in type_map:
        print(f"❌ Topic '{topic_name_requested}' not found in '{bag_path}'.")
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
        except Exception as e:
            continue
    return np.array(ts_list), np.array(data_list)

def odom_extractor(msg):
    pos = msg.pose.pose.position; twist = msg.twist.twist; ori = msg.pose.pose.orientation
    yaw = quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
    return np.array([pos.x, pos.y, pos.z, twist.linear.x, twist.linear.y, np.hypot(twist.linear.x, twist.linear.y), yaw, twist.angular.z])

def float_extractor(msg):
    return np.array([msg.data])

def process_bag_to_csv(bag_path: str, output_csv_path: str) -> bool:
    """
    Processes a single rosbag directory and writes the aligned data to a CSV file.
    Returns True on success, False on failure.
    """
    print(f"--- Processing Bag: {bag_path} ---")
    odom_t, odom_data = read_bag_series(bag_path, ODOM_TOPIC, odom_extractor)
    if odom_t.size == 0: return False

    steer_t, steer_val = read_bag_series(bag_path, STEERING_TOPIC, float_extractor)
    if steer_t.size == 0: return False
        
    throt_t, throt_val = read_bag_series(bag_path, THROTTLE_TOPIC, float_extractor)
    if throt_t.size == 0: return False

    steer_t_delayed = steer_t + STEERING_INPUT_DELAY_S
    throt_t_delayed = throt_t + THROTTLE_INPUT_DELAY_S
    
    t0 = max(odom_t[0], steer_t_delayed[0], throt_t_delayed[0])
    t_end = min(odom_t[-1], steer_t_delayed[-1], throt_t_delayed[-1])

    if t0 >= t_end:
        print(f"  ❌ No overlapping time range in {bag_path}. Skipping.")
        return False
        
    new_times = np.arange(t0, t_end, 1.0 / TARGET_HZ)
    if new_times.size < 10: # Skip very short clips
        print(f"  ❌ Clip too short ({new_times.size} samples) in {bag_path}. Skipping.")
        return False

    odom_interp = resample(odom_t, odom_data, new_times, kind='linear')
    steer_interp  = resample(steer_t_delayed, steer_val.flatten(), new_times, kind='previous')
    throt_interp  = resample(throt_t_delayed, throt_val.flatten(), new_times, kind='previous')

    steer_deg = np.clip(steer_interp, -STEER_LIMIT_DEG, STEER_LIMIT_DEG)
    acc = np.gradient(odom_interp[:, 5], 1.0 / TARGET_HZ)

    header = ["time", "pos_x", "pos_y", "yaw", "speed", "acceleration", "steer_deg"]

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(new_times)):
            row = [
                new_times[i],
                odom_interp[i, 0], # pos_x
                odom_interp[i, 1], # pos_y
                odom_interp[i, 6], # yaw
                odom_interp[i, 5], # speed
                acc[i],
                steer_deg[i]
            ]
            writer.writerow(row)
    print(f"  ✅ Successfully wrote to {output_csv_path}")
    return True