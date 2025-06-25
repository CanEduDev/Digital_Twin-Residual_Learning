#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import pandas as pd # Not explicitly used but good for data analysis later
from scipy.interpolate import interp1d
# from scipy.signal import savgol_filter # Not used in the script
from transforms3d.euler import euler2quat, quat2euler

# Assuming these are installed in your ROS 2 Python environment
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

# ======================  1. 配置  =============================
BAG_PATH       = '/home/sitong/Desktop/rosbag/u/rosbag2_1970_01_01-01_21_15'   # bag 目录 (Updated as per your script)
ODOM_TOPIC     = '/odometry/filtered'                   # nav_msgs/msg/Odometry
STEERING_TOPIC = '/rover/radio/steering'                # std_msgs/msg/Float32 (Updated)
THROTTLE_TOPIC = '/rover/radio/throttle'                # std_msgs/msg/Float32 (Updated)

STEER_LIMIT_DEG = 43.0        # 硬件极限
TARGET_HZ       = 200.0       # 目标采样率

# --- Control input delays ---
THROTTLE_INPUT_DELAY_S = 0.0  # Assumed delay for throttle in seconds
STEERING_INPUT_DELAY_S = 0.0   # Assumed delay for steering in seconds
# =============================================================


# ---------- 工具函数 ----------
def quat_to_yaw(qx, qy, qz, qw):
    """四元数 → yaw（弧度）"""
    return np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))


def resample(times, values, new_times, kind='linear'):
    """按 new_times 重采样（支持多列）"""
    if not isinstance(times, np.ndarray) or not isinstance(values, np.ndarray) or not isinstance(new_times, np.ndarray):
        raise TypeError("Inputs 'times', 'values', and 'new_times' must be numpy arrays.")
    if times.size == 0 or values.size == 0: # Handle empty input
        if values.ndim == 1:
            return np.full_like(new_times, np.nan, dtype=float)
        else: # ndim > 1
            return np.full((len(new_times), values.shape[1] if values.ndim > 1 else 1), np.nan, dtype=float)

    if values.ndim == 1:
        if len(times) == 1: # Handle single data point
             return np.full_like(new_times, values[0], dtype=values.dtype)
        f = interp1d(times, values, bounds_error=False,
                     fill_value=(values[0], values[-1]), kind=kind)
        return f(new_times)
    else: # ndim > 1
        resampled_values = np.zeros((len(new_times), values.shape[1]), dtype=values.dtype)
        for i in range(values.shape[1]):
            if len(times) == 1: # Handle single data point row for multi-column
                resampled_values[:, i] = np.full_like(new_times, values[0,i], dtype=values.dtype)
            else:
                f = interp1d(times, values[:, i], bounds_error=False,
                             fill_value=(values[0, i], values[-1, i]), kind=kind)
                resampled_values[:, i] = f(new_times)
        return resampled_values


# ---------- 2. 读取 rosbag ----------
def read_bag_series(bag_path, topic_name_requested, extractor_callback):
    """按 extractor_callback 回调提取指定 topic_name_requested 的序列数据"""
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
    
    actual_topic_name = topic_name_requested
    if topic_name_requested not in type_map:
        alt_topic_found = False
        for t_name_from_bag in type_map.keys():
            if topic_name_requested.strip('/') == t_name_from_bag.strip('/'):
                actual_topic_name = t_name_from_bag
                alt_topic_found = True
                print(f"ℹ️ Using actual topic name found in bag: '{actual_topic_name}' (for requested: '{topic_name_requested}')")
                break
        if not alt_topic_found:
            print(f"❌ Topic '{topic_name_requested}' not found in '{bag_path}'. Available topics: {list(type_map.keys())}")
            return np.array([]), np.array([])

    msg_type_str = type_map[actual_topic_name]
    try:
        msg_type = get_message(msg_type_str)
    except Exception as e:
        print(f"❌ Could not get message type for '{msg_type_str}': {e}")
        return np.array([]), np.array([])

    while reader.has_next():
        try:
            (topic_read, raw_data, timestamp_ns) = reader.read_next()
            if topic_read == actual_topic_name:
                msg = deserialize_message(raw_data, msg_type)
                
                current_timestamp_s = timestamp_ns / 1e9
                if hasattr(msg, 'header') and msg.header.stamp.sec > 0 : # Prefer header stamp if available and valid
                    current_timestamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                
                ts_list.append(current_timestamp_s)
                data_list.append(extractor_callback(msg))
        except Exception as e:
            print(f"⚠️ Error processing message for topic '{actual_topic_name}': {e}")
            continue 
    
    if not ts_list:
        print(f"⚠️ No messages successfully read for topic '{actual_topic_name}' in '{bag_path}'.")
        return np.array([]), np.array([])
        
    return np.array(ts_list), np.array(data_list)


# ---------- 2‑a Odometry 提取 ----------
def odom_extractor(msg):
    pos   = msg.pose.pose.position
    twist = msg.twist.twist
    ori   = msg.pose.pose.orientation
    yaw   = quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
    return np.array([pos.x, pos.y, pos.z,
                     twist.linear.x, twist.linear.y,
                     np.hypot(twist.linear.x, twist.linear.y), # speed
                     yaw,
                     twist.angular.z]) # yaw_rate


# ---------- 2‑b Control 信号提取 ----------
def float_extractor(msg): # Correct for std_msgs/msg/Float32
    return np.array([msg.data])


# ---------- 3. 主流程 ----------
def main():
    print(f"🚀 读取 odometry from {ODOM_TOPIC}...")
    odom_t, odom_data = read_bag_series(BAG_PATH, ODOM_TOPIC, odom_extractor)
    if odom_t.size == 0:
        print(f"❌ 未能从 {ODOM_TOPIC} 读取数据. 终止处理.")
        return

    print(f"🚀 读取 steering from {STEERING_TOPIC}...")
    steer_t, steer_val = read_bag_series(BAG_PATH, STEERING_TOPIC, float_extractor)
    if steer_t.size == 0:
        print(f"❌ 未能从 {STEERING_TOPIC} 读取数据. 终止处理.")
        return
        
    print(f"🚀 读取 throttle from {THROTTLE_TOPIC}...")
    throt_t, throt_val = read_bag_series(BAG_PATH, THROTTLE_TOPIC, float_extractor)
    if throt_t.size == 0:
        print(f"❌ 未能从 {THROTTLE_TOPIC} 读取数据. 终止处理.")
        return

    # Apply INDEPENDENT assumed delays to control input timestamps
    if STEERING_INPUT_DELAY_S != 0:
        print(f"ℹ️ Applying {STEERING_INPUT_DELAY_S*1000:.1f} ms delay to STEERING inputs.")
        steer_t_delayed = steer_t + STEERING_INPUT_DELAY_S
    else:
        print("ℹ️ No steering input delay applied (STEERING_INPUT_DELAY_S is 0).")
        steer_t_delayed = steer_t

    if THROTTLE_INPUT_DELAY_S != 0:
        print(f"ℹ️ Applying {THROTTLE_INPUT_DELAY_S*1000:.1f} ms delay to THROTTLE inputs.")
        throt_t_delayed = throt_t + THROTTLE_INPUT_DELAY_S
    else:
        print("ℹ️ No throttle input delay applied (THROTTLE_INPUT_DELAY_S is 0).")
        throt_t_delayed = throt_t
    
    # ---------- 4. 时间对齐到 TARGET_HZ ----------
    # Determine the common valid time range considering potentially delayed control signals
    # and ensuring arrays are not empty before accessing elements
    t_min_odom = odom_t[0] if odom_t.size > 0 else float('inf')
    t_max_odom = odom_t[-1] if odom_t.size > 0 else float('-inf')

    t_min_steer_delayed = steer_t_delayed[0] if steer_t_delayed.size > 0 else float('inf')
    t_max_steer_delayed = steer_t_delayed[-1] if steer_t_delayed.size > 0 else float('-inf')
    t_min_throt_delayed = throt_t_delayed[0] if throt_t_delayed.size > 0 else float('inf')
    t_max_throt_delayed = throt_t_delayed[-1] if throt_t_delayed.size > 0 else float('-inf')

    t_min_controls = min(t_min_steer_delayed, t_min_throt_delayed)
    t_max_controls = max(t_max_steer_delayed, t_max_throt_delayed)

    t0 = max(t_min_odom, t_min_controls)
    t_end = min(t_max_odom, t_max_controls)

    if t0 >= t_end:
        print(f"❌ No overlapping time range between odometry and control signals after applying delay(s).")
        print(f"   Odom effective range: [{t_min_odom:.2f}, {t_max_odom:.2f}]")
        print(f"   Steer (delayed) effective range: [{t_min_steer_delayed:.2f}, {t_max_steer_delayed:.2f}]")
        print(f"   Throttle (delayed) effective range: [{t_min_throt_delayed:.2f}, {t_max_throt_delayed:.2f}]")
        print(f"   Resulting common range for resampling: [{t0:.2f}, {t_end:.2f}]")
        return
        
    new_times = np.arange(t0, t_end, 1.0 / TARGET_HZ)
    if new_times.size == 0:
        print(f"❌ No common timestamps generated for resampling. Check time ranges and TARGET_HZ. t0={t0}, t_end={t_end}")
        return

    print(f"Resampling odometry data to {TARGET_HZ} Hz over range [{t0:.2f}, {t_end:.2f}]...")
    odom_interp = resample(odom_t, odom_data, new_times, kind='linear')
    
    print(f"Resampling steering data to {TARGET_HZ} Hz (using Zero-Order Hold)...")
    steer_interp  = resample(steer_t_delayed, steer_val.flatten(), new_times, kind='previous')
    
    print(f"Resampling throttle data to {TARGET_HZ} Hz (using Zero-Order Hold)...")
    throt_interp  = resample(throt_t_delayed, throt_val.flatten(), new_times, kind='previous')

    # ---------- 5. 后处理 ----------
    print("Processing and calculating derived values...")
    steer_deg = np.clip(steer_interp, -STEER_LIMIT_DEG, STEER_LIMIT_DEG)
    steer_norm = steer_deg / STEER_LIMIT_DEG
    steer_sin  = np.sin(np.deg2rad(steer_deg))
    steer_cos  = np.cos(np.deg2rad(steer_deg))

    # Assuming throt_interp represents a percentage for combined throttle/brake
    throttle_norm_combined = np.clip(throt_interp / 100.0, -1.0, 1.0) 
    throttle_pos = np.maximum(throttle_norm_combined, 0)  # Positive part [0, 1]
    brake_norm   = np.minimum(throttle_norm_combined, 0)  # Negative part [-1, 0]

    dt_resample = 1.0 / TARGET_HZ
    if odom_interp.shape[0] > 1 and odom_interp.shape[1] > 5: # Index 5 is speed
        acc = np.gradient(odom_interp[:, 5], dt_resample)
    else:
        print("⚠️ Not enough data or columns in odom_interp to calculate acceleration. Filling with zeros.")
        acc = np.zeros(len(new_times)) if len(new_times) > 0 else np.array([])


    # ---------- 6. 写 CSV ----------
    csv_name = "rosbag_aligned_data.csv"
    header = ["time",
              "pos_x", "pos_y", "pos_z",
              "vx", "vy", "speed",
              "yaw", "yaw_rate",
              "acceleration",
              "steer_deg", "steer_norm", "steer_sin", "steer_cos",
              "throttle_norm", "brake_norm"] # these are the columns written

    print(f"Writing aligned data to {csv_name}...")
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(new_times)):
            # Ensure all interpolated arrays have data for this index
            if i < odom_interp.shape[0] and i < acc.shape[0] and \
               i < steer_deg.shape[0] and i < steer_norm.shape[0] and \
               i < steer_sin.shape[0] and i < steer_cos.shape[0] and \
               i < throttle_norm_combined.shape[0] and i < brake_norm.shape[0]:
                row = [new_times[i],
                       odom_interp[i, 0], odom_interp[i, 1], odom_interp[i, 2], # pos_x, pos_y, pos_z
                       odom_interp[i, 3], odom_interp[i, 4], odom_interp[i, 5], # vx, vy, speed
                       odom_interp[i, 6], odom_interp[i, 7],                    # yaw, yaw_rate
                       acc[i],                                                  # calculated acceleration
                       steer_deg[i], steer_norm[i], steer_sin[i], steer_cos[i],
                       throttle_norm_combined[i], brake_norm[i]] # Using throttle_norm_combined for 'throttle_norm' column
                writer.writerow(row)
            else:
                print(f"⚠️ Warning: Skipping CSV row {i} due to data length mismatch after resampling or processing.")
                # This can happen if new_times extends beyond the valid range of some interpolated signals
                # Or if acc calculation resulted in a shorter array (though np.gradient should match input length)

    print(f"✅ 数据已导出到 {os.path.abspath(csv_name)}")


if __name__ == "__main__":
    main()