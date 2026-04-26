import json
import os
import numpy as np
import shutil
import random
from tqdm import tqdm

DRY_RUN = False 

TARGET_CAPS = {
    "ac": 2200,      
    "aitz": 2200,    
    "odyssey": 4400  
}
RATIOS = {"easy": 0.4, "medium": 0.4, "hard": 0.2}

WATER_THRESHOLDS = {
    "AC_test.json": 0.6,
    "AITZ_test.json": 0.8,
    "Odyssey_test.json": 0.8
}

VOTE_THRESHOLDS = {
    "AC_test.json": 0.9,
    "AITZ_test.json": 0.5,
    "Odyssey_test.json": 0.5
}

BASE_DIR = "/data1/home/wuzheng/mobile-SPEAR"
SCRIPT_DIR = os.path.join(BASE_DIR, "data_script")
FINAL_OUTPUT_ROOT = "/data2/datasets/OS-SPEAR/P-subset"


DATASET_MAP = {
    "ac": {
        "json": "AC_test.json",
        "large": ["owl32b_ac.txt", "qwen32b_ac.txt", "tars72b_ac.txt"],
        "small": ["qwen3_2b_ac.txt", "tars2b_ac.txt"]
    },
    "aitz": {
        "json": "AITZ_test.json",
        "large": ["owl32b_aitz.txt", "qwen32b_aitz.txt", "tars72b_aitz.txt"],
        "small": ["qwen3_2b_aitz.txt", "tars2b_aitz.txt"]
    },
    "odyssey": {
        "json": "Odyssey_test.json",
        "large": ["owl32b_odyssey.txt", "qwen32b_odyssey.txt", "tars72b_odyssey.txt"],
        "small": ["qwen3_2b_odyssey.txt", "tars2b_odyssey.txt"]
    }
}


def parse_txt_log(file_path):
    results = []
    if not os.path.exists(file_path): return None
    with open(file_path, 'r') as f:
        for line in f:
            if "SR:" in line:
                try:
                    part = line.split("SR:")[1].strip()
                    score = int(part.split(',')[0].strip())
                    results.append(score)
                except: results.append(0)
    return results

def run_scheme_b():
    print(f"🚀 Starting final generation (Scheme B: Small & High Impact)")
    print(f"🌊 Strategy: Differentiated water removal + 4:4:2 stratified sampling")
    
    global_total_steps = 0
    final_export_task = []

    for ds_name, config in DATASET_MAP.items():
        json_file = config['json']
        target_total = TARGET_CAPS[ds_name]
        water_thresh = WATER_THRESHOLDS[json_file]
        vote_thresh = VOTE_THRESHOLDS[json_file]
        
        print(f"\n====== Dataset: {ds_name} ======")
        print(f"  Config: Water removal>{water_thresh} | Vote>={vote_thresh} | Target={target_total}")
        
        # Load
        json_path = os.path.join(SCRIPT_DIR, json_file)
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: json_data = json.load(f)
            
        large_logs = [parse_txt_log(os.path.join(BASE_DIR, f)) for f in config['large']]
        small_logs = [parse_txt_log(os.path.join(BASE_DIR, f)) for f in config['small']]
        large_logs = [l for l in large_logs if l is not None]
        small_logs = [l for l in small_logs if l is not None]

        if not large_logs: continue

        # Cut Trajectories
        traj_indices = []
        curr = []
        last_task = None
        min_len = min(len(json_data), min([len(l) for l in large_logs + small_logs]))
        
        for i in range(min_len):
            task = json_data[i].get('task', '')
            if task != last_task and i > 0:
                traj_indices.append(curr)
                curr = []
            curr.append(i)
            last_task = task
        if curr: traj_indices.append(curr)

        # Buckets: Store (indices, avg_large_score)
        buckets = {"easy": [], "medium": [], "hard": []}
        count_dropped = 0
        
        has_small = len(small_logs) > 0

        for indices in traj_indices:
            # 1. Water removal
            if has_small:
                s_scores = [sum([log[i] for i in indices])/len(indices) for log in small_logs]
                avg_small = sum(s_scores) / len(s_scores)
                if avg_small > water_thresh:
                    count_dropped += 1
                    continue
            
            # 2. Voting levels
            votes = 0
            l_scores_temp = []
            for log in large_logs:
                step_scores = [log[i] for i in indices]
                acc = sum(step_scores)/len(step_scores)
                l_scores_temp.append(acc)
                if acc >= vote_thresh:
                    votes += 1
            
            # Sorting criterion within the group: average score of large models
            avg_large = sum(l_scores_temp) / len(l_scores_temp)
            item = (indices, avg_large)

            if votes == 3: buckets["easy"].append(item)
            elif votes == 2: buckets["medium"].append(item)
            elif votes == 1: buckets["hard"].append(item)
            # 0 votes are discarded

        # 3. Stratified sampling (Scheme B)
        # Sort each bucket by score
        for k in buckets:
            buckets[k].sort(key=lambda x: x[1], reverse=True)

        final_indices_map = {"easy": [], "medium": [], "hard": []}
        current_ds_steps = 0
        
        # Target quota
        target_easy = int(target_total * RATIOS['easy'])
        target_medium = int(target_total * RATIOS['medium'])
        target_hard = int(target_total * RATIOS['hard'])
        
        # Fill function
        def fill_bucket(target_count, source_bucket, dest_list):
            added = 0
            for item in source_bucket:
                traj_len = len(item[0])
                # Slightly loose overflow check to meet the number
                if added < target_count: 
                    dest_list.extend(item[0])
                    added += traj_len
            return added

        steps_e = fill_bucket(target_easy, buckets['easy'], final_indices_map['easy'])
        steps_m = fill_bucket(target_medium, buckets['medium'], final_indices_map['medium'])
        steps_h = fill_bucket(target_hard, buckets['hard'], final_indices_map['hard'])
        
        total_steps = steps_e + steps_m + steps_h
        global_total_steps += total_steps

        print(f"  - Dropped water questions: {count_dropped}")
        print(f"  - Inventory status: Easy({len(buckets['easy'])}) | Mid({len(buckets['medium'])}) | Hard({len(buckets['hard'])})")
        print(f"  ✅ Final cut: {total_steps} Steps")
        print(f"     Details: Easy({steps_e}) | Medium({steps_m}) | Hard({steps_h})")
        
        if not DRY_RUN:
            final_export_task.append({
                "json_base": json_file.replace(".json", ""),
                "ds_name": ds_name,
                "data_map": final_indices_map, # {easy: [], medium: [], hard: []}
                "source": json_data
            })

    print(f"\n🏁 Global total steps: {global_total_steps}")
    
    # --- Generation phase ---
    if not DRY_RUN:
        print("\n📦 Generating files and backing up images...")
        if not os.path.exists(FINAL_OUTPUT_ROOT): os.makedirs(FINAL_OUTPUT_ROOT)
        
        for task in final_export_task:
            ds_name = task['ds_name']
            target_img_dir = os.path.join(FINAL_OUTPUT_ROOT, ds_name, "images")
            os.makedirs(target_img_dir, exist_ok=True)
            
            # Export Easy/Medium/Hard files
            for level, indices in task['data_map'].items():
                if not indices: continue
                
                json_name = f"{task['json_base']}_{level}.json"
                new_data = []
                print(f"  -> Writing {json_name}...")
                
                for idx in tqdm(indices, leave=False):
                    record = task['source'][idx].copy()
                    
                    # Copy images and update paths
                    src_path = record['image_path']
                    fname = os.path.basename(src_path)
                    
                    # Path search
                    real_src = src_path
                    if not os.path.exists(real_src):
                        roots = [f"/data2/datasets/{ds_name}/images", "/data2/datasets/android_control/images"]
                        for r in roots:
                            if os.path.exists(os.path.join(r, fname)):
                                real_src = os.path.join(r, fname); break
                        
                    if os.path.exists(real_src):
                        dst_path = os.path.join(target_img_dir, fname)
                        if not os.path.exists(dst_path):
                            shutil.copy2(real_src, dst_path)
                        record['image_path'] = dst_path
                    
                    new_data.append(record)
                    
                with open(os.path.join(FINAL_OUTPUT_ROOT, json_name), 'w') as f:
                    json.dump(new_data, f, indent=4)

    print("🎉 Task completed successfully!")

if __name__ == "__main__":
    run_scheme_b()
