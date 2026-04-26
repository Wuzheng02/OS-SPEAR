import yaml
import os
import json
import time
from tqdm import tqdm
from get_action import *
from transfer import *
from eval_single import single_eval
from datetime import datetime

with open('config.yaml', 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)


MODEL_PATH = config["MODEL_PATH"]
DATA_PATH = config["DATA_PATH"]
LOG_PATH = config["LOG_PATH"]
MODEL = config["MODEL"]


def get_action(model, processor, obs, tokenizer=None):
    if MODEL == "OS-Atlas":
        try:
            action, input_tokens, output_tokens = get_action_atlas(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0

    elif MODEL == "UI-TARS":
        try:
            action, input_tokens, output_tokens = get_action_tars(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:
            action = transfer_tars2atlas(action)
        except Exception as e:
            action = "error"
   

    elif MODEL == "GUI-owl":
        try:
            action, input_tokens, output_tokens = get_action_owl(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:            
            action = transfer_owl2atlas(action, obs['image_path'])
            print(action)
        except Exception as e:
            action = "error"

    elif MODEL == "Qwen2.5-VL":
        try:
            action, input_tokens, output_tokens = get_action_qwen25(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:            
            action = transfer_qwen25toatlas(action, obs['image_path'])
            print(action)
        except Exception as e:
            action = "error"

    elif MODEL == "UI-TARS-1.5":
        try:
            action, input_tokens, output_tokens = get_action_tars15(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:
            action = transfer_tars15toatlas(action, obs['image_path'])
        except Exception as e:
            action = "error"

    elif MODEL == "GELab":
        try:
            action, input_tokens, output_tokens = get_action_gelab(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:
            action = transfer_gelab2atlas(action, obs['image_path'])
        except Exception as e:
            action = "error"
    
    elif MODEL == "Qwen3-VL":
        try:
            action, input_tokens, output_tokens = get_action_qwen3vl(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:
            action = transfer_qwen3vl2atlas(action, obs['image_path'])
        except Exception as e:
            action = "error"
    
    elif MODEL == "UI-Venus":
        try:
            action, input_tokens, output_tokens = get_action_UIvenus(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:
            action = transfer_venus2atlas(action, obs['image_path'])
        except Exception as e:
            action = "error"
    
    elif MODEL == "AgentCPM-GUI":
        try:
            action, input_tokens, output_tokens = get_action_agentcpm(model, tokenizer, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:
            action = transfer_cpm2atlas(action, obs['image_path'])
        except Exception as e:
            action = "error"
    
    elif MODEL == "MAI-UI":
        try:
            action, input_tokens, output_tokens = get_action_maiui(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:
            action = transfer_maiui2atlas(action, obs['image_path'])
        except Exception as e:
            action = "error"   
    
    elif MODEL == "UI-Venus-1.5":
        try:
            action, input_tokens, output_tokens = get_action_UIvenus15(model, processor, obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:
            action = transfer_venus15toatlas(action, obs['image_path'])
        except Exception as e:
            action = "error"
    
    elif MODEL == "GLM-4.5V":
        try:
            action, input_tokens, output_tokens = get_action_glm45v(obs)
        except Exception as e:
            action = "error"
            input_tokens = 0
            output_tokens = 0
        try:
            action = transfer_glms45vtoatlas(action, obs['image_path'])
        except Exception as e:
            action = "error"
    return action, input_tokens, output_tokens


# with open(JSON_PATH, "r", encoding="utf-8") as f:
#     data_list = json.load(f)
# print(f"Loaded {len(data_list)} items from JSON.")


def S_test_loop(model, processor, logs_path, data_path, tokenizer=None):
    s_subset_path = os.path.join(logs_path, "S-subset")
    os.makedirs(s_subset_path, exist_ok=True)

    # Define the files to process
    json_files = [
        "S-subset/EnvDistraction.json", 
        "S-subset/GUI-Robust.json", 
        "S-subset/JARVIS.json"
    ]

    # Metrics to track
    overall_gold = 0
    overall_dist = 0
    overall_inv = 0
    total_count = 0

    # Process each file
    for file_idx, json_file in enumerate(json_files):
        print(f"\n=== Processing file {json_file} ===")
        json_path = os.path.join(data_path, json_file)
        
        file_name_only = os.path.basename(json_file)
        
        with open(json_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(s_subset_path, f"eval_log_{file_name_only}_{ts}_{file_idx + 1}.txt")
        logs = []
        
        gold = 0
        dist = 0
        inv = 0
        
        for idx, obs in enumerate(tqdm(data_list, desc=f"Evaluating {json_file}", ncols=100)):
            print(f"\n=== Processing item {idx} ===")
            action, input_tokens, output_tokens = get_action(model, processor, obs)
            # print(obs)
            # print(action)
            label = None  # Default to None for cases where the key isn't available

            try:
                if json_file == "S-subset/EnvDistraction.json":
                    # Dynamically collect all keys like 'gold_action_数字' and 'bad_action_数字'
                    gold_actions = [obs[key] for key in obs if key.startswith("gold_action_")]
                    bad_actions = [obs[key] for key in obs if key.startswith("bad_action_")]
                    print(action)
                    # Check for gold actions
                    is_gold = any(single_eval(action, gold_action)[1] == 1 for gold_action in gold_actions)
                    if is_gold:
                        gold += 1
                    else:
                        # Check for bad actions
                        dist_sr = 0
                        for bad_action in bad_actions:
                            _, dist_sr = single_eval(action, bad_action)
                            if dist_sr == 1:
                                dist += 1
                                break
                        if dist_sr != 1:
                            inv += 1

                elif json_file == "S-subset/GUI-Robust.json":
                    label = obs["action"]
                    if label is not None:
                        _, sr = single_eval(action, label)
                        print(action)
                        print(label)
                        # print(sr)
                        if sr == 1:
                            gold += 1
                        else:
                            dist += 1

                elif json_file == "S-subset/JARVIS.json":
                    good_action = obs["good_action"]
                    if good_action is not None:
                        is_gold = single_eval(action, good_action)[1] == 1
                        if is_gold:
                            gold += 1
                        else:
                            dist_sr = single_eval(action, obs["bad_action"])[1]
                            if dist_sr == 1:
                                dist += 1
                            else:
                                inv += 1

            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                continue

            # Calculate percentages for Gold, Dist, and Inv for the current step
            total_actions = gold + dist + inv
            if total_actions > 0:
                gold_percentage = (gold / total_actions) * 100
                dist_percentage = (dist / total_actions) * 100
                inv_percentage = (inv / total_actions) * 100
            else:
                gold_percentage = dist_percentage = inv_percentage = 0

            # Logging per step for each item processed
            logs.append(f"=== Step {idx} ===\n")
            logs.append(f"Action: {action}\n")
            if label is not None:  # Only log the label if it's available
                logs.append(f"Label: {label}\n")
            logs.append(f"Gold: {gold}, Dist: {dist}, Inv: {inv}\n")
            logs.append(f"Gold Percentage: {gold_percentage:.2f}%\n")
            logs.append(f"Dist Percentage: {dist_percentage:.2f}%\n")
            logs.append(f"Inv Percentage: {inv_percentage:.2f}%\n")
            logs.append(f"==============================\n")

        # Log results for this file
        logs.append(f"\n=== {json_file} Evaluation Metrics ===\n")
        logs.append(f"Gold: {gold}\n")
        logs.append(f"Dist: {dist}\n")
        logs.append(f"Inv: {inv}\n")
        logs.append(f"Gold Percentage: {(gold / len(data_list)) * 100:.2f}%\n")
        logs.append(f"Dist Percentage: {(dist / len(data_list)) * 100:.2f}%\n")
        logs.append(f"Inv Percentage: {(inv / len(data_list)) * 100:.2f}%\n")

        # Update overall metrics
        overall_gold += gold
        overall_dist += dist
        overall_inv += inv
        total_count += len(data_list)

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.writelines(logs)

        print(f"Finished evaluating {json_file}. Gold: {gold}, Dist: {dist}, Inv: {inv}")

    # Final report
    avg_gold = (overall_gold / total_count) * 100 if total_count > 0 else 0
    avg_dist = (overall_dist / total_count) * 100 if total_count > 0 else 0
    avg_inv = (overall_inv / total_count) * 100 if total_count > 0 else 0

    overall_log_path = os.path.join(s_subset_path, "overall_eval_log.txt")
    with open(overall_log_path, "w", encoding="utf-8") as overall_log_file:
        overall_log_file.write(f"Avg Gold: {avg_gold:.2f}%\n")
        overall_log_file.write(f"Avg Dist: {avg_dist:.2f}%\n")
        overall_log_file.write(f"Avg Inv: {avg_inv:.2f}%\n")

    print(f"Overall Evaluation finished. Avg Gold: {avg_gold:.2f}%, Avg Dist: {avg_dist:.2f}%, Avg Inv: {avg_inv:.2f}%")
    return 0

def P_test_loop(model, processor, logs_path, data_path, tokenizer=None): 
    p_subset_path = os.path.join(logs_path, "P-subset")
    os.makedirs(p_subset_path, exist_ok=True)
    json_files = [
        f"{data_path}P-subset/{dataset}_test_{difficulty}.json"
        for dataset in ["AC", "AITZ", "Odyssey"]
        for difficulty in ["easy", "medium", "hard"]
    ]

    overall_total_type = 0
    overall_total_SR = 0
    overall_count = 0
    overall_inference_time = 0
    overall_token_count = 0
    overall_token_count_input = 0
    overall_success_traj = 0
    overall_total_traj = 0
    
    difficulty_metrics = {
        'easy': {'total_type': 0, 'total_SR': 0, 'count': 0, 'success_traj': 0, 'total_traj': 0},
        'medium': {'total_type': 0, 'total_SR': 0, 'count': 0, 'success_traj': 0, 'total_traj': 0},
        'hard': {'total_type': 0, 'total_SR': 0, 'count': 0, 'success_traj': 0, 'total_traj': 0},
    }
    
    for file_idx, json_file in enumerate(json_files):
        filename_base = json_file.split('/')[-1].replace('.json', '')
        parts = filename_base.split('_')
        dataset, difficulty = parts[0], parts[-1]
        
        print(f"\n=== Processing file {json_file} ===")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        file_total_type = 0
        file_total_SR = 0
        file_count = 0
        file_success_traj = 0
        file_total_traj = 0
        
        logs = []
        traj_SR_list = []
        prev_task = None

        for idx, obs in enumerate(tqdm(data_list, desc=f"Evaluating {filename_base}", ncols=100)):
            start_time = time.time()
            action, output_tokens, input_tokens = get_action(model, processor, obs)
            elapsed_time = time.time() - start_time
            
            label = obs["action"]
            try:
                _type, SR = single_eval(action, label)
            except Exception:
                _type, SR = 0, 0
            
            file_total_type += _type
            file_total_SR += SR
            file_count += 1
            
            overall_total_type += _type
            overall_total_SR += SR
            overall_count += 1
            overall_inference_time += elapsed_time
            overall_token_count += output_tokens
            overall_token_count_input += input_tokens

            difficulty_metrics[difficulty]['total_type'] += _type
            difficulty_metrics[difficulty]['total_SR'] += SR
            difficulty_metrics[difficulty]['count'] += 1
            logs.append(f"=== Step {idx} ===\n")
            logs.append(f"action: {action}\n")
            logs.append(f"label: {label}\n")
            logs.append(f"type: {_type}, SR: {SR}\n")
            logs.append(f"inference_time: {elapsed_time:.4f}s\n")
            logs.append(f"==============================\n")
            
            cur_task = obs["task"]
            task_changed = (prev_task is not None and cur_task != prev_task)
            is_last_item = (idx == len(data_list) - 1)
            traj_SR_list.append(SR)
            
            if task_changed or is_last_item:
                file_total_traj += 1
                is_suc = all(x == 1 for x in traj_SR_list)
                if is_suc:
                    file_success_traj += 1
                
                difficulty_metrics[difficulty]['total_traj'] += 1
                if is_suc:
                    difficulty_metrics[difficulty]['success_traj'] += 1
                
                traj_SR_list = []
            
            prev_task = cur_task

        file_success_rate = (file_success_traj / file_total_traj) * 100 if file_total_traj > 0 else 0
        file_type_pct = (file_total_type / file_count) * 100 if file_count > 0 else 0
        file_sr_pct = (file_total_SR / file_count) * 100 if file_count > 0 else 0
        
        logs.append(f"\n=== {json_file} LOCAL Evaluation Metrics ===\n")
        logs.append(f"Success rate: {file_success_traj}/{file_total_traj} Trajectories ({file_success_rate:.2f}%).\n")
        logs.append(f"Total SR: {file_total_SR}, Percentage: {file_sr_pct:.2f}%\n")
        logs.append(f"Total type: {file_total_type}, Percentage: {file_type_pct:.2f}%\n")
        logs.append(f"Count (Steps in this file): {file_count}\n")
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(p_subset_path, f"eval_log_{dataset}_{difficulty}_{ts}_{file_idx + 1}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(logs)

        overall_success_traj += file_success_traj
        overall_total_traj += file_total_traj

    avg_time = overall_inference_time / overall_count if overall_count > 0 else 0
    avg_tokens = overall_token_count / overall_count if overall_count > 0 else 0
    avg_tokens_input = overall_token_count_input / overall_count if overall_count > 0 else 0

    with open(os.path.join(p_subset_path, "overall_eval_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"Overall Success rate: {overall_success_traj}/{overall_total_traj} Trajectories ({(overall_success_traj/overall_total_traj*100 if overall_total_traj>0 else 0):.2f}%).\n")
        f.write(f"Total type: {overall_total_type}, Percentage: {(overall_total_type/overall_count*100 if overall_count>0 else 0):.2f}%\n")
        f.write(f"Total SR: {overall_total_SR}, Percentage: {(overall_total_SR/overall_count*100 if overall_count>0 else 0):.2f}%\n")
        f.write(f"Average Inference Time: {avg_time:.4f} s/step\n")
        f.write(f"Average Output Tokens: {avg_tokens:.2f} tokens/step\n")
        f.write(f"Average Input Tokens: {avg_tokens_input:.2f} tokens/step\n")
        f.write(f"Total Count: {overall_count}\n")
    
    with open(os.path.join(p_subset_path, "difficulty_eval_log.txt"), "w", encoding="utf-8") as f:
        for diff in ['easy', 'medium', 'hard']:
            m = difficulty_metrics[diff]
            s_rate = (m['success_traj'] / m['total_traj']) * 100 if m['total_traj'] > 0 else 0
            f.write(f"\n=== {diff.capitalize()} Difficulty ===\n")
            f.write(f"Success rate: {m['success_traj']}/{m['total_traj']} Trajectories ({s_rate:.2f}%).\n")
            f.write(f"Total SR: {m['total_SR']}, Percentage: {(m['total_SR']/m['count']*100 if m['count']>0 else 0):.2f}%\n")
            f.write(f"Total type: {m['total_type']}, Percentage: {(m['total_type']/m['count']*100 if m['count']>0 else 0):.2f}%\n")
            f.write(f"Count: {m['count']}\n")
    
    return 0

def R_test_loop(model, processor, logs_path, data_path, tokenizer=None):
    r_subset_path = os.path.join(logs_path, "R-subset")
    os.makedirs(r_subset_path, exist_ok=True)

    r_file_path = os.path.join(data_path, "R-subset/R_subset.json")
    with open(r_file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    ts = time.strftime("%Y%m%d_%H%M%S")
    
    round_logs = {i: [] for i in range(1, 12)} 

    round_stats = {i: {"total_type": 0, "total_SR": 0, "count": 0, "success_traj": 0, "total_traj": 0} for i in range(1, 12)}

    for idx, obs in enumerate(tqdm(data_list, desc="Evaluating R-subset.json", ncols=100)):
        print(f"\n=== Processing item {idx} ===")

        original_action = obs["action"]
        original_image_path = obs["image_path"]
        original_task = obs["task"]
        
        for round_idx in range(1, 12):
            if round_idx == 1:
                action, input_tokens, output_tokens = get_action(model, processor, obs)
                label = obs["action"]
            elif round_idx == 2:
                obs["image_path"] = obs["image_path_mask"]
                action, input_tokens, output_tokens = get_action(model, processor, obs)
                label = obs["action"]
            elif round_idx == 3:
                obs["image_path"] = obs["image_path_zoomin"]
                label = obs["action_zoomin"]
                action, input_tokens, output_tokens = get_action(model, processor, obs)
            elif round_idx == 4:
                obs["image_path"] = obs["image_path_gauss_30"]
                label = original_action
                action, input_tokens, output_tokens = get_action(model, processor, obs)
            elif round_idx == 5:
                obs["image_path"] = obs["image_path_gauss_50"]
                action, input_tokens, output_tokens = get_action(model, processor, obs)
                label = obs["action"]
            elif round_idx == 6:
                obs["image_path"] = obs["image_path_gauss_70"]
                action, input_tokens, output_tokens = get_action(model, processor, obs)
                label = obs["action"]
            elif round_idx == 7:
                obs["image_path"] = original_image_path
                obs["task"] = original_task + " Your current task status:" + obs["State_Conflict"]
                action, input_tokens, output_tokens = get_action(model, processor, obs)
                label = obs["action"]
            elif round_idx == 8:
                obs["task"] = original_task + " Your historical operations: " + " ".join(str(item) for item in obs["bad_history"])
                action, input_tokens, output_tokens = get_action(model, processor, obs)
                label = obs["action"]
            elif round_idx == 9:
                obs["task"] = original_task + " Current task's SOP: " + " ".join(str(item) for item in obs["random_knowledge"])
                action, input_tokens, output_tokens = get_action(model, processor, obs)
                label = obs["action"]
            elif round_idx == 10:
                obs["task"] = original_task + " Your memory: " + " ".join(str(item) for item in obs["irrelevant_memories"])
                action, input_tokens, output_tokens = get_action(model, processor, obs)
                label = obs["action"]
            elif round_idx == 11:
                obs["task"] = original_task + " Current task's SOP: " + " ".join(str(item) for item in obs["irrelevant_knowledge"])
                action, input_tokens, output_tokens = get_action(model, processor, obs)
                label = obs["action"]

            try:
                _type, SR = single_eval(action, label)
            except Exception as e:
                _type = 0
                SR = 0

            # Log the result for this round
            round_logs[round_idx].append(f"Item {idx} - Action: {action}\nLabel: {label}\nType: {_type}, SR: {SR}\n==============================\n")
            
            # Update statistics for this round
            round_stats[round_idx]["total_type"] += _type
            round_stats[round_idx]["total_SR"] += SR
            round_stats[round_idx]["count"] += 1

            if SR == 1:
                round_stats[round_idx]["success_traj"] += 1
            
            round_stats[round_idx]["total_traj"] += 1

            print(f"[Round {round_idx}] action = {action}")
            print(f"[Round {round_idx}] label = {label}")
            print(f"[Round {round_idx}] type = {_type}")
            print(f"[Round {round_idx}] SR = {SR}")
        
    # After processing all items, save the logs and statistics for each round
    for round_idx in range(1, 12):
        # Save logs for the current round
        round_log_path = os.path.join(r_subset_path, f"round_{round_idx}_eval_log_{ts}.txt")
        with open(round_log_path, "w", encoding="utf-8") as log_file:
            log_file.writelines(round_logs[round_idx])

        # Calculate round metrics
        success_rate = (round_stats[round_idx]["success_traj"] / round_stats[round_idx]["total_traj"]) * 100 if round_stats[round_idx]["total_traj"] > 0 else 0
        type_percentage = (round_stats[round_idx]["total_type"] / round_stats[round_idx]["count"]) * 100 if round_stats[round_idx]["count"] > 0 else 0
        sr_percentage = (round_stats[round_idx]["total_SR"] / round_stats[round_idx]["count"]) * 100 if round_stats[round_idx]["count"] > 0 else 0

        # Write round summary
        with open(round_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n=== Round {round_idx} Evaluation Metrics ===\n")
            log_file.write(f"Success rate: {round_stats[round_idx]['success_traj']}/{round_stats[round_idx]['total_traj']} Trajectories ({success_rate:.2f}%).\n")
            log_file.write(f"Total type: {round_stats[round_idx]['total_type']}, Percentage: {type_percentage:.2f}%\n")
            log_file.write(f"Total SR: {round_stats[round_idx]['total_SR']}, Percentage: {sr_percentage:.2f}%\n")
            log_file.write(f"Count: {round_stats[round_idx]['count']}\n")

        print(f"Round {round_idx} evaluation finished. Logs saved to {round_log_path}")
    
    print("Overall evaluation completed.")
    return 0
