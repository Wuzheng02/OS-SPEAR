
# OS-SPEAR

**A Toolkit for the Safety, Performance, Efficiency, and Robustness Analysis of OS Agents**

![OS-SPEAR Framework](https://github.com/Wuzheng02/OS-SPEAR/blob/main/main.png)
---

## 📖 Overview

**OS-SPEAR** is a comprehensive evaluation toolkit for **OS Agents**, designed to systematically assess their capabilities across four critical dimensions:

* **Safety**
* **Performance**
* **Efficiency**
* **Robustness**

The toolkit includes **22 popular OS agents** and provides standardized benchmarks for fair and reproducible evaluation.

---

## 📊 Evaluation Benchmarks

OS-SPEAR consists of three core subsets:

| Dimension   | Subset                  |
| ----------- | ----------------------- |
| Safety      | `S-subset`              |
| Performance | `P-subset`              |
| Robustness  | `R-subset`              |
| Efficiency  | Derived from `P-subset` |

---

## 🤖 Supported Models

OS-SPEAR evaluates two categories of models:

### 1. Specialized OS Agents

* UI-Venus-1.5 Series
* UI-TARS Series
* UI-TARS-1.5
* GUI-owl Series
* AgentCPM-GUI
* GELab-Zero
* MAI-UI Series
* OS-Atlas Series

### 2. General Multimodal Foundation Models (with OS capabilities)

* GLM-4.5V
* Qwen2.5-VL Series
* Qwen3-VL Series

---

## 📦 Dataset Preparation

Download the dataset from Hugging Face:

👉 [https://huggingface.co/datasets/wuuuuuz/OS-SPEAR/tree/main](https://huggingface.co/datasets/wuuuuuz/OS-SPEAR/tree/main)

After downloading:

1. Extract all files
2. Place the following folders in the **same directory as `README.md`**:

```
S-subset/
P-subset/
R-subset/
```

---

## 🚀 Usage

### 1. Evaluate Existing OS Agents

#### Step 1: Configure

Edit `eval/config.yaml`:

```yaml
MODEL: <model_name>          # choose from supported models
MODEL_PATH: <absolute_path_to_model>
DATA_PATH: <absolute_path_to_OS-SPEAR>
LOG_PATH: <path_to_save_logs>

TEST_S: true   # Safety
TEST_P: true   # Performance
TEST_R: true   # Robustness
```

#### Step 2: Run Evaluation

```bash
cd eval
python run.py
```

---

### 2. Add and Evaluate New OS Agents

To evaluate a new model, extend the framework as follows:

#### Step 2.1: Implement Action Function

Modify:

```
eval/get_action.py
```

Add:

```python
def get_action_<model_name>(...):
    ...
```

---

#### Step 2.2: Register in Test Loop

Modify:

```
eval/test_loop.py
```

Add:

```python
elif MODEL == "<model_name>":
    # load model and processor
```

---

#### Step 2.3: Register in Runner

Modify:

```
eval/run.py
```

Add:

```python
elif MODEL == "<model_name>":
    # load model and processor
```

---

#### Step 2.4: Run Evaluation

Same as before:

```bash
python run.py
```

---

## 📈 Generate Evaluation Report

After evaluation, generate the final report:

```bash
python report/evaluate.py
```
