import re
from qwen_vl_utils import process_vision_info
import json
from PIL import Image
from zhipuai import ZhipuAI
import base64

def get_action_tars(model, processor, obs):
    prompt = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
            ## Output Format
            ```
            Thought: ...
            Action: ...
            ```
            ## Action Space

            click(point='<point>x1 y1</point>')
            long_press(point='<point>x1 y1</point>')
            type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
            scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
            open_app(app_name=\'\')
            drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
            press_home()
            press_back()
            finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


            ## Note
            - Use English in `Thought` part.
            - Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

            ## User Instruction
            {obs['task']}
            """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": prompt,
                },
                {
                    "type": "image",
                    "image": obs['image_path'],
                }
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    num_input_tokens = inputs.input_ids.shape[-1]
    num_generated_tokens = generated_ids.shape[-1] - inputs.input_ids.shape[-1]
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    #print(output_text[0])
    pattern = r"Action:\s*(.*?)<\|im_end\|>"
    match = re.search(pattern, output_text[0])
    if match:
        action = match.group(1)
        print(action)
    else:
        print("No matching content found.")
    return action, num_generated_tokens, num_input_tokens

def get_action_tars15(model, processor, obs):

    prompt = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
            ## Output Format
            ```
            Thought: ...
            Action: ...
            ```
            ## Action Space

            click(point='<point>x1 y1</point>')
            long_press(point='<point>x1 y1</point>')
            type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
            scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
            open_app(app_name=\'\')
            press_home()
            press_back()
            wait() #Sleep for 5s and take a screenshot to check for any changes.
            finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


            ## Note
            - Use Chinese in `Thought` part.
            - Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

            ## User Instruction
            {obs['task']}
            """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": prompt,
                },
                {
                    "type": "image",
                    "image": obs['image_path'],
                }
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    input_token_count = len(inputs.input_ids[0])  
    generated_token_count = len(generated_ids_trimmed[0])  
    pattern = r"Action:\s*(.*?)<\|im_end\|>"
    match = re.search(pattern, output_text[0])
    if match:
        action = match.group(1)
        print(action)
    else:
        print("No matching content found.")
        action = None

    return action, generated_token_count, input_token_count

def get_action_qwen3vl(model, processor, obs):
    """
    适配 Qwen3-VL 系列模型。
    """
    
    QWEN3_SYSTEM_PROMPT = """

    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>

    # Response format

    Response format for every step:
    1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
    2) Action: a short imperative describing what to do in the UI.
    3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

    Rules:
    - Output exactly in the order: Thought, Action, <tool_call>.
    - Be brief: one sentence for Thought, one for Action.
    - Do not output anything else outside those three parts.
    - If finishing, use action=terminate in the tool call."""

    
    instruction = obs['task']

    history_str = " " 
    
    formatted_user_query = f"The user query: {instruction}.\nTask progress (You have done the following operation on the current device): {history_str}\n"

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": QWEN3_SYSTEM_PROMPT}] 
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": obs['image_path']},
                {"type": "text", "text": formatted_user_query},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, 
        image_patch_size=16, 
        return_video_kwargs=True, 
        return_video_metadata=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_kwargs.get("video_metadata", None),
        padding=True,
        return_tensors="pt",
        do_resize=False 
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=256 
    )

    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    
    # 获取输入 token 数量
    input_token_count = len(inputs.input_ids[0])  
    
    # 获取生成的 token 数量
    generated_token_count = len(generated_ids_trimmed[0])  

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    print(output_text)
    return output_text, generated_token_count, input_token_count
    
def get_action_UIvenus15(model, processor, obs):
    """
    适配 UI-Venus-1.5 系列模型（2B/8B）。
    """

    MOBILE_USER_PROMPT = """
    **You are a GUI Agent.** 
    Your task is to analyze a given user task, review current screenshot and previous actions, and determine the next action to complete the task.

    ### Available Actions
    You may execute one of the following functions:
    - Click(box=(x1,y1))
    - Drag(start=(x1,y1), end=(x2,y2))
    - Scroll(start=(x1,y1), end=(x2,y2))
    - Type(content='')
    - Launch(app='')
    - Wait()
    - Finished(content='')
    - CallUser(content='')
    - LongPress(box=(x1,y1))
    - PressBack()
    - PressHome()
    - PressEnter()
    - PressRecent()

    ### User Task
    {user_task}

    ### Previous Actions
    {previous_actions}

    ### Output Format
    <think> your thinking process </think>
    <action> the next action </action>
    <conclusion> the conclusion about the next action </conclusion>

    ### Instruction
    - Make sure you understand the task goal to avoid wrong actions.
    - Make sure you carefully examine the the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
    - For requests that are questions (or chat messages), remember to use the `CallUser` action to reply to user explicitly before finishing! Then, after you have replied, use the Finished action if the goal is achieved.
    - Consider exploring the screen by using the `scroll` action with different directions to reveal additional content.
    - To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
    - To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.
    """

    full_prompt = MOBILE_USER_PROMPT.format(user_task=obs['task'], previous_actions="None")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image", "image": obs['image_path']},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        do_sample=False,
    )

    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
    ]

    # 获取输入 token 数量
    input_token_count = len(model_inputs.input_ids[0])  
    
    # 获取生成的 token 数量
    generated_token_count = len(generated_ids_trimmed[0])  

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text, generated_token_count, input_token_count

def get_action_maiui(model, processor, obs):
    """
    适配 MAI-UI 系列模型（MAI-UI-2B / MAI-UI-8B）。
    使用提供的 system prompt（grounding agent）格式，返回模型原始输出和 token 数量。
    """

    MAI_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

    ## Output Format
    For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
    ```
    <thinking>
    ...
    </thinking>
    <tool_call>
    {"name": "mobile_use", "arguments": <args-json-object>}
    </tool_call>
    ```

    ## Action Space

    {"action": "click", "coordinate": [x, y]}
    {"action": "long_press", "coordinate": [x, y]}
    {"action": "type", "text": ""}
    {"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is optional. Use the "coordinate" if you want to swipe a specific UI element.
    {"action": "open", "text": "app_name"}
    {"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
    {"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter
    {"action": "wait"}
    {"action": "terminate", "status": "success or fail"}
    {"action": "answer", "text": "xxx"} # Use escape characters \\', \\", and \\n in text part to ensure we can parse the text in normal python string format.


    ## Note
    - Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
    - Available Apps: `["Camera","Chrome","Clock","Contacts","Dialer","Files","Settings","Markor","Tasks","Simple Draw Pro","Simple Gallery Pro","Simple SMS Messenger","Audio Recorder","Pro Expense","Broccoli APP","OSMand","VLC","Joplin","Retro Music","OpenTracks","Simple Calendar Pro"]`.
    You should use the `open` action to open the app as possible as you can, because it is the fast way to open the app.
    - You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags."""

    instruction = obs.get('task', '')

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": MAI_SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": obs['image_path']},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_kwargs.get("video_metadata", None),
        padding=True,
        return_tensors="pt",
        do_resize=False,
    )

    model_device = getattr(model, 'device', None)
    if model_device is None:
        model_device = 'cuda'
    inputs = inputs.to(model_device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]

    input_token_count = len(inputs.input_ids[0])  
    generated_token_count = len(generated_ids_trimmed[0])  

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(output_text)
    return output_text, generated_token_count, input_token_count

def get_action_agentcpm(model, tokenizer, obs):

    RAW_SCHEMA = {
      "type": "object",
      "description": "执行操作并决定当前任务状态",
      "additionalProperties": False,
      "properties": {
        "thought": {
          "type": "string",
          "description": "智能体的思维过程"
        },
        "POINT": {
          "$ref": "#/$defs/Location",
          "description": "点击屏幕上的指定位置"
        },
        "to": {
          "description": "移动，组合手势参数",
          "oneOf": [
            {
              "enum": ["up", "down", "left", "right"],
              "description": "从当前点（POINT）出发，执行滑动手势操作，方向包括向上、向下、向左、向右"
            },
            {
              "$ref": "#/$defs/Location",
              "description": "移动到某个位置"
            }
          ]
        },
        "duration": {
          "type": "integer",
          "description": "动作执行的时间或等待时间，毫秒",
          "minimum": 0,
          "default": 200
        },
        "PRESS": {
          "type": "string",
          "description": "触发特殊按键，HOME为回到主页按钮，BACK为返回按钮，ENTER为回车按钮",
          "enum": ["HOME", "BACK", "ENTER"]
        },
        "TYPE": {
          "type": "string",
          "description": "输入文本"
        },
        "STATUS": {
          "type": "string",
          "description": "当前任务的状态。特殊情况：satisfied，无需操作；impossible，任务无法完成；interrupt，任务中断；need_feedback，需要用户反馈；",
          "enum": ["continue", "finish", "satisfied", "impossible", "interrupt", "need_feedback"],
          "default": "continue"
        }
      },
      "$defs": {
        "Location": {
          "type": "array",
          "description": "坐标为相对于屏幕左上角位原点的相对位置，并且按照宽高比例缩放到0～1000，数组第一个元素为横坐标x，第二个元素为纵坐标y",
          "items": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1000
          },
          "minItems": 2,
          "maxItems": 2
        }
      }
    }

    items = list(RAW_SCHEMA.items())
    insert_index = 3
    items.insert(insert_index, ("required", ["thought"])) 
    ACTION_SCHEMA = dict(items)

    schema_str = json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))
    
    system_prompt = f'''# Role
    你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

    # Task
    针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

    # Rule
    - 以紧凑JSON格式输出
    - 输出操作必须遵循Schema约束

    # Schema
    {schema_str}'''

    image = Image.open(obs['image_path']).convert("RGB")
    
    def __resize__(origin_img):
        w, h = origin_img.size
        max_line = 1120
        if h > max_line:
            w = int(w * max_line / h)
            h = max_line
        if w > max_line:
            h = int(h * max_line / w)
            w = max_line
        return origin_img.resize((w, h), resample=Image.Resampling.LANCZOS)
    
    image = __resize__(image)
    instruction = obs['task']
    msgs = [{
        "role": "user",
        "content": [
            f"<Question>{instruction}</Question>\n当前屏幕截图：",
            image
        ]
    }]
    
    # Calculate input token count
    input_token_count = len(tokenizer.encode(str(msgs), add_special_tokens=False))

    res = model.chat(
        image=None, 
        msgs=msgs,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        temperature=0.1,
        top_p=0.3,
        max_new_tokens=2048
    )
    token_count = len(tokenizer.encode(res, add_special_tokens=False))
    return res, token_count, input_token_count


def get_action_magicgui(model,processor,obs):
    """
    适配 MagicGUIL RFT
    """
    MAGIC_SYSTEM_PROMPT = """你是一个训练有素的手机智能体，能够帮助用户进行单步导航任务。已知当前智能手机的截图<image>，和用户指令{task}请输出正确的函数调用以实现用户指令。除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：
    - UI基础操作：
    1. tap(x: float,y: float) 该函数用于在智能手机屏幕上点击特定点。坐标 x 和 y 表示待点击控件的中心位置。
    2. scroll(x: float,y: float,direction: str) 该函数用于从起始坐标 (x,y) 开始在智能手机屏幕上滑动操作，方向为手指滑动的方向。坐标 x 和 y 表示屏幕上待滑动控件的中心位置。方向可以是 "up"、"down"、"left" 或 "right"。
    3. text(x: float,y: float,text_input: str) 该函数用于在智能手机屏幕上输入指定的text。坐标 x 和 y 表示待点击控件的中心位置。
    - 手机按键操作：
    4. navigate_back() 该函数用于返回智能手机的上一个屏幕。
    5. navigate_home() 该函数用于返回手机的home screen或关闭当前应用。
    - 其他操作：
    6. long_press(x: float,y: float) 该函数用于在智能手机屏幕上的特定点执行长按操作。坐标 x 和 y 表示待点击控件的中心位置。
    7. wait() 该函数表示在当前页面等候。
    8. enter() 该函数表示按下enter键。
    9. take_over(text_input: str) 该函数用于提示用户接管智能手机，其中 text_input 是提示用户接管手机的原因。如果原因不确定，请填写“请您接管当前界面”。
    10. drag(x1: float,y1: float,x2: float,y2: float) 该函数执行一个对起始和终点敏感的拖动操作，表示手指从点1拖到点2。常见的场景包括滑块拖动、滚动选择器拖动和图片裁剪。
    11. screen_shot() 该函数用于截图。
    12. long_screen_shot() 该函数执行长截图。
    13. call_api(api_name: str,params: str) 调用指定的API并传入给定的参数。api_name是API的名称。params包含API所需的输入参数。例如，call_api(Amazon, open)意味着打开亚马逊APP。如果你发现当前指令无法在当前页面上执行，你需要输出no_answer。如果你发现当前指令已完成，你需要输出action_completed。"""
    user_instruction = MAGIC_SYSTEM_PROMPT.format(task=obs['task'])
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": obs['image_path']},
                {"type": "text", "text": user_instruction},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=False  
    )
    
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    
    token_count = len(generated_ids_trimmed[0])
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip(),token_count

def get_action_gelab(model, processor, obs):
    """
    适配 GELab-Zero
    """
    SYSTEM_PROMPT = """你是一个手机 GUI-Agent 操作专家，你需要根据用户下发的任务、手机屏幕截图和交互操作的历史记录，借助既定的动作空间与手机进行交互，从而完成用户的任务。
    请牢记，手机屏幕坐标系以左上角为原点，x轴向右，y轴向下，取值范围均为 0-1000。

    在 Android 手机的场景下，你的动作空间包含以下8类操作，所有输出都必须遵守对应的参数要求：
    1. CLICK：点击手机屏幕坐标，需包含点击的坐标位置 point。
    例如：action:CLICK\tpoint:x,y
    2. TYPE：在手机输入框中输入文字，需包含输入内容 value、输入框的位置 point。
    例如：action:TYPE\tvalue:输入内容\tpoint:x,y
    3. COMPLETE：任务完成后向用户报告结果，需包含报告的内容 value。
    例如：action:COMPLETE\treturn:完成任务后向用户报告的内容
    4. WAIT：等待指定时长，需包含等待时间 value（秒）。
    例如：action:WAIT\tvalue:等待时间
    5. AWAKE：唤醒指定应用，需包含唤醒的应用名称 value。
    例如：action:AWAKE\tvalue:应用名称
    6. INFO：询问用户问题或详细信息，需包含提问内容 value
    例如：action:INFO\tvalue:提问内容
    7. ABORT：终止当前任务，仅在当前任务无法继续执行时使用，需包含 value 说明原因。
    例如：action:ABORT\tvalue:终止任务的原因
    8. SLIDE：在手机屏幕上滑动，滑动的方向不限，需包含起点 point1 和终点 point2。
    例如：action:SLIDE\tpoint1:x1,y1\tpoint2:x2,y2
    9. LONGPRESS：长按手机屏幕坐标，需包含长按的坐标位置 point。
    例如：action:LONGPRESS\tpoint:x,y
    """
    
    def make_gelab_prompt(task, history_summary="暂无历史操作"):
        """构造符合 GELab-Zero 训练格式的 User Prompt"""
        return [
            {
                "type": "text",
                "text": f'''
            已知用户任务为：{task}
            已知已经执行过的历史动作如下：{history_summary}
            当前手机屏幕截图如下：
            '''
            },
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": f'''
            在执行操作之前，请务必回顾你的历史操作记录和限定的动作空间，先进行思考和解释然后输出动作空间和对应的参数：
            1. 思考（THINK）：在 <THINK> 和 </THINK> 标签之间。
            2. 解释（explain）：在动作格式中，使用 explain: 开头，简要说明当前动作的目的和执行方式。
            在执行完操作后，请输出执行完当前步骤后的新历史总结。
            输出格式示例：
            <THINK> 思考的内容 </THINK>
            explain:解释的内容\taction:动作空间和对应的参数\tsummary:执行完当前步骤后的新历史总结
            '''
            }
        ]
    # 1. 构造 System Prompt
    system_message = {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    }

    # 2. 构造 User Prompt
    user_content = make_gelab_prompt(obs['task'])
    user_content[1]['image'] = obs['image_path']

    messages = [
        system_message,
        {
            "role": "user",
            "content": user_content
        }
    ]

    # 3. 数据处理
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, 
        image_patch_size=16, 
        return_video_kwargs=True, 
        return_video_metadata=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_kwargs.get("video_metadata", None),
        padding=True,
        return_tensors="pt",
        do_resize=False 
    ).to(model.device)

    input_token_count = inputs.input_ids.shape[1]

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=2048, 
        do_sample=False    
    )
    
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    token_count = len(generated_ids_trimmed[0])
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text, token_count, input_token_count

def get_action_atlas(model, processor, obs):
    ATLAS_PROMPT =  """
    You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

    1. Basic Actions
    Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
    Basic Action 1: CLICK 
        - purpose: Click at the specified position.
        - format: CLICK <point>[[x-axis, y-axis]]</point>
        - example usage: CLICK <point>[[101, 872]]</point>
        
    Basic Action 2: TYPE
        - purpose: Enter specified text at the designated location.
        - format: TYPE [input text]
        - example usage: TYPE [Shanghai shopping mall]

    Basic Action 3: SCROLL
        - purpose: SCROLL in the specified direction.
        - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
        - example usage: SCROLL [UP]
        
    2. Custom Actions
    Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.
    Custom Action 1: LONG_PRESS 
        - purpose: Long press at the specified position.
        - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
        - example usage: LONG_PRESS <point>[[101, 872]]</point>
        
    Custom Action 2: OPEN_APP
        - purpose: Open the specified application.
        - format: OPEN_APP [app_name]
        - example usage: OPEN_APP [Google Chrome]

    Custom Action 3: PRESS_BACK
        - purpose: Press a back button to navigate to the previous screen.
        - format: PRESS_BACK
        - example usage: PRESS_BACK

    Custom Action 4: PRESS_HOME
        - purpose: Press a home button to navigate to the home page.
        - format: PRESS_HOME
        - example usage: PRESS_HOME

    Custom Action 5: PRESS_RECENT
        - purpose: Press the recent button to view or switch between recently used applications.
        - format: PRESS_RECENT
        - example usage: PRESS_RECENT

    Custom Action 6: ENTER
        - purpose: Press the enter button.
        - format: ENTER
        - example usage: ENTER

    Custom Action 7: WAIT
        - purpose: Wait for the screen to load.
        - format: WAIT
        - example usage: WAIT

    Custom Action 8: COMPLETE
        - purpose: Indicate the task is finished.
        - format: COMPLETE
        - example usage: COMPLETE

    In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
    Thoughts: Clearly outline your reasoning process for current step.
    Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

    Your current task instruction, action history, and associated screenshot are as follows:
    Screenshot: 
    """
    # print(obs['image_path'])
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": ATLAS_PROMPT,
                },
                {
                    "type": "image",
                    "image": obs['image_path'],
                },
                {"type": "text", "text": f"Task instruction: {obs['task']}" },
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    input_token_count = inputs.input_ids.shape[-1]
    num_generated_tokens = generated_ids.shape[-1] - inputs.input_ids.shape[-1]
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    pattern = r"Actions:\n(.*?)<\|im_end\|>"
    match = re.search(pattern, output_text[0], re.DOTALL)
    if match:
        action = match.group(1)
    else:
        pattern = r"actions:\n(.*?)<\|im_end\|>"
        match = re.search(pattern, output_text[0], re.DOTALL)
        if match:
            action = match.group(1)
        else:
            print("No matching content found.")
    # print(output_text[0])
    return action, num_generated_tokens, input_token_count

def get_action_qwen25(model, processor, obs):
    with Image.open(obs['image_path']) as img:
        width, height = img.size 

    prompt = """
    You are a helpful assistant.

    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:

    <tools>
    {
    "type": "function",
    "function": {
        "name_for_human": "mobile_use",
        "name": "mobile_use",
        "description":
        "Use a touchscreen to interact with a mobile device, and take screenshots.\n"
        "* This is an interface to a mobile device with touchscreen. You can perform "
        "actions like clicking, typing, swiping, etc.\n"
        "* Some applications may take time to start or process actions, so you may "
        "need to wait and take successive screenshots to see the results.\n"
        "* The screen's resolution is {{width}}x{{height}}.\n"
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in "
        "the center of the element. Don't click on edges unless asked.",
        "parameters": {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
            "type": "string",
            "description":
                "The action to perform. The available actions are:\n"
                "* `key`: Perform a key event (supports adb keyevent syntax).\n"
                "    - Examples: \"volume_up\", \"volume_down\", \"power\", \"camera\", "
                "\"clear\".\n"
                "* `click`: Click point (x, y).\n"
                "* `long_press`: Press point (x, y) for several seconds.\n"
                "* `swipe`: Swipe from (x, y) to (x2, y2).\n"
                "* `type`: Input text.\n"
                "* `answer`: Output the answer.\n"
                "* `system_button`: Press a system button.\n"
                "* `open`: Open an app.\n"
                "* `wait`: Wait some seconds.\n"
                "* `terminate`: Terminate task with status.",
            "enum": [
                "key", "click", "long_press", "swipe", "type",
                "answer", "system_button", "open", "wait", "terminate"
            ]
            },
            "coordinate": {
            "type": "array",
            "description":
                "(x, y): required by `click`, `long_press`, `swipe`."
            },
            "coordinate2": {
            "type": "array",
            "description":
                "(x, y): required by `swipe` as the ending point."
            },
            "text": {
            "type": "string",
            "description":
                "Required by `key`, `type`, `answer`, and `open`."
            },
            "time": {
            "type": "number",
            "description":
                "Seconds to wait. Required by `long_press` and `wait`."
            },
            "button": {
            "type": "string",
            "enum": ["Back", "Home", "Menu", "Enter"],
            "description":
                "System button type, for `system_button` action."
            },
            "status": {
            "type": "string",
            "enum": ["success", "failure"],
            "description":
                "Used only by `terminate`."
            }
        }
        },
        "args_format":
        "Format the arguments as a JSON object."
    }
    }
    </tools>

    For each function call, return a JSON object with the function name
    and arguments inside <tool_call></tool_call> tags:

    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>
    """

    prompt += f"The user query: {obs['task']}\nBefore answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags."
    
    prompt = prompt.replace("{{width}}", str(width)).replace("{{height}}", str(height))
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt,
                        },
                        {
                            "type": "image",
                            "image": obs['image_path'],
                        },
                    ],
                }
            ]

    # 处理输入并生成
    chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
    inputs = inputs.to("cuda")
    input_token_count = len(inputs.input_ids[0])  # 获取输入的token数量
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
    output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
    #print(output_text[0])
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    token_count = len(generated_ids_trimmed[0])
    match_action = re.search(r'"arguments"\s*:\s*(\{.*?\})', output_text[0], re.DOTALL)
    action = match_action.group(1) if match_action else None    
    return action, token_count, input_token_count  

def get_action_owl(model, processor, obs):
    return get_action_qwen25(model, processor, obs)

def get_action_glm45v(obs):

    prompt = f"""You are a GUI Agent, and your primary task is to respond accurately to user requests or questions. In addition to directly answering the user's queries, you can also use tools or perform GUI operations directly until you fulfill the user's request or provide a correct answer. You should carefully read and understand the images and questions provided by the user, and engage in thinking and reflection when appropriate. The coordinates involved are all represented in thousandths (0-999).

    # Task:
    {obs['task']}

    # Task Platform
    Mobile

    # Action Space
    ### status

    Calling rule: `{{"action_type": "status", "goal_status": "<complete|infeasible>"}}`
    {{
        "name": "status",
        "description": "Finish the task by using the status action with complete or infeasible as goal_status.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "goal_status": {{
                    "type": "string",
                    "description": "The goal status of the task.",
                    "enum": ["complete", "infeasible"]
                }}
            }},
            "required": [
                "goal_status"
            ]
        }}
    }}

    ### answer

    Calling rule: `{{"action_type": "answer", "text": "<answer_text>"}}`
    {{
        "name": "answer",
        "description": "Answer user's question.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "text": {{
                    "type": "string",
                    "description": "The answer text."
                }}
            }},
            "required": [
                "text"
            ]
        }}
    }}

    ### click

    Calling rule: `{{"action_type": "click", "box_2d": [[xmin,ymin,xmax,ymax]]}}`
    {{
        "name": "click",
        "description": "Click/tap on an element on the screen. Use the box_2d to indicate which element you want to click.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "box_2d": {{
                    "type": "array",
                    "description": "The box_2d should be [[xmin,ymin,xmax,ymax]] normalized to 0-999, indicating the position of the element."
                }}
            }},
            "required": [
                "box_2d"
            ]
        }}
    }}

    ### long_press

    Calling rule: `{{"action_type": "long_press", "box_2d": [[xmin,ymin,xmax,ymax]]}}`
    {{
        "name": "long_press",
        "description": "Long press on an element on the screen, similar with the click action above, use the box_2d to indicate which element you want to long press.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "box_2d": {{
                    "type": "array",
                    "description": "The box_2d should be [[xmin,ymin,xmax,ymax]] normalized to 0-999, indicating the position of the element."
                }}
            }},
            "required": [
                "box_2d"
            ]
        }}
    }}

    ### input_text

    Calling rule: `{{"action_type": "input_text", "text": "<text_input>", "box_2d": [[xmin,ymin,xmax,ymax]], "override": true/false}}`
    {{
        "name": "input_text",
        "description": "Type text into a text field (this action contains clicking the text field, typing in the text and pressing the enter). Use the box_2d to indicate the target text field.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "text": {{
                    "description": "The text to be input. Can be from the command, the memory, or the current screen."
                }},
                "box_2d": {{
                    "description": "The box_2d should be [[xmin,ymin,xmax,ymax]] normalized to 0-999, indicating the position of the element."
                }},
                "override": {{
                    "description": "If true, the text field will be cleared before typing. If false, the text will be appended."
                }}
            }},
            "required": [
                "text",
                "box_2d",
                "override"
            ]
        }}
    }}

    ### keyboard_enter

    Calling rule: `{{"action_type": "keyboard_enter"}}`
    {{
        "name": "keyboard_enter",
        "description": "Press the Enter key.",
        "parameters": {{
            "type": "object",
            "properties": {{}},
            "required": []
        }}
    }}

    ### navigate_home

    Calling rule: `{{"action_type": "navigate_home"}}`
    {{
        "name": "navigate_home",
        "description": "Navigate to the home screen.",
        "parameters": {{
            "type": "object",
            "properties": {{}},
            "required": []
        }}
    }}

    ### navigate_back

    Calling rule: `{{"action_type": "navigate_back"}}`
    {{
        "name": "navigate_back",
        "description": "Navigate back.",
        "parameters": {{
            "type": "object",
            "properties": {{}},
            "required": []
        }}
    }}

    ### swipe

    Calling rule: `{{"action_type": "swipe", "direction": "<up|down|left|right>", "box_2d": [[xmin,ymin,xmax,ymax]](optional)}}`
    {{
        "name": "swipe",
        "description": "Swipe the screen or a scrollable UI element in one of the four directions.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "direction": {{
                    "type": "string",
                    "description": "The direction to swipe.",
                    "enum": ["up", "down", "left", "right"]
                }},
                "box_2d": {{
                    "type": "array",
                    "description": "The box_2d to swipe a specific UI element, leave it empty when swiping the whole screen."
                }}
            }},
            "required": [
                "direction"
            ]
        }}
    }}

    ### wait

    Calling rule: `{{"action_type": "wait"}}`
    {{
        "name": "wait",
        "description": "Wait for the screen to update.",
        "parameters": {{
            "type": "object",
            "properties": {{}},
            "required": []
        }}
    }}

    """
    prompt += """# Output Format
    1. Memory: important information you want to remember for the future actions. The memory should be only contents on the screen that will be used in the future actions. It should satisfy that: you cannnot determine one or more future actions without this memory. 
    2. Reason: the reason for the action and the memory. Your reason should include, but not limited to:- the content of the GUI, especially elements that are tightly related to the user goal- the step-by-step thinking process of how you come up with the new action. 
    3. Action: the action you want to take, in the correct JSON format. The action should be one of the above list.

    Your answer should look like:
    Memory: ...
    Reason: ...
    Action: {"action_type":...}
    """
    client = ZhipuAI(
        api_key= "c455aa20cb0a443e88928dd1b8ca040e.umh8ubmc4OGVMEGK"
    )
    with open(obs['image_path'], "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":base64_image
                                },
                        },
                    ],
                }
            ]
    completion = client.chat.completions.create(
        model="GLM-4.5V",
        messages=messages
    )
    chat_response = completion
    answer = chat_response.choices[0].message.content
    answer = answer.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").replace("\n", "")
    pattern = r'\{.*?\}'
    match = re.search(pattern, answer, re.DOTALL)
    if match:
        action = match.group()
    input_tokens = chat_response.usage.prompt_tokens
    output_tokens = chat_response.usage.total_tokens - input_tokens
    # print(action)
    # print(output_tokens)
    return action, output_tokens, input_tokens

