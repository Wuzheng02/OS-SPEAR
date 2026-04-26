import re
from PIL import Image
import json

def transfer_tars2atlas(action):
    if action == "press_home()":
        return "PRESS_HOME"
    elif action == "press_back()":
        return "PRESS_BACK"
    elif action.startswith("finished"):
        return "COMPLETE"
    elif action.startswith("wait"):
        return "WAIT"
    def extract_coordinates(action_str):
        if "start_box=" in action_str:
            match = re.search(r'\((\d+),(\d+)\)', action_str)
            if match:
                return match.group(1), match.group(2)
        elif "point=" in action_str:
            match = re.search(r'point=[\'"]<point>(\d+)\s+(\d+)</point>[\'"]', action_str)
            if match:
                return match.group(1), match.group(2)
            match = re.search(r'point=[\'"](\d+)\s+(\d+)[\'"]', action_str)
            if match:
                return match.group(1), match.group(2)
        return None, None
    if action.startswith("long_press"):
        x, y = extract_coordinates(action)
        if x and y:
            return f"LONG_PRESS <point>[[{x}, {y}]]</point>"
    elif action.startswith("type"):
        content_match = re.search(r"content='([^']*)'", action)
        content = content_match.group(1) if content_match else ""
        return f"TYPE [{content}]"
    elif action.startswith("click"):
        x, y = extract_coordinates(action)
        if x and y:
            return f"CLICK <point>[[{x}, {y}]]</point>"
    elif action.startswith("scroll"):
        x, y = extract_coordinates(action)
        dir_match = re.search(r"direction='(up|down|left|right)'", action, re.IGNORECASE)
        direction = dir_match.group(1).upper() if dir_match else "UP"
        return f"SCROLL [{direction}]"
    elif action.startswith("open_app"):
        app_match = re.search(r"app_name=['\"](.*?)['\"]", action)
        app_name = app_match.group(1) if app_match else "App"
        return f"OPEN_APP [{app_name}]"
    return action

def transfer_tars15toatlas(action, image_path):
    if action == "press_home()":
        return "PRESS_HOME"
    elif action == "press_back()":
        return "PRESS_BACK"
    elif action.startswith("finished"):
        return "COMPLETE"
    elif action.startswith("wait"):
        return "WAIT"
        
    def extract_coordinates(action_str):
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except:
            width, height = 1000, 1000  # Default values if image can't be read
        
        if "point=" in action_str:
            match = re.search(r'\((\d+),(\d+)\)', action_str)
            if match:
                x = int((int(match.group(1)) / width) * 1000)
                y = int((int(match.group(2)) / height) * 1000)
                return str(x), str(y)
        elif "start_box=" in action_str:
            match = re.search(r'\((\d+),(\d+)\)', action_str)
            if match:
                x = int((int(match.group(1)) / width) * 1000)
                y = int((int(match.group(2)) / height) * 1000)
                return str(x), str(y)        
        elif "point=" in action_str:
            match = re.search(r'point=[\'"]<point>(\d+)\s+(\d+)</point>[\'"]', action_str)
            if match:
                x = int((int(match.group(1)) / width) * 1000)
                y = int((int(match.group(2)) / height) * 1000)
                return str(x), str(y)
            match = re.search(r'point=[\'"](\d+)\s+(\d+)[\'"]', action_str)
            if match:
                x = int((int(match.group(1)) / width) * 1000)
                y = int((int(match.group(2)) / height) * 1000)
                return str(x), str(y)
        return None, None
        
    if action.startswith("long_press"):
        x, y = extract_coordinates(action)
        if x and y:
            return f"LONG_PRESS <point>[[{x}, {y}]]</point>"
    elif action.startswith("type"):
        content_match = re.search(r"content='([^']*)'", action)
        content = content_match.group(1) if content_match else ""
        return f"TYPE [{content}]"
    elif action.startswith("click"):
        x, y = extract_coordinates(action)
        if x and y:
            return f"CLICK <point>[[{x}, {y}]]</point>"
    elif action.startswith("scroll"):
        x, y = extract_coordinates(action)
        dir_match = re.search(r"direction='(up|down|left|right)'", action, re.IGNORECASE)
        direction = dir_match.group(1).upper() if dir_match else "UP"
        if x and y:
            return f"SCROLL [{direction}]"
    elif action.startswith("open_app"):
        app_match = re.search(r"app_name=['\"](.*?)['\"]", action)
        app_name = app_match.group(1) if app_match else "App"
        return f"OPEN_APP [{app_name}]"
    return action

def transfer_qwen3vl2atlas(action_str, image_path):
    """
    将Qwen3-VL 的动作输出转换为 Atlas 格式。

    Schema:
      - click: coordinate=[x,y]
      - swipe: coordinate=[x1,y1], coordinate2=[x2,y2]
      - type: text="..."
      - system_button: button="Back/Home/..."

    映射规则：
    click  → CLICK
    long_press → LONG_PRESS
    swipe → SCROLL
    type → TYPE
    system_button(Home/Back/Menu/Enter) → PRESS_HOME / PRESS_BACK / PRESS_RECENT / ENTER
    wait → WAIT
    terminate → COMPLETE
    """
    action_str = action_str.strip()
    
    # 1. 提取 <tool_call> 内容
    tool_match = re.search(r"<tool_call>(.*?)(?:</tool_call>|<tool_call>)", action_str, re.DOTALL)

    if not tool_match:
        print(f"No tool_call found in output.")
        return "WAIT"

    json_str = tool_match.group(1).strip()
    
    try:
        tool_data = json.loads(json_str)
        
        if tool_data.get("name") == "mobile_use":
            args = tool_data.get("arguments", {})
            action_type = args.get("action")
            
            # --- 1. Click / Long Press ---
            if action_type in ["click", "long_press"]:
                coords = args.get("coordinate", [])
                if len(coords) == 2:
                    x, y = coords
                    act_name = "CLICK" if action_type == "click" else "LONG_PRESS"
                    return f"{act_name} <point>[[{x}, {y}]]</point>"
            
            # --- 2. Swipe ---
            # 官方输出: "action": "swipe", "coordinate": [499, 749], "coordinate2": [499, 249]
            elif action_type == "swipe":
                coord1 = args.get("coordinate", [])
                coord2 = args.get("coordinate2", [])
                
                if len(coord1) == 2 and len(coord2) == 2:
                    x1, y1 = coord1
                    x2, y2 = coord2
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    
                    # 根据矢量差计算滑动方向
                    if abs(dx) > abs(dy):

                        direction = "LEFT" if dx > 0 else "RIGHT" 
                    else:
                        direction = "DOWN" if dy < 0 else "UP"
                    
                    return f"SCROLL [{direction}]"
            
            # --- 3. Type ---
            elif action_type == "type":
                text = args.get("text", "")
                return f"TYPE [{text}]"
            
            # --- 4. System Button---
            elif action_type == "system_button":
                btn = args.get("button", "").lower()
                if "home" in btn: return "PRESS_HOME"
                if "back" in btn: return "PRESS_BACK"
                if "enter" in btn: return "ENTER"
                if "menu" in btn: return "PRESS_RECENT"
            
            # --- 5. Terminate ---
            elif action_type == "terminate":
                return "COMPLETE"
                
            # --- 6. Wait ---
            elif action_type == "wait":
                return "WAIT"

    except json.JSONDecodeError:
        print(f"JSON Parse Error: {json_str}")
        return "WAIT"
        
    return "WAIT"

def transfer_venus2atlas(action_str,image_path):
    """
    将 UI-Venus 的动作输出转换为 Atlas 格式。

    Schema:
      - Click(box=(500, 500))
      - Type(content='hello')
      - Scroll(start=(500, 800), end=(500, 200), direction='down')

    映射规则：
    Click → CLICK 
    LongPress → LONG_PRESS
    Drag → SCROLL（方向由 start/end 推断）
    Type → TYPE
    Scroll → SCROLL
    Launch → OPEN_APP
    Press(Back/Home/Enter/Recent) → PRESS_(Back/Home/Enter/Recent)
    Finished / CallUser → COMPLETE
    Wait → WAIT
    """
    
    def _extract_action_text(raw_text):
        match = re.search(r"<action>(.*?)</action>", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return raw_text.strip()

    def _direction_from_points(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        return "DOWN" if dy > 0 else "UP"

    parsed = _extract_action_text(action_str)

    # 1. 提取 Click / LongPress
    # 格式: Click(box=(x, y)) 或 LongPress(box=(x, y))
    coord_pattern = r"(Click|LongPress).*?\((\d+)\s*,\s*(\d+)\)"
    match = re.search(coord_pattern, parsed, re.IGNORECASE)
    
    if match:
        act_name = match.group(1).lower()
        x, y = match.group(2), match.group(3)
        
        act_type = "CLICK"
        if "long" in act_name:
            act_type = "LONG_PRESS"
            
        return f"{act_type} <point>[[{x}, {y}]]</point>"

    # 2. 提取 Type
    # 格式: Type(content='text')
    if "Type" in parsed:
        content_match = re.search(r"Type.*?content=['\"](.*?)['\"]", parsed, re.IGNORECASE)
        content = content_match.group(1) if content_match else ""
        return f"TYPE [{content}]"

    # 3. 提取 Scroll
    # 格式: Scroll(start=(x1, y1), end=(x2, y2), direction='down')
    if "Scroll" in parsed:
        dir_match = re.search(r"direction=['\"](up|down|left|right)['\"]", parsed, re.IGNORECASE)
        if dir_match:
            return f"SCROLL [{dir_match.group(1).upper()}]"
        coord_match = re.search(r"start=\((\d+)\s*,\s*(\d+)\).*?end=\((\d+)\s*,\s*(\d+)\)", parsed, re.IGNORECASE)
        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            return f"SCROLL [{_direction_from_points(x1, y1, x2, y2)}]"
        return "SCROLL [DOWN]"

    # 4. 提取 Drag
    # 格式: Drag(start=(x1, y1), end=(x2, y2))
    if "Drag" in parsed:
        drag_match = re.search(r"start=\((\d+)\s*,\s*(\d+)\).*?end=\((\d+)\s*,\s*(\d+)\)", parsed, re.IGNORECASE)
        if drag_match:
            x1, y1, x2, y2 = map(int, drag_match.groups())
            return f"SCROLL [{_direction_from_points(x1, y1, x2, y2)}]"

    # 4. 系统按键
    if "PressHome" in parsed: return "PRESS_HOME"
    if "PressBack" in parsed: return "PRESS_BACK"
    if "PressRecent" in parsed: return "PRESS_RECENT"
    if "PressEnter" in parsed: return "ENTER"
    if "Finished" in parsed: return "COMPLETE"
    if "CallUser" in parsed: return "COMPLETE"
    if "Wait" in parsed: return "WAIT"
    
    # 5.Launch App -> Open App
    if "Launch" in parsed:
        app_match = re.search(r"app=['\"](.*?)['\"]", parsed)
        app_name = app_match.group(1) if app_match else "App"
        return f"OPEN_APP [{app_name}]"

    return "WAIT"

def transfer_venus15toatlas(action_str, image_path):
    """
    将 UI-Venus-1.5 的动作输出转换为 Atlas 格式。
    Schema/映射规则:
    同 UI-Venus-Navi
    """

    def _extract_action_text(raw_text):
        match = re.search(r"<action>(.*?)</action>", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return raw_text.strip()

    def _to_int(v):
        try:
            return int(round(float(v)))
        except Exception:
            return None

    def _parse_coord(coord_text):
        if not coord_text:
            return None, None
        match = re.search(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", coord_text)
        if not match:
            return None, None
        x = _to_int(match.group(1))
        y = _to_int(match.group(2))
        return x, y

    def _direction_from_points(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        return "DOWN" if dy > 0 else "UP"

    parsed = _extract_action_text(action_str)
    parsed = parsed.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    # Click(box=(x, y))
    click_match = re.search(r"Click\s*\(\s*box\s*=\s*(\([^)]*\))\s*\)", parsed, re.IGNORECASE)
    if click_match:
        x, y = _parse_coord(click_match.group(1))
        if x is not None and y is not None:
            return f"CLICK <point>[[{x}, {y}]]</point>"

    # LongPress(box=(x, y))
    long_match = re.search(r"LongPress\s*\(\s*box\s*=\s*(\([^)]*\))\s*\)", parsed, re.IGNORECASE)
    if long_match:
        x, y = _parse_coord(long_match.group(1))
        if x is not None and y is not None:
            return f"LONG_PRESS <point>[[{x}, {y}]]</point>"

    # Drag(start=(x1, y1), end=(x2, y2)) -> SCROLL
    drag_match = re.search(
        r"Drag\s*\(\s*start\s*=\s*(\([^)]*\))\s*,\s*end\s*=\s*(\([^)]*\))\s*\)",
        parsed,
        re.IGNORECASE,
    )
    if drag_match:
        x1, y1 = _parse_coord(drag_match.group(1))
        x2, y2 = _parse_coord(drag_match.group(2))
        if None not in (x1, y1, x2, y2):
            return f"SCROLL [{_direction_from_points(x1, y1, x2, y2)}]"

    # Scroll(start=(x1, y1), end=(x2, y2), direction='...')
    scroll_match = re.search(
        r"Scroll\s*\(\s*start\s*=\s*(\([^)]*\))\s*,\s*end\s*=\s*(\([^)]*\))(?:\s*,\s*direction\s*=\s*['\"]?\s*([a-zA-Z]+)\s*['\"]?)?\s*\)",
        parsed,
        re.IGNORECASE,
    )
    if scroll_match:
        direction = scroll_match.group(3)
        if direction:
            return f"SCROLL [{direction.upper()}]"

        x1, y1 = _parse_coord(scroll_match.group(1))
        x2, y2 = _parse_coord(scroll_match.group(2))
        if None not in (x1, y1, x2, y2):
            return f"SCROLL [{_direction_from_points(x1, y1, x2, y2)}]"
        return "SCROLL [DOWN]"

    # Type(content='...')
    type_match = re.search(r"Type\s*\(\s*content\s*=\s*['\"](.*?)['\"]\s*\)", parsed, re.IGNORECASE | re.DOTALL)
    if type_match:
        return f"TYPE [{type_match.group(1)}]"

    # Launch(app='...')
    launch_match = re.search(r"Launch\s*\(\s*app\s*=\s*['\"](.*?)['\"]\s*\)", parsed, re.IGNORECASE)
    if launch_match:
        app_name = launch_match.group(1).strip() if launch_match.group(1) else "App"
        return f"OPEN_APP [{app_name}]"

    # 纯按键类动作
    if re.search(r"\bPressHome\s*\(\s*\)", parsed, re.IGNORECASE):
        return "PRESS_HOME"
    if re.search(r"\bPressBack\s*\(\s*\)", parsed, re.IGNORECASE):
        return "PRESS_BACK"
    if re.search(r"\bPressRecent\s*\(\s*\)", parsed, re.IGNORECASE):
        return "PRESS_RECENT"
    if re.search(r"\bPressEnter\s*\(\s*\)", parsed, re.IGNORECASE):
        return "ENTER"
    if re.search(r"\bWait\s*\(\s*\)", parsed, re.IGNORECASE):
        return "WAIT"

    # Finished/CallUser 在 Atlas 中都收敛为 COMPLETE
    if re.search(r"\bFinished\s*\(", parsed, re.IGNORECASE):
        return "COMPLETE"
    if re.search(r"\bCallUser\s*\(", parsed, re.IGNORECASE):
        return "COMPLETE"

    return "WAIT"

def transfer_maiui2atlas(action, image_path):
    """
    将 MAI-UI 的动作输出转换为 Atlas 格式。
    
    Schema：
        - {"action": "click", "coordinate": [x, y]}
        - {"action": "long_press", "coordinate": [x, y]}
        - {"action": "type", "text": ""}

    映射规则：
    click / long_press → CLICK / LONG_PRESS
    type → TYPE
    swipe → SCROLL
    drag → SCROLL
    open → OPEN_APP
    system_button → PRESS_*
    wait → WAIT
    terminate / answer / ask_user / mcp_call → COMPLETE
    """

    with Image.open(image_path) as img:
        width, height = img.size
    match_action = re.search(r'"action"\s*:\s*"(\w+)"', action)
    if not match_action:
        return None
    action_type = match_action.group(1)
    if action_type == "click" or "left_click":
        match_coord = re.search(r'"coordinate"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', action)
        if match_coord:
            x, y = int(int(match_coord.group(1))), int(int(match_coord.group(2)))
            return f"CLICK <point>[[{x},{y}]]</point>"
    elif action_type == "long_press":
        match_coord = re.search(r'"coordinate"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', action)
        if match_coord:
            x, y = int(int(match_coord.group(1))), int(int(match_coord.group(2)))
            return f"LONG_PRESS <point>[[{x},{y}]]</point>"
    elif action_type == "swipe":
        match_coords = re.search(
            r'"coordinate"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*"coordinate2"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            action
        )
        if match_coords:
            x1, y1 = int(match_coords.group(1)), int(match_coords.group(2))
            x2, y2 = int(match_coords.group(3)), int(match_coords.group(4))
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) > abs(dy):
                direction = "LEFT" if dx > 0 else "RIGHT" 
            else:
                direction = "DOWN" if dy < 0 else "UP" 
            return f"SCROLL [{direction}]"
    elif action_type == "type":
        match_text = re.search(r'"text"\s*:\s*"([^"]+)"', action)
        if match_text:
            text = match_text.group(1)
            return f"TYPE [{text}]"
    elif action_type == "system_button":
        match_button = re.search(r'"button"\s*:\s*"([^"]+)"', action)
        if match_button:
            button = match_button.group(1)
            if button == "Back":
                return "PRESS_BACK"
            elif button == "Home":
                return "PRESS_HOME"
            elif button == "Enter":
                return "ENTER"
    elif action_type == "wait":
        return "WAIT"
    elif action_type == "terminate":
        match_status = re.search(r'"status"\s*:\s*"([^"]+)"', action)
        if match_status:
            status = match_status.group(1)
            if status == "success":
                return "COMPLETE"
    return "WAIT"

    # terminate / answer / ask_user / mcp_call
    if action_type in ["terminate", "answer", "ask_user", "mcp_call"]:
        return "COMPLETE"

    return "WAIT"

def transfer_cpm2atlas(action_str, image_path):
    """
    将 AgentCPM-GUI 的 JSON 输出转换为 Atlas 格式。

    Schema: '{"thought":"...", "POINT":[729,69]}'

    映射规则：
    POINT → CLICK
    duration>500 + POINT → LONG_PRESS
    POINT + to → SCROLL
    TYPE → TYPE
    PRESS（HOME/BACK/ENTER） → PRESS_(Home/Back/Enter)
    STATUS=finish/satisfied/impossible → COMPLETE
    STATUS=interrupt/need_feedback/continue → WAIT
    duration without POINT → WAIT
    """
    action_str = action_str.strip()
    
    try:
        data = json.loads(action_str)
        
        # 1. Click (默认只有 POINT 且没有 duration 或者是 duration 很短)
        # AgentCPM 的 Click 只有 {"POINT": [x,y]}，没有 type 字段说明它是 Click
        # 检查是否包含 POINT
        if "POINT" in data:
            x, y = data["POINT"]
            
            # 区分 Click, Long Press, Swipe
            if "to" in data: 
                # Swipe: {"POINT": [x,y], "to": "down"}
                direction = data["to"]
                if isinstance(direction, list): 
                     return "SCROLL [DOWN]" 
                return f"SCROLL [{direction.upper()}]"
            
            elif "duration" in data and data["duration"] > 500:
                # Long Press: {"POINT": [x,y], "duration": 1000}
                return f"LONG_PRESS <point>[[{x}, {y}]]</point>"
            
            else:
                # Click: {"POINT": [x,y]}
                return f"CLICK <point>[[{x}, {y}]]</point>"

        # 2. Type Text
        if "TYPE" in data:
            return f"TYPE [{data['TYPE']}]"

        # 3. Press Key
        if "PRESS" in data:
            key = data["PRESS"].upper()
            if key == "HOME": return "PRESS_HOME"
            if key == "BACK": return "PRESS_BACK"
            if key == "ENTER": return "ENTER"
            return "PRESS_HOME" 

        # 4. Status (Finish)
        if "STATUS" in data:
            status = data["STATUS"].lower()
            if status in ["finish", "satisfied", "impossible"]:
                return "COMPLETE"
                
        # 5. Wait
        if "duration" in data and "POINT" not in data:
            return "WAIT"

    except json.JSONDecodeError:
        print(f"AgentCPM JSON Parse Error: {action_str}")
        return "WAIT"
        
    return "WAIT"

def transfer_magicgui2atlas(action_str, image_path):
    """
    将 MagicGUI_RFT 的动作输出转换为 Atlas 格式。
    
    Schema: 
        - scroll(x: float,y: float,direction: str)
        - text(x: float,y: float,text_input: str) 

    映射规则：
    tap → CLICK
    scroll → SCROLL
    drag → SCROLL
    text → TYPE
    long_press → LONG_PRESS
    navigate_back/home → PRESS_BACK / PRESS_HOME
    enter → ENTER
    wait → WAIT
    action_completed → COMPLETE
    call_api → OPEN_APP
    screen_shot / long_screen_shot /take_over → WAIT
    """
    action_str = action_str.strip().replace("{", "").replace("}", "").replace('"', '')

    # 1. Tap -> CLICK
    if "tap" in action_str:
        match = re.search(r"tap\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", action_str)
        if match:
            x, y = match.group(1), match.group(2)
            return f"CLICK <point>[[{x}, {y}]]</point>"

    # 2. Scroll -> SCROLL
    if "scroll" in action_str:
        dir_match = re.search(r"scroll.*?(up|down|left|right)", action_str, re.IGNORECASE)
        direction = dir_match.group(1).upper() if dir_match else "DOWN"
        return f"SCROLL [{direction}]"

    # 3. Drag -> SCROLL
    if "drag" in action_str:
        dir_match = re.search(r"drag.*?(up|down|left|right)", action_str, re.IGNORECASE)
        direction = dir_match.group(1).upper() if dir_match else "DOWN"
        return f"SCROLL [{direction}]"

    # 4. Text -> TYPE (text_input)
    if "text_input" in action_str:
        content_match = re.search(r"text_input\s*:\s*['\"]?(.*?)['\"]?(?:\)|,|$)", action_str, re.IGNORECASE)
        if not content_match:
            content_match = re.search(r"text\s*\(.*?,\s*(.*)\s*\)", action_str)
        content = content_match.group(1) if content_match else ""
        return f"TYPE [{content}]"

    # 5. Long Press -> LONG_PRESS
    if "long_press" in action_str:
        match = re.search(r"long_press\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", action_str)
        if match:
            x, y = match.group(1), match.group(2)
            return f"LONG_PRESS <point>[[{x}, {y}]]</point>"

    # 6. System Functions
    if "navigate_home" in action_str: return "PRESS_HOME"
    if "navigate_back" in action_str: return "PRESS_BACK"
    if "enter" in action_str: return "ENTER"
    if "wait" in action_str: return "WAIT"
    if "action_completed" in action_str: return "COMPLETE"

    # 7. screen_shot / long_screen_shot / take_over -> WAIT
    if "screen_shot" in action_str or "long_screen_shot" in action_str or "take_over" in action_str:
        return "WAIT"
    
    if "call_api" in action_str:
        match = re.search(r"call_api\s*\(\s*(.*?)\s*,", action_str)
        app_name = match.group(1) if match else "App"
        return f"OPEN_APP [{app_name}]"

    return "WAIT"

def transfer_gelab2atlas(action_str, image_path):
    """
    将 GELab-Zero 的动作输出转换为 Atlas 格式。
    
    Schema:
    <THINK>...</THINK>
    explain:...\taction:CLICK\tpoint:100,200
    
    映射规则：
    CLICK → CLICK
    LONGPRESS → LONG_PRESS
    SLIDE → SCROLL（方向推断）
    TYPE → TYPE
    WAIT → WAIT
    AWAKE → OPEN_APP
    COMPLETE → COMPLETE
    ABORT/INFO → WAIT

    """
    action_str = action_str.strip()
    
    # 1. 提取 <THINK> 之后的内容 (即 key-value 动作部分)
    if "</THINK>" in action_str:
        kv_part = action_str.split("</THINK>")[1].strip()
    else:
        kv_part = action_str # 兜底：如果没有 THINK 标签，直接解析全文

    # 2. 解析键值对
    params = {}
    
    segments = kv_part.split('\t')
    if len(segments) < 2:
        pass

    for seg in segments:
        if ":" in seg:
            k, v = seg.split(":", 1)
            params[k.strip()] = v.strip()

    # 3. 映射到 Atlas 格式
    action_type = params.get("action", "").upper()
    
    if action_type == "CLICK":
        point = params.get("point", "")
        if "," in point:
            x, y = point.split(",")[:2]
            return f"CLICK <point>[[{x}, {y}]]</point>"
            
    elif action_type == "LONGPRESS":
        point = params.get("point", "")
        if "," in point:
            x, y = point.split(",")[:2]
            return f"LONG_PRESS <point>[[{x}, {y}]]</point>"
            
    elif action_type == "SLIDE":
        p1 = params.get("point1", "")
        p2 = params.get("point2", "")
        if "," in p1 and "," in p2:
            x1, y1 = map(int, p1.split(","))
            x2, y2 = map(int, p2.split(","))
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) > abs(dy):
                direction = "RIGHT" if dx > 0 else "LEFT"
            else:
                direction = "DOWN" if dy > 0 else "UP"
            return f"SCROLL [{direction}]"
            
    elif action_type == "TYPE":
        content = params.get("value", "")
        return f"TYPE [{content}]"
        
    elif action_type == "WAIT":
        return "WAIT"

    elif action_type == "AWAKE":
        app_name = params.get("value", "App")
        return f"OPEN_APP [{app_name}]"
        
    elif action_type == "COMPLETE":
        return "COMPLETE"
        
    elif action_type == "ABORT":
        return "WAIT" 

    elif action_type == "INFO":
        return "WAIT"

    if "CLICK" in kv_part:
        match = re.search(r"point[:\s]*(\d+)[,\s]+(\d+)", kv_part)
        if match:
            return f"CLICK <point>[[{match.group(1)}, {match.group(2)}]]</point>"
    return "WAIT"

def transfer_qwen25toatlas(action, image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    match_action = re.search(r'"action"\s*:\s*"(\w+)"', action)
    if not match_action:
        return None
    action_type = match_action.group(1)
    if action_type == "click" or "left_click":
        match_coord = re.search(r'"coordinate"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', action)
        if match_coord:
            x, y = int(int(match_coord.group(1))*1000/width), int(int(match_coord.group(2))*1000/height)
            return f"CLICK <point>[[{x},{y}]]</point>"
    elif action_type == "long_press":
        match_coord = re.search(r'"coordinate"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', action)
        if match_coord:
            x, y = int(int(match_coord.group(1))*1000/width), int(int(match_coord.group(2))*1000/height)
            return f"LONG_PRESS <point>[[{x},{y}]]</point>"
    elif action_type == "swipe":
        match_coords = re.search(
            r'"coordinate"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*"coordinate2"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            action
        )
        if match_coords:
            x1, y1 = int(match_coords.group(1)), int(match_coords.group(2))
            x2, y2 = int(match_coords.group(3)), int(match_coords.group(4))
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) > abs(dy):
                direction = "LEFT" if dx > 0 else "RIGHT" 
            else:
                direction = "DOWN" if dy < 0 else "UP" 
            return f"SCROLL [{direction}]"
    elif action_type == "type":
        match_text = re.search(r'"text"\s*:\s*"([^"]+)"', action)
        if match_text:
            text = match_text.group(1)
            return f"TYPE [{text}]"
    elif action_type == "system_button":
        match_button = re.search(r'"button"\s*:\s*"([^"]+)"', action)
        if match_button:
            button = match_button.group(1)
            if button == "Back":
                return "PRESS_BACK"
            elif button == "Home":
                return "PRESS_HOME"
            elif button == "Enter":
                return "ENTER"
    elif action_type == "wait":
        return "WAIT"
    elif action_type == "terminate":
        match_status = re.search(r'"status"\s*:\s*"([^"]+)"', action)
        if match_status:
            status = match_status.group(1)
            if status == "success":
                return "COMPLETE"
    return "WAIT"

def transfer_owl2atlas(action, image_path):
    return transfer_qwen25toatlas(action, image_path)

def transfer_glms45vtoatlas(action, image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    try:
        action_json = json.loads(action)
        action_type = action_json.get('action_type')
        
        if action_type == 'click':
            box_2d = action_json.get('box_2d', [[]])[0]
            if len(box_2d) == 4:
                x = int((box_2d[0] + box_2d[2]) / 2)
                y = int((box_2d[1] + box_2d[3]) / 2)
                return f"CLICK <point>[[{x},{y}]]</point>"
            else:
                print(f"警告: 无效的box_2d格式: {box_2d}")
                return "WAIT"
                
        elif action_type == 'swipe':
            direction = action_json.get('direction', '').upper()
            return f"SCROLL [{direction}]"
            
        elif action_type == 'answer':
            return "COMPLETE"
            
        elif action_type == 'status':
            if action_json.get('goal_status') == 'complete':
                return "COMPLETE"
            else:
                return "WAIT"
                
        elif action_type == 'navigate_back':
            return "PRESS_BACK"
            
        elif action_type == 'wait':
            return "WAIT"
            
        elif action_type == 'input_text':
            text = action_json.get('text', '')
            return f"TYPE [{text}]"
            
        else:
            print(f"未知的action_type: {action_type}")
            return "WAIT"
            
    except json.JSONDecodeError:
        print(f"JSON解析错误: {json_str}")
        return "WAIT"
    return "WAIT"