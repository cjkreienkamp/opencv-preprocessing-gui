import streamlit as st
import cv2
from PIL import Image
import numpy as np



class PreprocessingStep:
    
    def __init__(self, name: str):
        
        transformation_params = {
            # name: [param1['value'], param1_widget, param2['value'], param2_widget]
            "Resize":       ["Height", "number", "Width", "number"],
            "Normalize":    ["Upper and Lower Values", "normalize_slider", None, None],
            "Grayscale":    [None, None, None, None],
            "Threshold":    ["Below → 0, Above → 255", "threshold_slider", "Invert", "checkbox"],
            "Dilate":       ["Kernel (2n+1)", "slider", "Iterations", "slider"],
            "Erode":        ["Kernel (2n+1)", "slider", "Iterations", "slider"],
            "Laplacian":    ["Kernel (2n+1)", "slider", None, None],
            "Mask":         ["Direction", "select_slider", "Percentage %", "percent_slider"]
        }
        
        self.idx = None
        self.name = name
        self.param1 = {
            "label": transformation_params[name][0],
            "widget": transformation_params[name][1],
            "value": None
        }
        self.param2 = {
            "label": transformation_params[name][2],
            "widget": transformation_params[name][3],
            "value": None
        }
        self.visible = False
        self.python_code_str = ""

    def apply_step(self, img, step_number: int):
        src = f"out{step_number-1}"
        dst = f"out{step_number}"
        if self.name == "Resize":
            self.python_code_str = f"{dst} = cv2.resize({src}, dsize=({self.param2['value']},{self.param1['value']}))"
            return cv2.resize(img, dsize=(self.param2['value'],self.param1['value']))
        elif self.name == "Normalize":
            self.python_code_str = f"{dst} = cv2.normalize({src}, None, alpha={self.param1['value']}, beta={self.param2['value']}, norm_type=cv2.NORM_MINMAX)"
            return cv2.normalize(img, None, alpha=self.param1['value'], beta=self.param2['value'], norm_type=cv2.NORM_MINMAX)
        elif self.name == "Grayscale":
            self.python_code_str = f"{dst} = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY)"
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.name == "Threshold":
            if self.param2['value']:
                self.python_code_str = f"_,{dst} = cv2.threshold({src}, {self.param1['value']}, 255, cv2.THRESH_BINARY_INV)"
                return cv2.threshold(img, self.param1['value'], 255, cv2.THRESH_BINARY_INV)[1]
            else:
                self.python_code_str = f"_,{dst} = cv2.threshold({src}, {self.param1['value']}, 255, cv2.THRESH_BINARY)"
                return cv2.threshold(img, self.param1['value'], 255, cv2.THRESH_BINARY)[1]
        elif self.name == "Dilate":
            ksize = self.param1['value']*2 + 1
            self.python_code_str = f"{dst} = cv2.dilate({src}, np.ones(({ksize},{ksize}), np.uint8), iterations={self.param2['value']})"
            return cv2.dilate(img, np.ones((ksize,ksize), np.uint8), iterations=self.param2['value'])
        elif self.name == "Erode":
            ksize = self.param1['value']*2 + 1
            self.python_code_str = f"{dst} = cv2.erode({src}, np.ones(({ksize},{ksize}), np.uint8), iterations={self.param2['value']})"
            return cv2.erode(img, np.ones((ksize,ksize), np.uint8), iterations=self.param2['value'])
        elif self.name == "Laplacian":
            ksize = self.param1['value']*2 + 1
            self.python_code_str = f"{dst} = cv2.Laplacian({src}, 0, ksize={ksize})"
            return cv2.Laplacian(img, 0, ksize=ksize)
        elif self.name == "Mask":
            dst = img.copy()
            if self.param1['value'] == "north":
                self.python_code_str = f"{src}[ :{src}.shape[0]*{self.param2['value']}//100 , : ] = 0"
                dst[:dst.shape[0]*self.param2['value']//100, :] = 0
            elif self.param1['value'] == "west":
                self.python_code_str = f"{src}[ : , :{src}.shape[1]*{self.param2['value']}//100 ] = 0"
                dst[:, :dst.shape[1]*self.param2['value']//100] = 0
            elif self.param1['value'] == "south":
                self.python_code_str = f"{src}[ {src}.shape[0]*(1-{self.param2['value']})//100 , : ] = 0"
                dst[dst.shape[0]*(1-self.param2['value'])//100:, :] = 0
            else:
                self.python_code_str = f"{src}[ : , {src}.shape[1]*(1-{self.param2['value']})//100: ] = 0"
                dst[:, dst.shape[1]*(1-self.param2['value'])//100:] = 0
            return dst

    def render_param1(self, key_prefix: str = ""):
        if self.param1['label']:
            if self.param1['widget'] == "number":
                self.param1['value'] = st.number_input(self.param1['label'], key=f"{key_prefix}_p1", value=500, step=1, min_value=0, max_value=10000)
            elif self.param1['widget'] == "text":
                self.param1['value'] = st.text_input(self.param1['label'], key=f"{key_prefix}_p1")
            elif self.param1['widget'] == "normalize_slider":
                (self.param1['value'], self.param2['value']) = st.slider(self.param1['label'], min_value=0, max_value=255, value=(0,255), key=f"{key_prefix}_p1")
            elif self.param1['widget'] == "threshold_slider":
                self.param1['value'] = st.slider(self.param1['label'], min_value=0, max_value=255, value=123, key=f"{key_prefix}_p1")
            elif self.param1['widget'] == "percent_slider":
                self.param1['value'] = st.slider(self.param1['label'], min_value=0, max_value=100, value=30, key=f"{key_prefix}_p1")
            elif self.param1['widget'] == "slider":
                self.param1['value'] = st.slider(self.param1['label'], 1, 10, 1, key=f"{key_prefix}_p1")
            elif self.param1['widget'] == "checkbox":
                self.param1['value'] = st.checkbox(self.param1['label'], key=f"{key_prefix}_p1")
            elif self.param1['widget'] == "select_slider":
                self.param1['value'] = st.select_slider(self.param1['label'], options=["north", "west", "south", "east"], key=f"{key_prefix}_p1")


    def render_param2(self, key_prefix: str = ""):
        if self.param2['label']:
            if self.param2['widget'] == "number":
                self.param2['value'] = st.number_input(self.param2['label'], key=f"{key_prefix}_p2", value=500, step=1, min_value=0, max_value=10000)
            elif self.param2['widget'] == "text":
                self.param2['value'] = st.text_input(self.param2['label'], key=f"{key_prefix}_p2")
            elif self.param2['widget'] == "slider":
                self.param2['value'] = st.slider(self.param2['label'], 1, 10, 1, key=f"{key_prefix}_p2")
            elif self.param2['widget'] == "percent_slider":
                self.param2['value'] = st.slider(self.param2['label'], min_value=0, max_value=100, value=30, key=f"{key_prefix}_p2")
            elif self.param2['widget'] == "checkbox":
                self.param2['value'] = st.checkbox(self.param2['label'], key=f"{key_prefix}_p2")
            elif self.param2['widget'] == "select_slider":
                self.param2['value'] = st.select_slider(self.param1['label'], options=["north", "west", "south", "east"], key=f"{key_prefix}_p2")
    
def render_widget(param_dict, key):
    if param_dict['widget'] == "number":
        return st.number_input(param_dict['label'], key=key, value=500, step=1, min_value=0, max_value=10000)
    elif param_dict['widget'] == "text":
        return st.text_input(param_dict['label'], key=key)
    elif param_dict['widget'] == "normalize_slider":
        return st.slider(param_dict['label'], key=key, min_value=0, max_value=255, value=(0,255))
    elif param_dict['widget'] == "threshold_slider":
        return st.slider(param_dict['label'], key=key, min_value=0, max_value=255, value=123)
    elif param_dict['widget'] == "percent_slider":
        return st.slider(param_dict['label'], key=key, min_value=0, max_value=100, value=30)
    elif param_dict['widget'] == "slider":
        return st.slider(param_dict['label'], key=key, min_value=1, max_value=10, value=1)
    elif param_dict['widget'] == "select_slider":
        return st.select_slider(param_dict['label'], key=key, options=["north","west","south","east"])
    elif param_dict['widget'] == "checkbox":
        return st.checkbox(param_dict['label'], key=key)

# ----------------------
# App logic
# ----------------------
st.set_page_config(layout="wide")
options = ["Resize", "Normalize", "Grayscale", "Threshold", "Dilate", "Erode", "Laplacian", "Mask"]

if "steps" not in st.session_state:
    st.session_state.steps = []

if "last_action" not in st.session_state:
    st.session_state.last_action = None

st.title("Preprocessing Pipeline")

def move_item(i, direction):
    """Move step up or down."""
    new_index = i + direction
    if 0 <= new_index < len(st.session_state.steps):
        st.session_state.steps[i], st.session_state.steps[new_index] = (
            st.session_state.steps[new_index],
            st.session_state.steps[i],
        )
        st.session_state.last_action = f"Moved {st.session_state.steps[new_index].name} ↔ {st.session_state.steps[i].name}"
        st.rerun()


col1, col2 = st.columns(2)

with col1:

    uploader_cols = st.columns([0.92, 0.08])
    with uploader_cols[0]:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "pdf"])
    with uploader_cols[1]:
        original_visible = st.checkbox("👀", key=f"view_original")

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file

    for i, step in enumerate(st.session_state.steps):
        cols = st.columns([0.18, 0.25, 0.25, 0.08, 0.08, 0.08, 0.08])
        with cols[0]:
            st.markdown(f"**{i+1}. {step.name}**")
        with cols[1]:
            if step.param1['label']:
                if step.param1['widget'] == "normalize_slider":
                    step.param1['value'], step.param2['value'] = render_widget(step.param1, key=f"step{i}_p1")
                else:
                    step.param1['value'] = render_widget(step.param1, key=f"step{i}_p1")
        with cols[2]:
            if step.param2['label']:
                step.param2['value'] = render_widget(step.param2, key=f"step{i}_p2")
        with cols[3]:
            if st.button("⬆️", key=f"up_{i}"):
                move_item(i, -1)
        with cols[4]:
            if st.button("⬇️", key=f"down_{i}"):
                move_item(i, 1)
        with cols[5]:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.steps.pop(i)
                st.session_state.last_action = f"Removed step {i+1}"
                st.rerun()
        with cols[6]:        
            step.visible = st.checkbox("👀", key=f"view_{i}", value=step.visible)

# Add new step
    new_step = st.selectbox("Choose step to add:", options, key="add_select")
    if st.button("➕ Add Step"):
        st.session_state.steps.append(PreprocessingStep(new_step))
        st.session_state.last_action = f"Added {new_step}"
        st.rerun()

# Show last action
    # if st.session_state.last_action:
    #     st.info(st.session_state.last_action)

with col2:
    if "uploaded_file" in st.session_state:
        img = np.array(Image.open(st.session_state["uploaded_file"]))
        if original_visible:
            st.image(img, caption="Original Image")
        img_processed = img.copy()
        for i, step in enumerate(st.session_state.steps):
            img_processed = step.apply_step(img_processed, step_number=i+1)
            if step.visible:
                st.image(img_processed, caption=f"{i+1}. {step.name}")

with col1:
    # Show full pipeline
    with st.expander("See code"):
        for i, step in enumerate(st.session_state.steps):
            st.write(f"{i+1}. {step.python_code_str}")
