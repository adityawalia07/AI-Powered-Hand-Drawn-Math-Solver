import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")
st.image("MathsImg.jpeg")

col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.empty()

genai.configure(api_key="AIzaSyAVf4iXI_hbEk1Q4GtmNFItPZ70c2FlhJQ")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = tuple(lmList[8][0:2])
        if prev_pos is not None:
            cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 1]:
        canvas = np.zeros_like(canvas)
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        response = model.generate_content(["Solve this Maths Problem and explain it.", pil_image])
        return response.text
    return None

prev_pos = None
canvas = None
image_combined = None
output_text = ""

while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to capture image from camera.")
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)
        image_combined = img.copy()

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        current_pos, canvas = draw(info, prev_pos, canvas)
        prev_pos = current_pos
        new_output = sendToAI(model, canvas, fingers)
        if new_output:
            output_text = new_output

    image_combined = cv2.addWeighted(img, 0.65, canvas, 0.35, 0)
    FRAME_WINDOW.image(cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB))

    if output_text:
        output_text_area.subheader(output_text)

# Release the camera when the app is not running
cap.release()