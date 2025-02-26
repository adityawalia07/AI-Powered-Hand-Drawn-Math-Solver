import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st


# HandTracking Module
class HandTrackingTheory:
    def __init__(self):
        self.prev_frame = None
        self.threshold = 30
        self.min_contour_area = 1000

    def detect_hand_landmarks(self, frame):
        """
        Theoretical implementation of hand landmark detection.
        This demonstrates the understanding of computer vision concepts.
        """

        def preprocess_frame(frame):
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            return blurred

        def detect_skin(frame):
            # Convert to YCrCb color space for better skin detection
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
            # Define skin color range in YCrCb
            lower_skin = np.array([0, 135, 85])
            upper_skin = np.array([255, 180, 135])
            # Create binary mask for skin regions
            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            return skin_mask

        def find_contours(binary_mask):
            # Find contours in the binary mask
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            # Filter contours by area to remove noise
            valid_contours = [
                cnt for cnt in contours
                if cv2.contourArea(cnt) > self.min_contour_area
            ]
            return valid_contours

        def extract_landmarks(contour):
            # Find convex hull and defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            landmarks = []
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    landmarks.append({
                        'tip': start,
                        'base': far,
                        'angle': self._calculate_angle(start, far, end)
                    })
            return landmarks

        def detect_fingers(landmarks):
            # Count potential fingers based on angle and position
            finger_count = 0
            for lm in landmarks:
                # Angles between 20 and 120 degrees typically represent fingers
                if 20 < lm['angle'] < 120:
                    finger_count += 1
            return finger_count

        # Main processing pipeline
        processed = preprocess_frame(frame)
        skin_mask = detect_skin(frame)
        contours = find_contours(skin_mask)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            landmarks = extract_landmarks(main_contour)
            finger_count = detect_fingers(landmarks)
            return landmarks, finger_count
        return None, 0

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def detect_motion(self, frame):
        """
        Detect hand motion between frames using optical flow
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None

        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame,
            current_frame,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        self.prev_frame = current_frame
        return flow


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