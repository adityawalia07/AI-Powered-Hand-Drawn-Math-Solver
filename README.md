# AI-Powered-Hand-Drawn-Math-Solver

This project combines real-time hand tracking, computer vision, and AI to solve hand-drawn mathematical problems.<br>
Users can draw equations using hand gestures, which are then interpreted and solved by an AI model.

## Features
  - Real-time hand tracking and gesture recognition.<br>
  - Virtual canvas for drawing equations.<br>
  - AI-powered math problem solving and explanation.<br>
  - Live video feed with drawn equations overlay.<br>
  - Interactive Streamlit user interface.<br><br>

## Technologies Used
  - Python
  - OpenCV
  - NumPy
  - cvzone
  - Google's Generative AI (Gemini 1.5 Flash model)
  - Streamlit

## Installation

### Clone this repository
- Install the required packages:
- Copy
  ```bash
  pip install opencv-python cvzone numpy google-generativeai streamlit 
  ```

### Set up Google API credentials:
- Obtain an API key from the Google Cloud Console.
- Replace "YOUR_API_KEY" in the code with your actual API key.

## How It Works

 - The script captures video from the default webcam.
 - It uses the HandTrackingModule from cvzone to detect and track hands in the video feed.
 - A virtual canvas is created for drawing equations.
 - The program tracks the position of the user's index finger tip for drawing.
 - Different hand gestures are used for drawing, erasing, and requesting solutions.
 - Drawn equations are sent to the Gemini AI model for interpretation and solving.
 - Solutions are displayed in real-time on the Streamlit interface.

## Code Structure

### Main loop:

 - Captures video frames
 - Detects hands and finger positions
 - Updates canvas based on hand gestures
 - Sends drawn equations to AI for solving
 - Renders the canvas and solutions on the Streamlit interface

## Usage

- Run the Streamlit app:
-Copy
```bash
streamlit run main2.py
```
### Use hand gestures to interact with the app:

- Index finger: Draw equations
- All fingers open: Erase canvas
- Four fingers up: Request solution

### View the AI-generated solutions in the sidebar

## Customization

 - Adjust the detectionCon parameter in the HandDetector initialization to change the hand detection sensitivity.
 - Modify the canvas size by changing the cap.set() parameters.
 - Adjust the thickness of drawing lines by modifying the cv2.line() thickness parameter.

## Future Improvements

 - Add support for more complex mathematical notations.
 - Implement handwriting recognition for improved accuracy.
 - Add ability to save and load equations.
 - Improve UI for better user experience.
 - Optimize performance for smoother real-time interaction.

## Output

https://github.com/user-attachments/assets/53a5a0fa-72a2-4cb5-9486-d4c6778054a2

