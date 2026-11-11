import streamlit as st 
from PIL import Image
import numpy as np
import cv2
import tempfile
from tensorflow.keras.models import load_model as keras_load_model 

@st.cache_resource
def load_model(model_path="lane_detection_model.h5"):
    model = keras_load_model(model_path, compile=False)
    return model



def predict_lanes(image, model):
    image = image.convert("RGB").resize((320, 320))
    input_image = np.array(image, dtype=np.float32)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    input_image = np.expand_dims(input_image / 255.0, axis=0)

    output = model.predict(input_image, verbose=0)

    print("Output Min:", np.min(output))
    print("Output Max:", np.max(output))
    print("Output Mean:", np.mean(output))

    predicted_mask = output.squeeze() > 0.01
    print("White Pixels:", np.sum(predicted_mask))

    return predicted_mask



def grayscale(image):
    """Convert image (NumPy array) to grayscale."""
    if isinstance(image, np.ndarray):  
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise TypeError("Input should be a NumPy array.")
    return gray


def gaussian_blur(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred


def canny(image, min_thresh=50, max_thresh=150):
    blur = gaussian_blur(grayscale(image))
    edges = cv2.Canny(blur, min_thresh, max_thresh)
    return edges


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(50, height), (image.shape[1] - 50, height), (image.shape[1] // 2, int(height * 0.6))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked



def average_lines(image, lines):
    left_lines = []
    right_lines = []
    left_weights = []
    right_weights = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  
        slope, intercept = parameters
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  

        if slope < 0:  
            left_lines.append((slope, intercept))#left lane
            left_weights.append(length)
        else:  
            right_lines.append((slope, intercept))#right lane
            right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None

    lanes = []
    for lane in (left_lane, right_lane):
        if lane is not None:
            slope, intercept = lane
            y1 = image.shape[0]
            y2 = int(y1 * 0.6)  
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            lanes.append((x1, y1, x2, y2))
    return lanes

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image



# Streamlit UI 
html_temp =html_temp = html_temp = """
    <style>
        body {
            background-color: #1f1f2e;
        }
        .title {
            background-color: #1f1f2e;
            color: white;
            padding: 15px;
            text-align: center;
            font-size: 30px;
            border-radius: 10px;
        }
        .subtitle {
            background-color: #1f1f2e;
            color: #00ff00;
            padding: 10px;
            font-size: 20px;
            margin: 5px;
        }
    </style>
    <div>
        <h1 class="title">Lane Detection Application</h1>
        <h3 class="subtitle">Upload an Image or Video to Start Detecting Lanes!</h3>
    </div>
""" 
st.markdown(html_temp, unsafe_allow_html=True)




activities = ['Image', 'Video']
choice = st.selectbox('Select Input Type', activities)
#For Image
if choice == 'Image':
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        image_np = np.array(image)

        st.image(image, caption='Original Image', use_column_width=True)

        options = ['Original', 'Grayscale', 'Gaussian Blur', 'Canny Edges', 'Region of Interest', 'Model Prediction', 'Final Overlay']
        transformation = st.selectbox('Select Transformation', options)

        if transformation == 'Original':
            st.image(image, caption='Original Image', use_column_width=True)

        elif transformation == 'Gaussian Blur':
             blurred_image = gaussian_blur(image_np) 
             st.image(blurred_image, caption='Gaussian Blurred Image', use_column_width=True)

        elif transformation == 'Grayscale':
            gray_image = grayscale(image_np)  
            st.image(gray_image, caption='Grayscale Image', use_column_width=True)

    

        elif transformation == 'Canny Edges':
            min_thresh = st.slider('Min Threshold', 0, 255, 50)
            max_thresh = st.slider('Max Threshold', 0, 255, 150)
            edges = canny(image_np, min_thresh, max_thresh)  
            st.image(edges, caption='Edges Detected (Canny)', use_column_width=True)

        elif transformation == 'Region of Interest':
            edges = canny(image_np)  
            roi_image = region_of_interest(edges)  
            st.image(roi_image, caption='Region of Interest', use_column_width=True)

        elif transformation == 'Model Prediction':
            model = load_model()
            lane_mask = predict_lanes(image, model)
            st.image(lane_mask.astype(np.uint8) * 255,
                     caption='Predicted Lane Mask',
                     use_column_width=True)

        
        elif transformation == 'Final Overlay':
            frame = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            height, width = edges.shape
            mask = np.zeros_like(edges)

            polygon = np.array([[
                (int(width * 0.1), height),
                (int(width * 0.4), int(height * 0.55)),
                (int(width * 0.6), int(height * 0.55)),
                (int(width * 0.9), height)
            ]], dtype=np.int32)

            cv2.fillPoly(mask, polygon, 255)
            cropped = cv2.bitwise_and(edges, mask)

            lines = cv2.HoughLinesP(
                cropped, 1, np.pi / 180, 40,
                minLineLength=40, maxLineGap=100
            )

            overlay = frame.copy()

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = map(int, np.array(line).reshape(-1)[:4])
                    cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 5)

            result = cv2.addWeighted(frame, 0.8, overlay, 0.8, 0)
            st.image(result, caption="Final Lane Detection", use_column_width=True)

            
#For Video
elif choice == 'Video':
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        video = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            edges = canny(frame_rgb)
            roi_image = region_of_interest(edges)
            lines = cv2.HoughLinesP(roi_image, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=5)
            left_right_lines = average_lines(frame, lines)
            line_image = display_lines(frame, left_right_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

            stframe.image(combo_image, channels="BGR", use_column_width=True)

        video.release()
        st.write("Video Processing Complete")
