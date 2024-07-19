from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import json
from typing import Optional
import base64
import random
import numpy as np
import cv2
import face_recognition
import face_recognition_models
import dlib
import time
import os

from sqlalchemy import Column, Integer, String, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    face_encoding = Column(LargeBinary, nullable=False)  # Store face encoding

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

app = FastAPI()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

INITIAL_FRAMES = 4

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def detect_blinks(image):
    '''
        TODO: could be improved by tracking the image sequence and checking 
            if the eyes are closed for a certain number of frames and open before and after
    '''
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_recognition.face_landmarks(image)
            left_eye = shape[0]["left_eye"]
            right_eye = shape[0]["right_eye"]
            
            left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye)
            right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye)
            
            if left_eye_aspect_ratio < 0.2 and right_eye_aspect_ratio < 0.2:
                return True    
        return False
    except Exception as e:
        return False

def calculate_eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def get_face_encoding(image):
    face_locations = face_recognition.face_locations(image, model="cnn") # TODO: use model="cnn" for better accuracy?
    if len(face_locations) == 0:
        raise Exception("No face detected")
    return face_recognition.face_encodings(image, known_face_locations=face_locations)[0]

# FIXME: this is not working as expected
def optical_flow_liveness_detection(prev_frame, curr_frame, threshold=5.0):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_magnitude = np.mean(magnitude)
    
    return mean_magnitude > threshold

# FIXME: this is not working as expected
def is_live_feed(image_sequence):
    if len(image_sequence) < 2:
        return False
    for i in range(len(image_sequence) - 1):
        if not optical_flow_liveness_detection(image_sequence[i], image_sequence[i + 1]):
            return False
    return True

def detect_smile(image):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for (top, right, bottom, left) in face_locations:
            face_gray = gray_image[top:bottom, left:right]
            smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=120, minSize=(40, 40))
            if len(smiles) > 0:
                return True
        return False
    except Exception as e:
        return False

def detect_nod(image_sequence):
    '''
    # TODO: should probably keep track of original y position, 
                and all subsequent y positions and detect if nod direction stayed the same throughout
                and if the nod was significant enough at the end
    '''
    try:
        chunk = INITIAL_FRAMES if len(image_sequence) >= INITIAL_FRAMES else len(image_sequence)
        nod_threshold = 10
        previous_y = None

        for frame in image_sequence[-chunk:]:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
            
            for face_landmarks in face_landmarks_list:
                nose_tip = face_landmarks['nose_tip'][2]  # [2] is typically the bottom-most point of the nose tip
                current_y = nose_tip[1]
                if previous_y is not None:
                    movement = current_y - previous_y
                    if movement > nod_threshold:
                        return True
                    
                previous_y = current_y
        
        return False
    except Exception as e:
        return False

def detect_turn(image_sequence, direction: str):
    if direction.lower() not in ["left", "right"]:
        return False
    try:
        turn_threshold = 10
        previous_x = None

        for frame in image_sequence:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
            for face_landmarks in face_landmarks_list:                
                nose_tip = face_landmarks['nose_tip'][2] # [2] is typically the bottom-most point of the nose tip
                current_x = nose_tip[0]
                
                if previous_x is not None:
                    if direction == "left":
                        movement = previous_x - current_x  # Positive movement indicates turning left
                    elif direction == "right":
                        movement = current_x - previous_x
                    if movement > turn_threshold:
                        return True
                previous_x = current_x
        return False
    except Exception as e:
        return False

def detect_wink(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_recognition.face_landmarks(image)
            left_eye = shape[0]["left_eye"]
            right_eye = shape[0]["right_eye"]
            
            left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye)
            right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye)

            return (left_eye_aspect_ratio < 0.2 and right_eye_aspect_ratio > 0.2) or (left_eye_aspect_ratio > 0.2 and right_eye_aspect_ratio < 0.2)
        return False
    except Exception as e:
        return False
    
def is_full_face_detected(face_landmarks_list):
    required_landmarks = ["chin", "left_eye", "right_eye", "nose_tip", "top_lip", "bottom_lip", "left_eyebrow", "right_eyebrow", "nose_bridge"]
    for face_landmarks in face_landmarks_list:
        for landmark in required_landmarks:
            if landmark not in face_landmarks:
                return False
    return True

def calculate_landmark_shifts(image_sequence):
    sequence_length = INITIAL_FRAMES if len(image_sequence) >= INITIAL_FRAMES else len(image_sequence)
    image_sequence = image_sequence[-INITIAL_FRAMES:]
    landmarks_list = []
    for image in image_sequence:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(rgb_image)
        if face_landmarks_list:
            # Assuming we're dealing with the first detected face for simplicity
            face_landmarks = face_landmarks_list[0]
            # Convert the landmarks to a numpy array
            coords = []
            for key in face_landmarks:
                coords.extend(face_landmarks[key])
            landmarks_list.append(np.array(coords))

    shifts_list = []
    # Calculate the Euclidean distance between corresponding landmarks in two frames
    for i in range(1, len(landmarks_list)):
        landmarks1 = landmarks_list[i-1]
        landmarks2 = landmarks_list[i]
        shifts = np.linalg.norm(landmarks1 - landmarks2, axis=1)
        shifts_list.append(shifts)
    return shifts_list

def average_shifts(shifts_list):
    # Compute the average shifts across all frames
    if len(shifts_list) == 0:
        return 0
    return np.mean([np.mean(shifts) for shifts in shifts_list if len(shifts) > 0])

def detect_micromovents(image_sequence):
    avg_shifts = average_shifts(calculate_landmark_shifts(image_sequence))
    mean_shift = np.mean(avg_shifts)
    return mean_shift > 0.5

async def process(websocket: WebSocket, db: Session, process_type: str):
    await websocket.accept()
    start_time = time.time()
    try:
        image_sequence = []
        challenges = ["blink", "wink", "smile", "nod", "turn_left", "turn_right"]
        selected_challenges = random.sample(challenges, 2)
        challenge_index = 0

        async for message in websocket.iter_text():
            if time.time() - start_time > 30:
                await websocket.send_json({"success": False, "msg": "Time limit exceeded, try again"})
                await websocket.close()
                return
            message = json.loads(message)
            image_message = message['image']
            base64_str = image_message.split(",")[1]
            image_data = base64.b64decode(base64_str)
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image_sequence.append(image)
            if challenge_index >= len(selected_challenges):
                break

            rgb_image = cv2.cvtColor(image_sequence[-1], cv2.COLOR_BGR2RGB)
            face_landmarks_list = face_recognition.face_landmarks(rgb_image)

            if not face_landmarks_list or not is_full_face_detected(face_landmarks_list):
                await websocket.send_json({"success": False, "msg": "Full face not detected"})
                await websocket.close()
                return

            if len(image_sequence) > INITIAL_FRAMES:
                if not detect_micromovents(image_sequence):
                    await websocket.send_json({"success": False, "msg": "Face not moving enough"})
                    await websocket.close()
                    return
                if selected_challenges[challenge_index] == "blink" and detect_blinks(image_sequence[-1]):
                    await websocket.send_json({"msg": f"{selected_challenges[challenge_index]} completed"})
                    challenge_index += 1
                elif selected_challenges[challenge_index] == "smile" and detect_smile(image_sequence[-1]):
                    await websocket.send_json({"msg": f"{selected_challenges[challenge_index]} completed"})
                    challenge_index += 1
                elif selected_challenges[challenge_index] == "nod" and detect_nod(image_sequence):
                    await websocket.send_json({"msg": f"{selected_challenges[challenge_index]} completed"})
                    challenge_index += 1
                elif selected_challenges[challenge_index] == "wink" and detect_wink(image_sequence[-1]):
                    await websocket.send_json({"msg": f"{selected_challenges[challenge_index]} completed"})
                    challenge_index += 1
                elif selected_challenges[challenge_index] == "turn_left" and detect_turn(image_sequence, "left"):
                    await websocket.send_json({"msg": f"{selected_challenges[challenge_index]} completed"})
                    challenge_index += 1
                elif selected_challenges[challenge_index] == "turn_right" and detect_turn(image_sequence, "right"):
                    await websocket.send_json({"msg": f"{selected_challenges[challenge_index]} completed"})
                    challenge_index += 1
                if challenge_index >= len(selected_challenges):
                    face_encoding = get_face_encoding(image)
                    if process_type == "login":
                        user = db.query(User).filter_by(username = message['username']).one_or_none()
                        if user is None:
                            break
                        db_face_encoding = np.frombuffer(user.face_encoding, dtype=np.float64)
                        matches = face_recognition.compare_faces([db_face_encoding], face_encoding)
                        if True in matches:
                            await websocket.send_json({"success": True, "msg": f"Login successful for {user.username}"})
                            await websocket.close()
                            return
                        else:
                            await websocket.send_json({"success": False, "msg": "Login failed"})
                            await websocket.close()
                            return
                    if process_type == "register":
                        user = User(username=message['username'], face_encoding=face_encoding.tobytes())
                        db.add(user)
                        db.commit()
                        db.refresh(user)
                        await websocket.send_json({"success": True, "msg": f"{message['username']} registered successfully"})
                        await websocket.close()
                        return

                if challenge_index < len(selected_challenges):
                    await websocket.send_json({"current_challenge": selected_challenges[challenge_index]})
        await websocket.send_json({"success": False, "msg": f"{"registration" if type == "register" else "login"} failed"})
        await websocket.close()
    except Exception as e:
        await websocket.send_json({"success": False, "msg": f"{"registration" if type == "register" else "login"} failed"})
        await websocket.close()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/hc')
def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/{filename:path}")
async def serve_static_file(request: Request, filename: str, mode: Optional[str] = None):
    if filename in ("main.html","index.html","frontend_script.js"):
        # Serve main.html with mode query parameter
        # Here, you could include logic to serve different content based on the mode
        return FileResponse(f"static/{filename}")
    else:
        return {"error": "File not found"}

@app.websocket("/register")
async def register(websocket: WebSocket, db: Session = Depends(get_db)):
    await process(websocket, db, "register")

@app.websocket("/login")
async def login(websocket: WebSocket, db: Session = Depends(get_db)):
    await process(websocket, db, "login")

# TODO: improve/revisit logic for challenges and improve arithmetics
# TODO: someone may use a script to send images based on the challenge so we need to detect that/detect liveness
# TODO: can we detect depth of the face/image to detect if it's a real face or a photo?
# TODO: deepfake/fake detection
# TODO: frame life check to assert consistency, lighting, texture, continuity etc. instead of single images anywhere