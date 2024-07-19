from fastapi import FastAPI, WebSocket, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import base64
import random
import numpy as np
import cv2
import face_recognition
import dlib

from sqlalchemy import Column, Integer, String, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "mysql+mysqldb://root:@localhost:3306/facial_recognition"

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

@app.get('/')
def f():
    return {"status": "ok"}

@app.get('/hc')
def health_check():
    return {"status": "ok"}


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def detect_blinks(image):
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

def calculate_eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def get_face_encoding(image):
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        raise Exception("No face detected")
    return face_recognition.face_encodings(image, known_face_locations=face_locations)[0]

def optical_flow_liveness_detection(prev_frame, curr_frame, threshold=5.0):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_magnitude = np.mean(magnitude)
    
    return mean_magnitude > threshold

def is_live_feed(image_sequence):
    if len(image_sequence) < 2:
        return False
    for i in range(len(image_sequence) - 1):
        if not optical_flow_liveness_detection(image_sequence[i], image_sequence[i + 1]):
            return False
    return True

def detect_smile(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_recognition.face_landmarks(image)
        top_lip = shape[0]["top_lip"]
        bottom_lip = shape[0]["bottom_lip"]
        mouth = top_lip + bottom_lip[::-1]
        
        mouth_aspect_ratio = calculate_mouth_aspect_ratio(mouth)
        if mouth_aspect_ratio > 0.5:
            return True
    return False

def calculate_mouth_aspect_ratio(mouth):
    A = np.linalg.norm(np.array(mouth[3]) - np.array(mouth[9]))
    B = np.linalg.norm(np.array(mouth[2]) - np.array(mouth[10]))
    C = np.linalg.norm(np.array(mouth[4]) - np.array(mouth[8]))
    D = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))
    mar = (A + B + C) / (3.0 * D)
    return mar

def detect_nod(image_sequence):
    # Use optical flow or other methods to detect nodding motion
    return is_live_feed(image_sequence)

def detect_wink(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
    except Exception as e:
        print(image)
        raise e
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_recognition.face_landmarks(image)
        left_eye = shape[0]["left_eye"]
        right_eye = shape[0]["right_eye"]
        
        left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye)
        right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye)
        
        if (left_eye_aspect_ratio < 0.2 and right_eye_aspect_ratio > 0.2) or (left_eye_aspect_ratio > 0.2 and right_eye_aspect_ratio < 0.2):
            return True
    return False

def detect_color(image, color):
    # Simple color detection logic
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array(color['lower'])
    upper_color = np.array(color['upper'])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return cv2.countNonZero(mask) > 0

@app.websocket("/register")
async def register(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    
    image_sequence = []
    # challenges = ["blink", "turn_left", "turn_right", "smile", "nod", "wink"]
    # challenges = ["blink", "smile", "nod", "wink"]
    # challenges = ["blink", "nod", "wink"]
    challenges = ["blink", "wink"]
    selected_challenges = random.sample(challenges, 2)
    challenge_index = 0

    await websocket.send_json({"current_challenge": selected_challenges[challenge_index]})

    async for message in websocket.iter_text():
        message = json.loads(message)
        image_message = message['image']
        base64_str = image_message.split(",")[1]
        image_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_sequence.append(image)
        if challenge_index >= len(selected_challenges):
            break

        if len(image_sequence) > 10:
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
            elif selected_challenges[challenge_index] in ["turn_left", "turn_right"]:
                await websocket.send_json({"msg": f"trying to detect {selected_challenges[challenge_index]}"})
                await websocket.send_json({"msg": f"live feed: {is_live_feed(image_sequence)}"})
                if is_live_feed(image_sequence):
                    await websocket.send_json({"msg": f"{selected_challenges[challenge_index]} completed"})
                    challenge_index += 1
            
            if challenge_index >= len(selected_challenges):
                face_encoding = get_face_encoding(image)
                user = User(username=message['username'], face_encoding=face_encoding.tobytes())
                db.add(user)
                db.commit()
                db.refresh(user)
                await websocket.send_json({"success": True, "msg": f"{message['username']} registered successfully"})
                await websocket.close()
                return

            if challenge_index < len(selected_challenges):
                await websocket.send_json({"current_challenge": selected_challenges[challenge_index]})
            image_sequence.pop(0)
    
    await websocket.send_json({"success": False, "msg": "registration failed"})
    await websocket.close()



@app.websocket("/login")
async def login(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    
    image_sequence = []
    # challenges = ["blink", "turn_left", "turn_right", "smile", "nod", "wink"]
    # challenges = ["blink", "smile", "nod", "wink"]
    # challenges = ["blink", "nod", "wink"]
    challenges = ["blink", "wink"]
    
    selected_challenges = random.sample(challenges, 2)
    challenge_index = 0

    await websocket.send_json({"current_challenge": selected_challenges[challenge_index]})
    
    async for message in websocket.iter_text():
        message = json.loads(message)
        image_message = message['image']
        base64_str = image_message.split(",")[1]
        image_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_sequence.append(image)
        if challenge_index >= len(selected_challenges):
            break

        if len(image_sequence) > 10:
            if selected_challenges[challenge_index] == "blink" and detect_blinks(image_sequence[-1]):
                challenge_index += 1
            elif selected_challenges[challenge_index] == "smile" and detect_smile(image_sequence[-1]):
                challenge_index += 1
            elif selected_challenges[challenge_index] == "nod" and detect_nod(image_sequence):
                challenge_index += 1
            elif selected_challenges[challenge_index] == "wink" and detect_wink(image_sequence[-1]):
                challenge_index += 1
            elif selected_challenges[challenge_index] in ["turn_left", "turn_right"]:
                if is_live_feed(image_sequence):
                    challenge_index += 1
            
            if challenge_index >= len(selected_challenges):
                face_encoding = get_face_encoding(image)
                user = db.query(User).filter_by(username = message['username']).one_or_none()
                if user is None:
                    break
                # for user in users:
                db_face_encoding = np.frombuffer(user.face_encoding, dtype=np.float64)
                matches = face_recognition.compare_faces([db_face_encoding], face_encoding)
                if True in matches:
                    await websocket.send_json({"success": True, "msg": f"Login successful for {user.username}"})
                    await websocket.close()
                    return
            if challenge_index < len(selected_challenges):
                await websocket.send_json({"current_challenge": selected_challenges[challenge_index]})
            image_sequence.pop(0)
    
    await websocket.send_json({"success": False, "msg": "Login failed"})
    await websocket.close()

# TODO: full face detection and not partial
# TODO: fix challenges
# TODO: deepfake/fake detection
# TODO: wink and blink differentiation
# TODO: left wink and right wink separation
