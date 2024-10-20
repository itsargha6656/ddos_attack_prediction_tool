import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, WebSocket
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import jwt
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from databases import Database
from passlib.context import CryptContext

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "sqlite:///./test.db"
database = Database(DATABASE_URL)
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# JWT Configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database Models
class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

class DBPatient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    date_of_birth = Column(DateTime)
    records = relationship("DBMedicalRecord", back_populates="patient")

class DBMedicalRecord(Base):
    __tablename__ = "medical_records"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    date = Column(DateTime, default=datetime.utcnow)
    diagnosis = Column(String)
    image_path = Column(String)
    patient = relationship("DBPatient", back_populates="records")

Base.metadata.create_all(bind=engine)

# Pydantic Models
class User(BaseModel):
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str

class FractureDetection(BaseModel):
    fracture_detected: bool
    fracture_type: str
    probability: float
    affected_bone: str
    severity: str

class Patient(BaseModel):
    name: str
    date_of_birth: datetime

class MedicalRecord(BaseModel):
    patient_id: int
    diagnosis: str
    image_path: str

# ML Model
def load_model():
    # Create a simple model without specifying input_shape
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model = load_model()

def preprocess_image(image: Image.Image):
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=0)

def detect_fracture(image: np.ndarray):
    prediction = model.predict(image)
    fracture_detected = bool(prediction[0][0] > 0.5)
    fracture_type = ["transverse", "oblique", "comminuted"][int(prediction[0][0] * 3) % 3]
    probability = float(prediction[0][0])
    affected_bone = ["femur", "tibia", "radius", "ulna"][int(prediction[0][0] * 4) % 4]
    severity = ["mild", "moderate", "severe"][int(prediction[0][0] * 3) % 3]

    return FractureDetection(
        fracture_detected=fracture_detected,
        fracture_type=fracture_type,
        probability=probability,
        affected_bone=affected_bone,
        severity=severity
    )

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    db = SessionLocal()
    user = db.query(DBUser).filter(DBUser.username == username).first()
    db.close()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    db = SessionLocal()
    user = db.query(DBUser).filter(DBUser.username == username).first()
    db.close()
    if user is None:
        raise credentials_exception
    return user

# API Endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    preprocessed_image = preprocess_image(image)
    result = detect_fracture(preprocessed_image)

    return result

@app.post("/patients/", response_model=Patient)
async def create_patient(patient: Patient, current_user: User = Depends(get_current_user)):
    db_patient = DBPatient(**patient.dict())
    db = SessionLocal()
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    db.close()
    return db_patient

@app.post("/medical_records/", response_model=MedicalRecord)
async def create_medical_record(record: MedicalRecord, current_user: User = Depends(get_current_user)):
    db_record = DBMedicalRecord(**record.dict())
    db = SessionLocal()
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    db.close()
    return db_record

@app.get("/patients/{patient_id}/records/", response_model=List[MedicalRecord])
async def get_patient_records(patient_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    records = db.query(DBMedicalRecord).filter(DBMedicalRecord.patient_id == patient_id).all()
    db.close()
    return records

@app.get("/educational_content/")
async def get_educational_content(fracture_type: Optional[str] = None):
    content = {
        "transverse": "A transverse fracture is a break that goes straight across the bone...",
        "oblique": "An oblique fracture is a break that occurs at an angle across the bone...",
        "comminuted": "A comminuted fracture is when the bone breaks into three or more pieces..."
    }
    if fracture_type:
        return {fracture_type: content.get(fracture_type, "Content not found")}
    return content

# WebSocket for telemedicine chat
@app.websocket("/ws/chat/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received in room {room_id}: {data}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# PACS Integration (simplified example)
@app.post("/pacs/send_image/")
async def send_to_pacs(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    contents = await file.read()
    pacs_id = hash(contents)  # This is a simplification; real PACS systems use more complex identifiers
    return {"message": "Image sent to PACS", "pacs_id": pacs_id}

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)