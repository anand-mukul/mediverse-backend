from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import httpx
import json
import base64
import asyncio
from twilio.rest import Client
import paho.mqtt.client as mqtt
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)

# 1. Setup Environment and Logging
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Global Client Variables (Initialized in Lifespan)
mongo_client: Optional[AsyncIOMotorClient] = None
db = None
mqtt_client: Optional[mqtt.Client] = None
twilio_client: Optional[Client] = None

# 3. Lifespan Manager (Startup/Shutdown Logic)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    global mongo_client, db, mqtt_client, twilio_client
    
    # Mongo
    try:
        mongo_url = os.environ['MONGODB_URI']
        mongo_client = AsyncIOMotorClient(mongo_url)
        db = mongo_client[os.environ['DB_NAME']]
        logger.info("✅ MongoDB Connected")
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")

    # Twilio
    try:
        twilio_client = Client(
            os.environ['TWILIO_ACCOUNT_SID'],
            os.environ['TWILIO_AUTH_TOKEN']
        )
        logger.info("✅ Twilio Client Initialized")
    except Exception as e:
        logger.warning(f"⚠️ Twilio Client Failed (SMS/Calls will fail): {e}")

    # MQTT
    try:
        mqtt_client = mqtt.Client()
        broker = os.environ.get('MQTT_BROKER', 'mqtt.eclipseprojects.io')
        port = int(os.environ.get('MQTT_PORT', '1883'))
        mqtt_client.connect(broker, port, 60)
        mqtt_client.loop_start()
        logger.info(f"✅ MQTT Connected to {broker}:{port}")
    except Exception as e:
        logger.warning(f"⚠️ MQTT Broker not available: {e}")

    yield # App runs here

    # --- Shutdown ---
    if mongo_client:
        mongo_client.close()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    logger.info("🛑 Services Shutdown")

# 4. App Definition
app = FastAPI(title="Mediverse API", version="1.0.0", lifespan=lifespan)
api_router = APIRouter(prefix="/api")

# CORS Setup
origins = os.environ.get('CORS_ORIGINS', '*').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    phone: str
    emergency_contact: Optional[str] = None
    blood_type: Optional[str] = None
    allergies: Optional[List[str]] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Appointment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    doctor_name: str
    specialty: str
    date: str
    time: str
    status: str = "scheduled"
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Prescription(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    medication: str
    dosage: str
    frequency: str
    duration: str
    doctor: str
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class IoTCommand(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    command: str
    device: str
    status: str = "pending"
    user_id: str
    executed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EmergencyLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    type: str
    location: Optional[str] = None
    status: str = "active"
    responders_notified: List[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ==================== INPUT MODELS ====================

class VoiceTranscribeRequest(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None

class AIIntentRequest(BaseModel):
    text: str
    user_context: Optional[Dict[str, Any]] = {}

class AppointmentBookRequest(BaseModel):
    user_id: str
    doctor_name: str
    specialty: str
    date: str
    time: str
    notes: Optional[str] = None

class EmergencyTriggerRequest(BaseModel):
    user_id: str
    type: str  # heart_attack, fall, medical_emergency
    location: Optional[str] = None

class IoTCommandRequest(BaseModel):
    user_id: str
    device: str  # medicine_bot, care_bot
    action: str  # deliver, monitor, assist

class N8NTriggerRequest(BaseModel):
    flow: str
    data: Dict[str, Any]

# ==================== SERVICES ====================

async def deepgram_transcribe_rest(audio_data: bytes) -> Dict[str, Any]:
    """Transcribe audio using Deepgram REST API (for files/chunks)"""
    try:
        url = f"{os.environ.get('DEEPGRAM_URL', 'https://api.deepgram.com/v1/listen')}?model=nova-2&smart_format=true"
        headers = {
            "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
            "Content-Type": "audio/wav" # Defaulting to wav, allows Deepgram to sniff
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, content=audio_data)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Deepgram REST error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

async def groq_intent_analysis(text: str, context: Dict) -> Dict[str, Any]:
    """Analyze intent using Groq AI"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""You are Mediverse.AI. Analyze this user command:
User: "{text}"
Response JSON: {{
    "intent": "<book_appointment|emergency|iot_command|prescription_refill|health_query>",
    "action": "<specific action>",
    "entities": {{"key": "value"}},
    "urgency": "<low|medium|high|critical>",
    "response": "<natural language response>"
}}"""
            
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            response = await client.post(
                os.environ['GROQ_URL'], # https://api.groq.com/openai/v1/chat/completions
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return json.loads(result['choices'][0]['message']['content'])
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return {"intent": "error", "response": "I'm having trouble thinking right now."}

async def trigger_n8n_webhook(flow: str, data: Dict) -> Dict[str, Any]:
    try:
        base_url = os.environ.get('N8N_WEBHOOK_BASE', 'http://localhost:5678/webhook')
        webhook_url = f"{base_url}/{flow}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            # We don't await the response body to prevent blocking if n8n is slow
            await client.post(webhook_url, json=data)
            return {"status": "triggered", "flow": flow}
    except Exception as e:
        logger.warning(f"n8n webhook error: {e}")
        return {"status": "mocked", "message": "Triggered (Mock)"}

def publish_iot_command(device: str, action: str) -> bool:
    if not mqtt_client:
        return False
    try:
        topic = f"{os.environ.get('MQTT_TOPIC', 'mediverse/iot/commands')}/{device}"
        payload = json.dumps({"action": action, "ts": datetime.now(timezone.utc).isoformat()})
        info = mqtt_client.publish(topic, payload)
        return info.rc == 0
    except Exception as e:
        logger.error(f"MQTT Publish Error: {e}")
        return False

# ==================== ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "Mediverse API - Intelligent Health Ecosystem", "status": "active"}

# Voice Recognition Routes
@api_router.post("/voice/transcribe")
async def transcribe_voice(request: VoiceTranscribeRequest):
    if request.audio_base64:
        audio_data = base64.b64decode(request.audio_base64)
    elif request.audio_url:
        async with httpx.AsyncClient() as client:
            resp = await client.get(request.audio_url)
            audio_data = resp.content
    else:
        raise HTTPException(status_code=400, detail="Provide audio source")
    
    result = await deepgram_transcribe_rest(audio_data)
    # Safely extract transcript
    try:
        transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
    except (KeyError, IndexError):
        transcript = ""
        
    return {"transcript": transcript, "raw": result}

@api_router.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """Real-time bi-directional voice streaming."""
    await websocket.accept()
    
    try:
        deepgram = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) > 0:
                await websocket.send_json({
                    "transcript": sentence,
                    "is_final": result.is_final
                })

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2", 
            language="en-US", 
            smart_format=True,
            interim_results=True,
        )

        if await dg_connection.start(options) is False:
            logger.error("Failed to connect to Deepgram")
            await websocket.close()
            return

        try:
            while True:
                data = await websocket.receive_bytes()
                await dg_connection.send(data)
        except WebSocketDisconnect:
            pass
        finally:
            await dg_connection.finish()

    except Exception as e:
        logger.error(f"WS Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

# AI Intent Routes
@api_router.post("/ai/intent")
async def analyze_intent(request: AIIntentRequest):
    return await groq_intent_analysis(request.text, request.user_context)

@api_router.post("/ai/chat")
async def ai_chat(request: AIIntentRequest):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": request.text}
                ]
            }
            response = await client.post(os.environ['GROQ_URL'], headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return {"response": data['choices'][0]['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Appointment Routes
@api_router.post("/appointments/book", response_model=Appointment)
async def book_appointment(request: AppointmentBookRequest):
    appointment = Appointment(**request.model_dump())
    doc = appointment.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.appointments.insert_one(doc)
    
    # SMS Logic
    if twilio_client:
        try:
            user = await db.users.find_one({"id": request.user_id})
            if user and user.get('phone'):
                twilio_client.messages.create(
                    body=f"Confirmed: Dr. {request.doctor_name} at {request.time}",
                    from_=os.environ.get('TWILIO_PHONE_NUMBER'),
                    to=user['phone']
                )
        except Exception as e:
            logger.error(f"SMS Failed: {e}")
            
    return appointment

@api_router.get("/appointments/{user_id}", response_model=List[Appointment])
async def get_appointments(user_id: str):
    cursor = db.appointments.find({"user_id": user_id}, {"_id": 0})
    appointments = await cursor.to_list(length=100)
    return appointments

# Emergency Routes
@api_router.post("/emergency/trigger", response_model=EmergencyLog)
async def trigger_emergency(request: EmergencyTriggerRequest):
    emergency = EmergencyLog(**request.model_dump())
    doc = emergency.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.emergencies.insert_one(doc)

    user = await db.users.find_one({"id": request.user_id})
    contact = user.get('emergency_contact') if user else None

    if twilio_client and contact:
        try:
            twilio_client.messages.create(
                body=f"🚨 EMERGENCY: {user.get('name', 'User')} needs help. {request.type}",
                from_=os.environ['TWILIO_PHONE_NUMBER'],
                to=contact
            )
            emergency.responders_notified.append("sms_sent")
        except Exception as e:
            logger.error(f"Emergency SMS failed: {e}")

    await trigger_n8n_webhook("emergency", doc)
    publish_iot_command("care_bot", "emergency_assist")
    
    return emergency

# IoT Routes
@api_router.post("/iot/command", response_model=IoTCommand)
async def send_iot_command(request: IoTCommandRequest):
    command = IoTCommand(
        command=request.action,
        device=request.device,
        user_id=request.user_id
    )
    
    success = publish_iot_command(request.device, request.action)
    if success:
        command.status = "sent"
        command.executed_at = datetime.now(timezone.utc)
    
    doc = command.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    if doc['executed_at']:
        doc['executed_at'] = doc['executed_at'].isoformat()
        
    await db.iot_commands.insert_one(doc)
    return command

# User Management
@api_router.post("/users/register", response_model=User)
async def register_user(user: User):
    doc = user.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.users.insert_one(doc)
    return user

@api_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@api_router.get("/health/dashboard/{user_id}")
async def get_health_dashboard(user_id: str):
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Parallel execution for speed
    appts_task = db.appointments.find({"user_id": user_id}, {"_id": 0}).to_list(5)
    rx_task = db.prescriptions.find({"user_id": user_id}, {"_id": 0}).to_list(5)
    
    appointments, prescriptions = await asyncio.gather(appts_task, rx_task)
    
    return {
        "user": user,
        "appointments": appointments,
        "prescriptions": prescriptions,
        "vitals": {"heart_rate": 75, "bp": "120/80"} # Mock data
    }

# Register Router
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)