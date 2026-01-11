from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import httpx
import json
import base64
import asyncio
import bcrypt
from twilio.rest import Client
import paho.mqtt.client as mqtt

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

from middleware.cors import setup_cors
from middleware.logging import log_requests
from middleware.auth import create_access_token, get_current_user, optional_auth
from utils.validators import validate_email, validate_password
from utils.responses import success_response, error_response

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mongo_client: Optional[AsyncIOMotorClient] = None
db = None
mqtt_client: Optional[mqtt.Client] = None
twilio_client: Optional[Client] = None

# ==================== LIFESPAN / STARTUP / SHUTDOWN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mongo_client, db, mqtt_client, twilio_client

    # --- Startup ---
    # Mongo
    try:
        mongo_url = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
        mongo_client = AsyncIOMotorClient(mongo_url)
        db = mongo_client[os.environ.get('DB_NAME', 'mediverse')]
        logger.info("✅ MongoDB Connected")
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")

    # Twilio
    try:
        if os.environ.get('TWILIO_ACCOUNT_SID') and os.environ.get('TWILIO_AUTH_TOKEN'):
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

    yield  # Application runs here

    # --- Shutdown ---
    if mongo_client:
        mongo_client.close()
    if mqtt_client:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        except Exception:
            pass
    logger.info("🛑 Services Shutdown")

app = FastAPI(title="Mediverse.AI API", version="1.0.0", lifespan=lifespan)

# Create router before any routes are declared
api_router = APIRouter(prefix="/api")

# Setup CORS & logging middleware
setup_cors(app)
app.middleware("http")(log_requests)

# ==================== MODELS ====================
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: EmailStr
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
    
    doctor_id: Optional[str] = None
    type: Optional[str] = "video"

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
    
    medication_name: Optional[str] = None
    doctor_id: Optional[str] = None
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    refills_left: Optional[int] = 0
    status: Optional[str] = "active"

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

# ==================== AUTH MODELS ====================
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    phone: str
    blood_type: Optional[str] = None
    allergies: Optional[List[str]] = []

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

# ==================== HELPERS / SERVICES ====================
async def deepgram_transcribe_rest(audio_data: bytes) -> Dict[str, Any]:
    """Transcribe audio using Deepgram REST API (for files/chunks)"""
    logger.info(f"🎤 Deepgram Transcription Request - Audio size: {len(audio_data)} bytes")
    
    if not os.environ.get('DEEPGRAM_API_KEY'):
        logger.warning("⚠️ DEEPGRAM_API_KEY not configured - returning placeholder")
        return {"transcript": "Deepgram not configured", "raw": {}}
        
    try:
        url = f"{os.environ.get('DEEPGRAM_URL', 'https://api.deepgram.com/v1/listen')}?model=nova-2&smart_format=true"
        headers = {
            "Authorization": f"Token {os.environ.get('DEEPGRAM_API_KEY','')}",
            "Content-Type": "audio/wav"
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info("🚀 Sending audio to Deepgram API...")
            response = await client.post(url, headers=headers, content=audio_data)
            response.raise_for_status()
            logger.info(f"✅ Deepgram Response Status: {response.status_code}")
            result = response.json()
            transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
            logger.info(f"✅ Deepgram Transcription Success: {transcript[:100]}...")
            return {"transcript": transcript, "raw": result}
    except Exception as e:
        logger.error(f"❌ Deepgram Transcription Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

async def groq_intent_analysis(text: str, context: Dict) -> Dict[str, Any]:
    """Analyze intent using Groq AI"""
    logger.info(f"🤖 Groq Analysis Request - Text: {text[:100]}...")
    
    if not os.environ.get('GROQ_API_KEY'):
        logger.warning("⚠️ GROQ_API_KEY not configured - returning fallback response")
        return {
            "intent": "health_query",
            "action": "provide_information",
            "entities": {},
            "urgency": "low",
            "response": "I'm here to help. Please ensure Groq API is configured for advanced AI responses."
        }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY','')}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""You are Mediverse.AI, an Indian healthcare assistant. Analyze this user command:
User: "{text}"
Response JSON: {{
  "intent": "<book_appointment|emergency|iot_command|prescription_refill|health_query>",
  "action": "<specific action>",
  "entities": {{"key": "value"}},
  "urgency": "<low|medium|high|critical>",
  "response": "<natural language response in Indian English>"
}}"""
            
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            logger.info(f"🚀 Sending request to Groq API...")
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers
            )
            
            logger.info(f"✅ Groq Response Status: {resp.status_code}")
            
            if resp.status_code != 200:
                logger.error(f"❌ Groq API Error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=500, detail=f"Groq API error: {resp.status_code}")
            
            data = resp.json()
            content = data['choices'][0]['message']['content']
            logger.info(f"✅ Groq AI Response received: {content[:200]}...")
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(f"✅ Parsed Intent: {result.get('intent')}, Urgency: {result.get('urgency')}")
                return result
            else:
                logger.warning("⚠️ Could not parse JSON from Groq response, using fallback")
                return {
                    "intent": "health_query",
                    "action": "provide_information",
                    "entities": {},
                    "urgency": "low",
                    "response": content
                }
    except Exception as e:
        logger.error(f"❌ Groq Analysis Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

async def trigger_n8n_webhook(flow: str, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        base_url = os.environ.get('N8N_WEBHOOK_BASE', 'http://localhost:5678/webhook')
        webhook_url = f"{base_url}/{flow}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(webhook_url, json=data)
            return {"status": "triggered", "flow": flow}
    except Exception as e:
        logger.warning(f"n8n webhook error: {e}")
        return {"status": "mocked", "message": "Triggered (Mock)"}

def publish_iot_command(device: str, action: str) -> bool:
    global mqtt_client
    if not mqtt_client:
        logger.warning("MQTT client not initialized, cannot publish IoT command.")
        return False
    try:
        topic = f"{os.environ.get('MQTT_TOPIC', 'mediverse/iot/commands')}/{device}"
        payload = json.dumps({"action": action, "ts": datetime.now(timezone.utc).isoformat()})
        info = mqtt_client.publish(topic, payload)
        # paho-mqtt publish returns a tuple in some versions; check rc attribute safely
        rc = getattr(info, "rc", None)
        if rc is None and isinstance(info, tuple) and len(info) > 0:
            rc = info[0]
        return rc == 0
    except Exception as e:
        logger.error(f"MQTT Publish Error: {e}")
        return False

# ==================== ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "Mediverse.AI API - Intelligent Health Ecosystem", "status": "active"}

# Voice Recognition Routes
@api_router.post("/voice/transcribe")
async def transcribe_voice(request: VoiceTranscribeRequest):
    if not os.environ.get('DEEPGRAM_API_KEY'):
        raise HTTPException(status_code=503, detail="Deepgram API not configured")
        
    if request.audio_base64:
        audio_data = base64.b64decode(request.audio_base64)
    elif request.audio_url:
        async with httpx.AsyncClient() as client:
            resp = await client.get(request.audio_url)
            audio_data = resp.content
    else:
        raise HTTPException(status_code=400, detail="Provide audio source")
    
    result = await deepgram_transcribe_rest(audio_data)
    try:
        transcript = result['transcript']
    except (KeyError, IndexError, TypeError):
        transcript = ""
        
    return {"transcript": transcript, "raw": result}

@api_router.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """Real-time bi-directional voice streaming."""
    await websocket.accept()
    
    try:
        dg_api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not dg_api_key:
            await websocket.send_json({"error": "Deepgram API key not configured"})
            await websocket.close()
            return

        deepgram = DeepgramClient(dg_api_key)
        # Using the Deepgram websocket helper; API surface may vary by package version
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        async def on_message(result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                if sentence:
                    await websocket.send_json({
                        "transcript": sentence,
                        "is_final": getattr(result, "is_final", False)
                    })
            except Exception as e:
                logger.error("Error in Deepgram message handler: %s", e)

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2", 
            language="en-IN", 
            smart_format=True,
            interim_results=True,
        )

        if await dg_connection.start(options) is False:
            logger.error("Failed to connect to Deepgram")
            await websocket.close()
            return

        try:
            while True:
                # receive audio bytes from client and forward to deepgram
                data = await websocket.receive_bytes()
                await dg_connection.send(data)
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await dg_connection.finish()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"WS Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass

# AI Intent Routes
@api_router.post("/ai/intent")
async def analyze_intent(request: AIIntentRequest):
    if not os.environ.get('GROQ_API_KEY'):
        raise HTTPException(status_code=503, detail="Groq API not configured")
    return await groq_intent_analysis(request.text, request.user_context)

@api_router.post("/ai/chat")
async def ai_chat(request: AIIntentRequest):
    if not os.environ.get('GROQ_API_KEY'):
        raise HTTPException(status_code=503, detail="Groq API not configured")
        
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY','')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": request.text}
                ]
            }
            response = await client.post(
                os.environ.get('GROQ_URL', 'https://api.groq.com/openai/v1/chat/completions'),
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return {"response": data.get('choices', [{}])[0].get('message', {}).get('content', '')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Appointment Routes
@api_router.post("/appointments/book", response_model=Appointment)
async def book_appointment(
    request: AppointmentBookRequest,
    current_user: dict = Depends(get_current_user)
):
    # Ensure user can only book for themselves
    if request.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Create Appointment object, setting default values for new fields if not provided
    appointment_data = request.model_dump()
    appointment_data.setdefault('doctor_id', None) # Or fetch from a doctor lookup if available
    appointment_data.setdefault('type', 'video') # Default type
    
    appointment = Appointment(**appointment_data)
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
async def get_appointments(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    # Ensure user can only view their own appointments
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    cursor = db.appointments.find({"user_id": user_id}, {"_id": 0})
    appointments = await cursor.to_list(length=100)
    
    for apt in appointments:
        apt['doctorName'] = apt.get('doctor_name', apt.get('doctorName', 'Doctor'))
        apt['doctorId'] = apt.get('doctor_id', apt.get('doctorId', ''))
        apt['userId'] = apt.get('user_id', apt.get('userId', ''))
        apt['createdAt'] = apt.get('created_at', apt.get('createdAt', ''))
        if 'type' not in apt:
            apt['type'] = 'video'
    
    return appointments

# Emergency Routes
@api_router.post("/emergency/trigger", response_model=EmergencyLog)
async def trigger_emergency(request: EmergencyTriggerRequest):
    """Trigger emergency alert (Indian emergency services integration)"""
    logger.info(f"🚨 EMERGENCY TRIGGERED - User: {request.user_id}, Type: {request.type}, Location: {request.location}")
    
    # Validate emergency type
    valid_types = ["heart_attack", "fall", "medical_emergency", "accident", "breathing_difficulty"]
    if request.type not in valid_types:
        logger.warning(f"⚠️ Invalid emergency type: {request.type}")
        raise HTTPException(status_code=422, detail=f"Invalid emergency type. Must be one of: {', '.join(valid_types)}")
    
    emergency = EmergencyLog(**request.model_dump())
    doc = emergency.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    try:
        await db.emergencies.insert_one(doc)
        logger.info(f"✅ Emergency logged to database: {emergency.id}")
    except Exception as e:
        logger.error(f"❌ Failed to log emergency: {str(e)}")

    # Fetch user details
    user = await db.users.find_one({"id": request.user_id})
    contact = user.get('emergency_contact') if user else None
    user_name = user.get('name', 'User') if user else 'User'

    # Send SMS to emergency contact
    if twilio_client and contact:
        try:
            logger.info(f"📱 Sending emergency SMS to: {contact}")
            message = twilio_client.messages.create(
                body=f"🚨 आपातकाल / EMERGENCY: {user_name} को मदद की जरूरत है। {request.type}. स्थान: {request.location or 'Unknown'}",
                from_=os.environ.get('TWILIO_PHONE_NUMBER'),
                to=contact
            )
            emergency.responders_notified.append("sms_sent")
            logger.info(f"✅ Emergency SMS sent successfully: {message.sid}")
        except Exception as e:
            logger.error(f"❌ Emergency SMS failed: {str(e)}")
            emergency.responders_notified.append("sms_failed")
    else:
        logger.warning("⚠️ Twilio not configured or no emergency contact found")

    # Trigger IoT devices and webhooks
    try:
        asyncio.create_task(trigger_n8n_webhook("emergency", doc))
        publish_iot_command("care_bot", "emergency_assist")
        logger.info("✅ Emergency IoT commands triggered")
    except Exception as e:
        logger.error(f"❌ IoT trigger failed: {str(e)}")
    
    logger.info(f"✅ Emergency response complete for: {emergency.id}")
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
    if doc.get('executed_at'):
        doc['executed_at'] = doc['executed_at'].isoformat()
        
    await db.iot_commands.insert_one(doc)
    return command

# IoT Device Command Endpoint
@api_router.post("/iot/devices/{device_id}/command")
async def send_device_command(
    device_id: str,
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Send command to IoT device"""
    try:
        # Verify device belongs to user
        device = await db.iot_devices.find_one({"id": device_id})
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        
        if device.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        command = request.get("command", "")
        logger.info(f"🤖 IoT Command - Device: {device_id}, Command: {command}")
        
        # Publish command via MQTT/HTTP
        success = publish_iot_command(device.get("name", "device"), command)
        
        # Log command
        command_doc = {
            "id": str(uuid.uuid4()),
            "device_id": device_id,
            "user_id": current_user["id"],
            "command": command,
            "status": "sent" if success else "failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "executed_at": datetime.now(timezone.utc).isoformat() if success else None
        }
        
        await db.iot_commands.insert_one(command_doc)
        
        return {"success": success, "message": "Command sent successfully" if success else "Command failed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ IoT command failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Command failed: {str(e)}")

# User Management
@api_router.post("/users/register", response_model=User)
async def register_user(user: User):
    # Check if user already exists
    existing = await db.users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="User with this email already exists")
        
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
    appts_task = db.appointments.find({"user_id": user_id}, {"_id": 0}).sort("date", -1).to_list(5)
    rx_task = db.prescriptions.find({"user_id": user_id}, {"_id": 0}).sort("startDate", -1).to_list(5)
    metrics_task = db.health_metrics.find_one({"user_id": user_id}, {"_id": 0})
    activity_task = db.activity_logs.find({"user_id": user_id}, {"_id": 0}).sort("timestamp", -1).to_list(10)
    
    appointments, prescriptions, metrics, activity = await asyncio.gather(
        appts_task, rx_task, metrics_task, activity_task
    )
    
    for apt in appointments or []:
        apt['doctorName'] = apt.get('doctor_name', apt.get('doctorName', 'Doctor'))
        apt['doctorId'] = apt.get('doctor_id', apt.get('doctorId', ''))
        apt['userId'] = apt.get('user_id', apt.get('userId', ''))
        apt['createdAt'] = apt.get('created_at', apt.get('createdAt', ''))
        if 'type' not in apt:
            apt['type'] = 'video'
    
    for rx in prescriptions or []:
        rx['medicationName'] = rx.get('medication', rx.get('medicationName', ''))
        rx['doctorName'] = rx.get('doctor', rx.get('doctorName', 'Doctor'))
        rx['doctorId'] = rx.get('doctor_id', rx.get('doctorId', ''))
        rx['startDate'] = rx.get('startDate', datetime.utcnow().isoformat())
        rx['endDate'] = rx.get('endDate', (datetime.utcnow() + timedelta(days=7)).isoformat())
        if 'refillsLeft' not in rx:
            rx['refillsLeft'] = rx.get('refills_left', 0)
        if 'status' not in rx:
            rx['status'] = 'active'
    
    health_score = {
        "score": 85,
        "trend": "up",
        "lastUpdated": datetime.utcnow().isoformat(),
        "factors": {
            "activity": 80,
            "vitals": 85,
            "lifestyle": 90
        }
    }
    
    if not metrics:
        metrics = {
            "heart_rate": 72,
            "blood_pressure": "120/80",
            "temperature": 37.0,
            "steps_today": 8500,
            "oxygen_level": 98,
            "sleep_hours": 7.5,
            "weight": None,
            "bmi": None
        }
    
    return {
        "user": user,
        "healthScore": health_score,
        "metrics": metrics,
        "appointments": appointments or [],
        "prescriptions": prescriptions or [],
        "recentActivity": activity or []
    }

# Doctors Endpoint
@api_router.get("/doctors")
async def get_doctors(specialty: Optional[str] = None):
    """Get all doctors, optionally filtered by specialty (Indian medical system)"""
    logger.info(f"👨‍⚕️ Fetching doctors - Specialty filter: {specialty or 'All'}")
    
    query = {}
    if specialty and specialty != "all":
        query["specialty"] = {"$regex": specialty, "$options": "i"}
    
    cursor = db.doctors.find(query, {"_id": 0})
    doctors = await cursor.to_list(length=100)
    
    # Ensure all doctors have required fields with defaults
    for doctor in doctors:
        doctor.setdefault("id", str(uuid.uuid4()))
        doctor.setdefault("name", "Dr. Unknown")
        doctor.setdefault("specialty", "General Physician")
        doctor.setdefault("experience", 0)
        doctor.setdefault("rating", 0.0)
        doctor.setdefault("reviews", 0)
        doctor.setdefault("availability", [])
        doctor.setdefault("languages", ["English", "Hindi"])
        doctor.setdefault("consultation_fee", 500)
        doctor.setdefault("hospital", "Unknown Hospital")
        doctor.setdefault("location", "India")
    
    logger.info(f"✅ Fetched {len(doctors)} doctors")
    return doctors

# Doctor Slots Endpoint
@api_router.get("/doctors/{doctor_id}/slots")
async def get_doctor_slots(
    doctor_id: str,
    date: str,
    current_user: dict = Depends(get_current_user)
):
    """Get available time slots for a doctor on a specific date"""
    logger.info(f"📅 Fetching slots for doctor: {doctor_id}, date: {date}")
    
    # Check if doctor exists
    doctor = await db.doctors.find_one({"id": doctor_id}, {"_id": 0})
    if not doctor:
        logger.warning(f"⚠️ Doctor not found: {doctor_id}")
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    # Get existing appointments for this doctor on this date
    existing_appointments = await db.appointments.find({
        "doctor_id": doctor_id,
        "date": date,
        "status": {"$in": ["scheduled", "confirmed"]}
    }).to_list(length=100)
    
    booked_times = {apt["time"] for apt in existing_appointments}
    
    # Generate time slots (9 AM to 6 PM, 30-minute intervals)
    # Indian working hours: 9:00 AM to 6:00 PM
    time_slots = []
    for hour in range(9, 18):
        for minute in [0, 30]:
            time_str = f"{hour:02d}:{minute:02d}"
            time_slots.append({
                "id": f"{doctor_id}-{date}-{time_str}",
                "time": time_str,
                "available": time_str not in booked_times,
                "date": date
            })
    
    logger.info(f"✅ Generated {len(time_slots)} slots, {len(booked_times)} booked")
    return time_slots

# Medications Endpoint
@api_router.get("/medications")
async def get_medications(category: Optional[str] = None, search: Optional[str] = None):
    """Get medications with optional category filter and search"""
    query = {}
    if category and category != "all":
        query["category"] = category
    if search:
        query["name"] = {"$regex": search, "$options": "i"}
    
    cursor = db.medications.find(query, {"_id": 0})
    medications = await cursor.to_list(length=100)
    return medications

# Pharmacy Order Endpoint
@api_router.post("/pharmacy/orders")
async def create_pharmacy_order(
    user_id: str,
    items: List[Dict[str, Any]],
    shipping_address: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Create pharmacy order"""
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    order_doc = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "items": items,
        "shipping_address": shipping_address,
        "status": "pending",
        "total_amount": sum(item.get("price", 0) * item.get("quantity", 1) for item in items),
        "created_at": datetime.utcnow().isoformat()
    }
    
    await db.orders.insert_one(order_doc)
    return order_doc

@api_router.get("/pharmacy/orders/{user_id}")
async def get_user_orders_pharmacy(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get user's pharmacy orders"""
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    cursor = db.orders.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1)
    orders = await cursor.to_list(length=50)
    return orders

@api_router.get("/pharmacy/orders/track/{order_id}")
async def track_order(order_id: str, current_user: dict = Depends(get_current_user)):
    """Track order status"""
    order = await db.orders.find_one({"id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Verify user owns this order
    if order["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    return order

# IoT Devices Endpoint
@api_router.get("/iot/devices/{user_id}")
async def get_user_devices(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get user's IoT devices"""
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    cursor = db.iot_devices.find({"user_id": user_id}, {"_id": 0})
    devices = await cursor.to_list(length=50)
    return devices

# Health Metrics Endpoint
@api_router.get("/health/metrics/{user_id}")
async def get_health_metrics(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get user's health metrics"""
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    metrics = await db.health_metrics.find_one({"user_id": user_id}, {"_id": 0})
    if not metrics:
        # Return default metrics if none exist
        return {
            "user_id": user_id,
            "health_score": 75,
            "heart_rate": 72,
            "blood_pressure": "120/80",
            "steps": 0,
            "calories": 0,
            "last_updated": datetime.utcnow().isoformat()
        }
    return metrics

# Voice Command Endpoints
@api_router.post("/voice/query")
async def process_voice_query(request: AIIntentRequest):
    """Process voice command query"""
    if not os.environ.get('GROQ_API_KEY'):
        return {
            "id": str(uuid.uuid4()),
            "command": request.text,
            "response": "Voice assistant is not configured. Please add GROQ_API_KEY to environment variables.",
            "timestamp": datetime.utcnow().isoformat(),
            "successful": False
        }
    
    try:
        result = await groq_intent_analysis(request.text, request.user_context)
        
        # Store command history
        command_doc = {
            "id": str(uuid.uuid4()),
            "command": request.text,
            "query": request.text,
            "intent": result.get("intent"),
            "action": result.get("action"),
            "response": result.get("response"),
            "timestamp": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "successful": True
        }
        
        await db.voice_commands.insert_one(command_doc.copy())
        
        return command_doc
    except Exception as e:
        logger.error(f"Voice query failed: {str(e)}")
        return {
            "id": str(uuid.uuid4()),
            "command": request.text,
            "response": "I'm having trouble processing your request right now. Please try again later.",
            "timestamp": datetime.utcnow().isoformat(),
            "successful": False
        }

@api_router.get("/voice/history/{user_id}")
async def get_voice_history(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get voice command history"""
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    cursor = db.voice_commands.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1)
    history = await cursor.to_list(length=50)
    return history

# Diagnostics Endpoints
class DiagnosticsRequest(BaseModel):
    user_id: str
    symptoms: List[str]

@api_router.post("/diagnostics/analyze")
async def analyze_symptoms(
    request: DiagnosticsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze symptoms using AI (Indian healthcare context)"""
    logger.info(f"🏥 Diagnostics Request - User: {request.user_id}, Symptoms: {request.symptoms}")
    
    if request.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Validate symptoms
    if not request.symptoms or len(request.symptoms) == 0:
        logger.warning("⚠️ No symptoms provided")
        raise HTTPException(status_code=422, detail="Please provide at least one symptom")
    
    # AI-powered analysis
    symptom_text = ", ".join(request.symptoms)
    prompt = f"""You are an Indian healthcare AI assistant. Analyze these symptoms: {symptom_text}. 
Provide a JSON response with:
1. possibleConditions: array of objects with name, probability (0-1), description
2. urgencyLevel: one of "low", "medium", "high", "critical"
3. recommendations: array of strings
4. severity: one of "mild", "moderate", "severe"

Example format:
{{
  "possibleConditions": [{{"name": "Common Cold", "probability": 0.7, "description": "Viral infection"}}],
  "urgencyLevel": "low",
  "recommendations": ["Rest", "Stay hydrated"],
  "severity": "mild"
}}"""
    
    try:
        logger.info("🤖 Analyzing symptoms with Groq AI...")
        result = await groq_intent_analysis(prompt, {"symptoms": request.symptoms})
        
        # Parse the AI response
        response_text = result.get("response", "{}")
        
        # Try to extract JSON from response
        import json
        import re
        
        # Look for JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        parsed_analysis = {}
        
        if json_match:
            try:
                parsed_analysis = json.loads(json_match.group())
            except:
                logger.warning("Failed to parse AI JSON response")
        
        # Build structured response
        possible_conditions = parsed_analysis.get("possibleConditions", [
            {
                "name": "General Illness",
                "probability": 0.5,
                "description": "Based on the symptoms provided, this appears to be a general illness. Please consult a healthcare professional for proper diagnosis."
            }
        ])
        
        urgency_level = parsed_analysis.get("urgencyLevel", "medium")
        recommendations = parsed_analysis.get("recommendations", [
            "परामर्श के लिए डॉक्टर से मिलें (Consult a doctor)",
            "Monitor your symptoms closely",
            "Stay hydrated and rest"
        ])
        severity = parsed_analysis.get("severity", "moderate")
        
        analysis_doc = {
            "id": str(uuid.uuid4()),
            "symptoms": request.symptoms,
            "possibleConditions": possible_conditions,
            "recommendations": recommendations,
            "urgencyLevel": urgency_level,
            "severity": severity,
            "analyzedAt": datetime.now(timezone.utc).isoformat(),
            "user_id": request.user_id
        }
        
        # Save to database
        # Renamed collection to 'symptom_analyses' for clarity and consistency
        result = await db.symptom_analyses.insert_one(analysis_doc.copy()) 
        logger.info(f"✅ Diagnostics analysis saved: {analysis_doc['id']}")
        
        # Return without MongoDB's _id
        return analysis_doc
    except Exception as e:
        logger.error(f"❌ Symptom analysis failed: {str(e)}")
        # Return a fallback response instead of raising error
        fallback_doc = {
            "id": str(uuid.uuid4()),
            "symptoms": request.symptoms,
            "possibleConditions": [
                {
                    "name": "General Health Concern",
                    "probability": 0.5,
                    "description": "Based on your symptoms, we recommend consulting with a healthcare professional for proper evaluation."
                }
            ],
            "recommendations": [
                "परामर्श के लिए डॉक्टर से मिलें (Consult a doctor for consultation)",
                "Monitor your symptoms",
                "Stay hydrated and get adequate rest"
            ],
            "urgencyLevel": "medium",
            "severity": "moderate",
            "analyzedAt": datetime.now(timezone.utc).isoformat(),
            "user_id": request.user_id
        }
        return fallback_doc

@api_router.get("/diagnostics/history/{user_id}")
async def get_diagnostics_history(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get user's symptom analysis history"""
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Querying the 'symptom_analyses' collection
    cursor = db.symptom_analyses.find({"user_id": user_id}, {"_id": 0}).sort("analyzedAt", -1) 
    history = await cursor.to_list(length=50)
    return history

@api_router.get("/diagnostics/guides")
async def get_health_guides(category: Optional[str] = None):
    """Get health guides"""
    query = {}
    if category:
        query["category"] = category
    
    cursor = db.health_guides.find(query, {"_id": 0})
    guides = await cursor.to_list(length=100)
    
    if not guides:
        # Return default guides if none in database
        default_guides = [
            {
                "id": "1",
                "title": "Managing Diabetes",
                "category": "Chronic Conditions",
                "description": "Complete guide to diabetes management",
                "content": "Monitor blood sugar regularly...",
                "author": "Dr. Smith",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "2",
                "title": "Heart Health",
                "category": "Cardiovascular",
                "description": "Tips for maintaining a healthy heart",
                "content": "Exercise regularly, eat healthy...",
                "author": "Dr. Johnson",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        return default_guides
    
    return guides

@api_router.get("/diagnostics/guides/{guide_id}")
async def get_health_guide_by_id(guide_id: str):
    """Get specific health guide"""
    guide = await db.health_guides.find_one({"id": guide_id}, {"_id": 0})
    if not guide:
        raise HTTPException(status_code=404, detail="Guide not found")
    return guide

# ==================== AUTH (register/login/me) ====================
@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    """Register new user with password hashing"""
    # Validate email
    if not validate_email(user_data.email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    # Validate password
    is_valid, error_msg = validate_password(user_data.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Check if user exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
    
    # Create user
    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "password": hashed_password.decode('utf-8'),
        "name": user_data.name,
        "phone": user_data.phone,
        "blood_type": user_data.blood_type,
        "allergies": user_data.allergies,
        "created_at": datetime.utcnow().isoformat()
    }
    
    await db.users.insert_one(user_doc)
    
    # Create access token
    token = create_access_token({"sub": user_id, "email": user_data.email})
    
    # Remove password from response
    user_doc.pop("password", None)
    user_doc.pop("_id", None)
    
    return {
        "token": token,
        "access_token": token,  # Keep for backward compatibility
        "token_type": "bearer",
        "user": user_doc
    }

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    """Login user and return JWT token"""
    # Find user
    user = await db.users.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not bcrypt.checkpw(credentials.password.encode('utf-8'), user["password"].encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create access token
    token = create_access_token({"sub": user["id"], "email": user["email"]})
    
    # Remove password from response
    user.pop("password", None)
    user.pop("_id", None)
    
    return {
        "token": token,
        "access_token": token,  # Keep for backward compatibility
        "token_type": "bearer",
        "user": user
    }

@api_router.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user"""
    user = await db.users.find_one({"id": current_user["id"]}, {"_id": 0, "password": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Health Score Endpoint
@api_router.get("/health/score/{user_id}")
async def get_health_score(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Calculate and return user's health score"""
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    metrics = await db.health_metrics.find_one({"user_id": user_id}, {"_id": 0})
    
    # Calculate score based on available metrics
    if not metrics:
        # Return default healthy score
        return {
            "score": 85,
            "trend": "stable",
            "lastUpdated": datetime.utcnow().isoformat(),
            "factors": {
                "activity": 80,
                "vitals": 85,
                "lifestyle": 90
            }
        }
    
    # Calculate individual factor scores
    activity_score = min(100, metrics.get("steps_today", 0) / 100)
    vitals_score = 85  # Would be calculated from BP, HR, etc
    lifestyle_score = 90  # Would be calculated from sleep, etc
    
    # Calculate overall score
    overall_score = int((activity_score + vitals_score + lifestyle_score) / 3)
    
    # Determine trend (would be calculated from historical data)
    trend = "up"
    
    return {
        "score": overall_score,
        "trend": trend,
        "lastUpdated": datetime.utcnow().isoformat(),
        "factors": {
            "activity": int(activity_score),
            "vitals": int(vitals_score),
            "lifestyle": int(lifestyle_score)
        }
    }

# Register Router with app
app.include_router(api_router)

# Entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
