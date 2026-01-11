# app/main.py

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

# Deepgram / Groq imports — keep as you had them (may require the package names used)
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

# Import custom middleware / utils (assumed to exist)
from middleware.cors import setup_cors
from middleware.logging import log_requests
from middleware.auth import create_access_token, get_current_user, optional_auth
from utils.validators import validate_email, validate_password
from utils.responses import success_response, error_response

# ==================== ENV & LOGGING (move to top so helper functions can use logger) ====================
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== GLOBALS ====================
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

# 4. App Definition (create app with lifespan)
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
    try:
        url = f"{os.environ.get('DEEPGRAM_URL', 'https://api.deepgram.com/v1/listen')}?model=nova-2&smart_format=true"
        headers = {
            "Authorization": f"Token {os.environ.get('DEEPGRAM_API_KEY','')}",
            "Content-Type": "audio/wav"
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
                "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY','')}",
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
                os.environ.get('GROQ_URL', 'https://api.groq.com/openai/v1/chat/completions'),
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            # best-effort: try extracting the content safely
            content = result.get('choices', [{}])[0].get('message', {}).get('content', "{}")
            # if content is already a dict-like string, parse it
            try:
                return json.loads(content)
            except Exception:
                return {"intent": "unknown", "response": content}
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return {"intent": "error", "response": "I'm having trouble thinking right now."}

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
        transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
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
async def get_appointments(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    # Ensure user can only view their own appointments
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
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
                from_=os.environ.get('TWILIO_PHONE_NUMBER'),
                to=contact
            )
            emergency.responders_notified.append("sms_sent")
        except Exception as e:
            logger.error(f"Emergency SMS failed: {e}")

    # Fire-and-forget-ish: don't block on webhooks
    asyncio.create_task(trigger_n8n_webhook("emergency", doc))
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
    if doc.get('executed_at'):
        doc['executed_at'] = doc['executed_at'].isoformat()
        
    await db.iot_commands.insert_one(doc)
    return command

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
    appts_task = db.appointments.find({"user_id": user_id}, {"_id": 0}).to_list(5)
    rx_task = db.prescriptions.find({"user_id": user_id}, {"_id": 0}).to_list(5)
    
    appointments, prescriptions = await asyncio.gather(appts_task, rx_task)
    
    return {
        "user": user,
        "appointments": appointments,
        "prescriptions": prescriptions,
        "vitals": {"heart_rate": 75, "bp": "120/80"}
    }

# Doctors Endpoint
@api_router.get("/doctors")
async def get_doctors(specialty: Optional[str] = None):
    """Get all doctors, optionally filtered by specialty"""
    query = {}
    if specialty and specialty != "all":
        query["specialty"] = {"$regex": specialty, "$options": "i"}
    
    cursor = db.doctors.find(query, {"_id": 0})
    doctors = await cursor.to_list(length=100)
    return doctors

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
            "intent": "unknown",
            "action": "none",
            "response": "Voice assistant is not configured",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    result = await groq_intent_analysis(request.text, request.user_context)
    
    # Store command history
    command_doc = {
        "id": str(uuid.uuid4()),
        "query": request.text,
        "intent": result.get("intent"),
        "action": result.get("action"),
        "response": result.get("response"),
        "created_at": datetime.utcnow().isoformat()
    }
    
    await db.voice_commands.insert_one(command_doc)
    
    return command_doc

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
@api_router.post("/diagnostics/analyze")
async def analyze_symptoms(
    symptoms: List[str],
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Analyze symptoms using AI"""
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Use Groq AI for symptom analysis
    if not os.environ.get('GROQ_API_KEY'):
        # Return mock analysis if no API key
        return {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "symptoms": symptoms,
            "analysis": "Please consult a healthcare professional for accurate diagnosis.",
            "severity": "moderate",
            "recommendations": ["Schedule a consultation", "Monitor symptoms"],
            "created_at": datetime.utcnow().isoformat()
        }
    
    # AI-powered analysis
    symptom_text = ", ".join(symptoms)
    prompt = f"Analyze these symptoms: {symptom_text}. Provide severity and recommendations."
    
    try:
        result = await groq_intent_analysis(prompt, {})
        analysis_doc = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "symptoms": symptoms,
            "analysis": result.get("response", "Analysis unavailable"),
            "severity": result.get("urgency", "moderate"),
            "recommendations": ["Consult a healthcare professional"],
            "created_at": datetime.utcnow().isoformat()
        }
        
        await db.symptom_analyses.insert_one(analysis_doc)
        return analysis_doc
    except Exception as e:
        logger.error(f"Symptom analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@api_router.get("/diagnostics/history/{user_id}")
async def get_diagnostics_history(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get user's symptom analysis history"""
    if user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    cursor = db.symptom_analyses.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1)
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

# Register Router with app
app.include_router(api_router)

# Entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
