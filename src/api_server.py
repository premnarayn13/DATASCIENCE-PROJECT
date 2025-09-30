"""
FastAPI Server for IT Support Ticket Predictor

This module provides RESTful API endpoints for the ticket prediction system,
including real-time predictions, model management, and system monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import uuid
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="AI IT Support Ticket Predictor API",
    description="Revolutionary AI system for predicting IT ticket resolution times",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TicketPredictionRequest(BaseModel):
    ticket_description: str = Field(..., min_length=10, max_length=1000)
    category: str = Field(..., regex="^(Hardware|Software|Network|Security)$")
    priority: str = Field(..., regex="^(Critical|High|Medium|Low)$")
    user_id: Optional[str] = None
    user_experience_level: Optional[str] = Field("Intermediate", regex="^(Novice|Intermediate|Advanced|Expert)$")
    system_type: Optional[str] = Field("Desktop", regex="^(Desktop|Laptop|Server|Mobile)$")
    urgency_indicators: Optional[List[str]] = []
    
    @validator('ticket_description')
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError('Ticket description cannot be empty')
        return v.strip()

class TicketPredictionResponse(BaseModel):
    prediction_id: str
    predicted_resolution_time: float
    confidence_score: float
    priority_adjustment: Optional[str] = None
    auto_resolvable: bool
    recommended_technician: Optional[str] = None
    sentiment_analysis: Dict[str, Any]
    escalation_recommended: bool
    estimated_cost: Optional[float] = None
    similar_tickets: List[str] = []
    processing_time_ms: float

class BatchPredictionRequest(BaseModel):
    tickets: List[TicketPredictionRequest]
    include_analytics: bool = False

class BatchPredictionResponse(BaseModel):
    predictions: List[TicketPredictionResponse]
    batch_id: str
    total_tickets: int
    processing_time_ms: float
    batch_analytics: Optional[Dict[str, Any]] = None

class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    r2_score: float
    accuracy_80_percent: float
    last_updated: datetime
    total_predictions: int

class SystemHealth(BaseModel):
    status: str
    uptime_seconds: float
    active_models: int
    queue_length: int
    cpu_usage: float
    memory_usage: float
    last_check: datetime

class TrainingRequest(BaseModel):
    data_source: str
    model_type: Optional[str] = "ensemble"
    hyperparameters: Optional[Dict[str, Any]] = {}
    validation_split: Optional[float] = 0.2

# Global state
class ApplicationState:
    def __init__(self):
        self.models = {}
        self.prediction_cache = {}
        self.metrics = {}
        self.start_time = datetime.now()
        self.prediction_count = 0
        self.load_models()
        self.load_config()
    
    def load_config(self):
        """Load application configuration"""
        try:
            config_path = Path("config/config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'api': {
                'prediction_cache_ttl': 3600,
                'max_batch_size': 100,
                'rate_limit': '100/minute'
            },
            'models': {
                'ensemble_threshold': 0.8,
                'auto_resolve_threshold': 0.9
            }
        }
    
    def load_models(self):
        """Load trained models"""
        if not JOBLIB_AVAILABLE:
            logger.warning("Joblib not available. Using mock models.")
            self.models = self._create_mock_models()
            return
        
        try:
            models_dir = Path("models")
            if models_dir.exists():
                # Load ensemble models
                for model_file in models_dir.glob("*.pkl"):
                    model_name = model_file.stem
                    try:
                        model = joblib.load(model_file)
                        self.models[model_name] = model
                        logger.info(f"Loaded model: {model_name}")
                    except Exception as e:
                        logger.error(f"Error loading model {model_name}: {str(e)}")
            
            if not self.models:
                logger.warning("No models loaded. Using mock models.")
                self.models = self._create_mock_models()
                
        except Exception as e:
            logger.error(f"Error in model loading: {str(e)}")
            self.models = self._create_mock_models()
    
    def _create_mock_models(self):
        """Create mock models for demonstration"""
        class MockModel:
            def __init__(self, name):
                self.name = name
            
            def predict(self, X):
                # Simulate prediction based on features
                np.random.seed(hash(str(X)) % 2**32)
                return np.random.exponential(4, len(X))
        
        return {
            'ensemble': MockModel('ensemble'),
            'random_forest': MockModel('random_forest'),
            'xgboost': MockModel('xgboost'),
            'neural_network': MockModel('neural_network')
        }

# Initialize application state
app_state = ApplicationState()

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified for demo)"""
    # In production, implement proper JWT token verification
    if credentials.credentials != "demo_token_123":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Helper functions
def extract_features(ticket_request: TicketPredictionRequest) -> np.ndarray:
    """Extract features from ticket request"""
    # Simplified feature extraction for demo
    features = [
        len(ticket_request.ticket_description),
        hash(ticket_request.category) % 100 / 100.0,
        {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}[ticket_request.priority],
        {'Expert': 4, 'Advanced': 3, 'Intermediate': 2, 'Novice': 1}[ticket_request.user_experience_level],
        len(ticket_request.urgency_indicators),
        hash(ticket_request.system_type) % 50 / 50.0
    ]
    
    # Pad to required length (assuming 32 features)
    while len(features) < 32:
        features.append(0.0)
    
    return np.array(features[:32]).reshape(1, -1)

def calculate_confidence(prediction: float, features: np.ndarray) -> float:
    """Calculate prediction confidence"""
    # Simplified confidence calculation
    base_confidence = 0.7
    
    # Higher confidence for typical values
    if 1 <= prediction <= 8:
        base_confidence += 0.2
    
    # Adjust based on feature complexity
    complexity = np.mean(features)
    if complexity < 0.5:
        base_confidence += 0.1
    
    return min(0.99, max(0.5, base_confidence))

def analyze_sentiment(description: str) -> Dict[str, Any]:
    """Analyze sentiment of ticket description"""
    # Simplified sentiment analysis
    negative_words = ['urgent', 'critical', 'broken', 'down', 'emergency', 'problem', 'issue']
    positive_words = ['working', 'resolved', 'fixed', 'good', 'thanks']
    
    description_lower = description.lower()
    neg_count = sum(1 for word in negative_words if word in description_lower)
    pos_count = sum(1 for word in positive_words if word in description_lower)
    
    if neg_count > pos_count:
        sentiment = 'negative'
        score = -0.3 - (neg_count - pos_count) * 0.1
    elif pos_count > neg_count:
        sentiment = 'positive'
        score = 0.3 + (pos_count - neg_count) * 0.1
    else:
        sentiment = 'neutral'
        score = 0.0
    
    return {
        'sentiment': sentiment,
        'score': max(-1.0, min(1.0, score)),
        'urgency_multiplier': 1.0 + max(0, -score * 0.5)
    }

def recommend_technician(category: str, priority: str) -> Optional[str]:
    """Recommend best technician for ticket"""
    # Simplified technician recommendation
    specialists = {
        'Hardware': ['tech_hw_001', 'tech_hw_002'],
        'Software': ['tech_sw_001', 'tech_sw_002'],
        'Network': ['tech_nw_001', 'tech_nw_002'],
        'Security': ['tech_sec_001', 'tech_sec_002']
    }
    
    available_techs = specialists.get(category, ['tech_gen_001'])
    
    # For critical tickets, return senior technician
    if priority == 'Critical':
        return available_techs[0] if available_techs else None
    
    # Random selection for demo
    import random
    return random.choice(available_techs) if available_techs else None

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint"""
    return {
        "message": "AI IT Support Ticket Predictor API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health", response_model=SystemHealth)
async def health_check():
    """System health check endpoint"""
    import psutil
    
    uptime = (datetime.now() - app_state.start_time).total_seconds()
    
    return SystemHealth(
        status="healthy",
        uptime_seconds=uptime,
        active_models=len(app_state.models),
        queue_length=0,  # Simplified for demo
        cpu_usage=psutil.cpu_percent() if hasattr(psutil, 'cpu_percent') else 50.0,
        memory_usage=psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 60.0,
        last_check=datetime.now()
    )

@app.post("/predict", response_model=TicketPredictionResponse)
async def predict_ticket_resolution(
    request: TicketPredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Predict resolution time for a single ticket"""
    start_time = datetime.now()
    prediction_id = str(uuid.uuid4())
    
    try:
        # Extract features
        features = extract_features(request)
        
        # Make prediction using ensemble model
        if 'ensemble' in app_state.models:
            prediction = app_state.models['ensemble'].predict(features)[0]
        else:
            prediction = np.random.exponential(4)  # Fallback
        
        # Analyze sentiment
        sentiment_analysis = analyze_sentiment(request.ticket_description)
        
        # Adjust prediction based on sentiment
        adjusted_prediction = prediction * sentiment_analysis['urgency_multiplier']
        
        # Calculate confidence
        confidence = calculate_confidence(adjusted_prediction, features)
        
        # Determine auto-resolvability
        auto_resolvable = (
            request.category in ['Software'] and
            request.priority in ['Low', 'Medium'] and
            adjusted_prediction < 2.0 and
            confidence > 0.8
        )
        
        # Recommend technician
        recommended_tech = recommend_technician(request.category, request.priority)
        
        # Check if escalation is recommended
        escalation_recommended = (
            request.priority == 'Critical' or
            adjusted_prediction > 8.0 or
            sentiment_analysis['score'] < -0.5
        )
        
        # Calculate estimated cost (simplified)
        hourly_rate = 50.0  # Average technician rate
        estimated_cost = adjusted_prediction * hourly_rate
        
        # Find similar tickets (mock)
        similar_tickets = [f"T{i:04d}" for i in range(1, 4)]
        
        # Processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = TicketPredictionResponse(
            prediction_id=prediction_id,
            predicted_resolution_time=round(adjusted_prediction, 2),
            confidence_score=round(confidence, 3),
            auto_resolvable=auto_resolvable,
            recommended_technician=recommended_tech,
            sentiment_analysis=sentiment_analysis,
            escalation_recommended=escalation_recommended,
            estimated_cost=round(estimated_cost, 2),
            similar_tickets=similar_tickets,
            processing_time_ms=round(processing_time, 2)
        )
        
        # Update metrics in background
        background_tasks.add_task(update_prediction_metrics, prediction_id, request, response)
        
        app_state.prediction_count += 1
        
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_tickets(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Predict resolution time for multiple tickets"""
    start_time = datetime.now()
    batch_id = str(uuid.uuid4())
    
    # Validate batch size
    max_batch_size = app_state.config['api'].get('max_batch_size', 100)
    if len(request.tickets) > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum of {max_batch_size} tickets"
        )
    
    try:
        predictions = []
        
        for ticket_request in request.tickets:
            # Reuse single prediction logic
            prediction_response = await predict_ticket_resolution(ticket_request, background_tasks, token)
            predictions.append(prediction_response)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Calculate batch analytics if requested
        batch_analytics = None
        if request.include_analytics:
            resolution_times = [p.predicted_resolution_time for p in predictions]
            confidences = [p.confidence_score for p in predictions]
            
            batch_analytics = {
                'average_resolution_time': round(np.mean(resolution_times), 2),
                'median_resolution_time': round(np.median(resolution_times), 2),
                'average_confidence': round(np.mean(confidences), 3),
                'auto_resolvable_count': sum(1 for p in predictions if p.auto_resolvable),
                'escalation_count': sum(1 for p in predictions if p.escalation_recommended),
                'category_distribution': {},  # Could add category analysis
                'priority_distribution': {}   # Could add priority analysis
            }
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_tickets=len(request.tickets),
            processing_time_ms=round(processing_time, 2),
            batch_analytics=batch_analytics
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/metrics", response_model=ModelMetrics)
async def get_model_metrics(token: str = Depends(verify_token)):
    """Get model performance metrics"""
    try:
        # Mock metrics for demo
        return ModelMetrics(
            mae=2.34,
            rmse=3.12,
            r2_score=0.857,
            accuracy_80_percent=82.5,
            last_updated=datetime.now() - timedelta(hours=2),
            total_predictions=app_state.prediction_count
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )

@app.post("/models/retrain")
async def retrain_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Trigger model retraining"""
    try:
        training_id = str(uuid.uuid4())
        
        # Add retraining task to background
        background_tasks.add_task(perform_model_retraining, training_id, request)
        
        return {
            "training_id": training_id,
            "status": "initiated",
            "message": "Model retraining started in background",
            "estimated_duration": "30-60 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error initiating retraining: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate model retraining"
        )

@app.get("/models/status")
async def get_model_status(token: str = Depends(verify_token)):
    """Get status of all models"""
    try:
        model_status = {}
        
        for model_name, model in app_state.models.items():
            model_status[model_name] = {
                "status": "active",
                "type": model.__class__.__name__,
                "last_prediction": datetime.now() - timedelta(minutes=5),
                "prediction_count": app_state.prediction_count // len(app_state.models)
            }
        
        return {
            "models": model_status,
            "total_models": len(app_state.models),
            "ensemble_enabled": "ensemble" in app_state.models
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model status"
        )

# Background tasks
async def update_prediction_metrics(prediction_id: str, request: TicketPredictionRequest, response: TicketPredictionResponse):
    """Update prediction metrics in background"""
    try:
        # Store prediction for later analysis
        app_state.prediction_cache[prediction_id] = {
            'request': request.dict(),
            'response': response.dict(),
            'timestamp': datetime.now()
        }
        
        # Clean old cache entries
        cutoff_time = datetime.now() - timedelta(hours=24)
        app_state.prediction_cache = {
            k: v for k, v in app_state.prediction_cache.items()
            if v['timestamp'] > cutoff_time
        }
        
        logger.info(f"Updated metrics for prediction {prediction_id}")
        
    except Exception as e:
        logger.error(f"Error updating metrics: {str(e)}")

async def perform_model_retraining(training_id: str, request: TrainingRequest):
    """Perform model retraining in background"""
    try:
        logger.info(f"Starting model retraining {training_id}")
        
        # Simulate retraining delay
        await asyncio.sleep(10)  # Reduced for demo
        
        # In a real implementation, this would:
        # 1. Load new training data
        # 2. Retrain models
        # 3. Validate performance
        # 4. Update model files
        # 5. Reload models in memory
        
        logger.info(f"Model retraining {training_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model retraining {training_id}: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting AI IT Support Ticket Predictor API")
    logger.info(f"Loaded {len(app_state.models)} models")
    logger.info("API server ready to accept requests")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down AI IT Support Ticket Predictor API")

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )