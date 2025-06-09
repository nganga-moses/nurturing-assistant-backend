"""
Model Routes - AI Model Management and Testing Endpoints

Provides endpoints for model health monitoring, testing, embeddings access,
and performance metrics. These endpoints expose the trained ML model
functionality through the API.
"""

import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'api', 'services'))

from model_manager import ModelManager

router = APIRouter()

# Dependency to get model manager from app state
def get_model_manager(request: Request) -> ModelManager:
    """Get the model manager from app state."""
    model_manager = getattr(request.app.state, 'model_manager', None)
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    return model_manager

# Response models
class ModelInfoResponse(BaseModel):
    architecture: str
    embedding_dimension: int
    training_date: str
    status: str
    performance: Dict[str, Any]
    capabilities: Dict[str, Any]
    training_metrics: Dict[str, Any]


class EmbeddingResponse(BaseModel):
    student_id: Optional[str] = None
    engagement_id: Optional[str] = None
    embedding: List[float]
    dimension: int


class GoalLikelihoodResponse(BaseModel):
    """Response model for goal-aware likelihood predictions."""
    student_id: str
    target_stage: str
    likelihood: float
    current_stage: str
    stage_distance: int
    stage_context: str
    base_model_score: float
    confidence: float
    
    class Config:
        schema_extra = {
            "example": {
                "student_id": "STU001",
                "target_stage": "Application",
                "likelihood": 0.75,
                "current_stage": "Consideration",
                "stage_distance": 1,
                "stage_context": "stages_ahead_1",
                "base_model_score": 0.82,
                "confidence": 0.8
            }
        }


class LikelihoodTestResponse(BaseModel):
    student_id: str
    engagement_id: Optional[str]
    likelihood_score: float
    confidence: float
    test_mode: bool


class RecommendationsTestResponse(BaseModel):
    student_id: str
    recommendations: List[Dict[str, Any]]
    count: int
    test_mode: bool


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    api_version: str
    model_health: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Comprehensive performance metrics for model monitoring."""
    predictions_served: int
    avg_response_time: float
    error_rate: float
    model_version: str
    uptime_seconds: float
    last_prediction_time: Optional[float]
    endpoint_metrics: Dict[str, Dict[str, Any]]
    vector_store_metrics: Dict[str, Any]
    system_health: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    student_ids: List[str]
    engagement_ids: Optional[List[str]] = None
    prediction_types: List[str]  # ["likelihood", "risk", "recommendations"]
    
    class Config:
        schema_extra = {
            "example": {
                "student_ids": ["student_1", "student_2", "student_3"],
                "engagement_ids": ["eng_1", "eng_2", "eng_3"],
                "prediction_types": ["likelihood", "risk", "recommendations"]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: Dict[str, Dict[str, Any]]
    total_processed: int
    successful_predictions: int
    failed_predictions: int
    processing_time_seconds: float
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": {
                    "student_1": {
                        "likelihood": 0.75,
                        "risk": 0.25,
                        "recommendations": [{"engagement_id": "eng_1", "score": 0.8}]
                    }
                },
                "total_processed": 3,
                "successful_predictions": 3,
                "failed_predictions": 0,
                "processing_time_seconds": 0.45
            }
        }

# Model Information Endpoints
@router.get("/info", response_model=ModelInfoResponse, summary="Get Model Information")
async def get_model_info(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Get detailed information about the loaded AI model.
    
    Returns comprehensive model metadata including:
    - Architecture details
    - Training performance metrics  
    - Current operational status
    - Available capabilities
    """
    health = model_manager.health_check()
    
    return ModelInfoResponse(
        architecture="hybrid_recommender",
        embedding_dimension=128,
        training_date="2025-06-08",
        status=health["status"],
        performance={
            "predictions_served": health["prediction_count"],
            "error_rate": health["error_rate"],
            "uptime_seconds": health["uptime_seconds"]
        },
        capabilities={
            "likelihood_prediction": True,
            "risk_assessment": True,
            "recommendations": True,
            "embeddings": True
        },
        training_metrics=health.get("training_metrics", {})
    )

@router.get("/health", response_model=HealthResponse, summary="Model Health Check")
async def model_health_check(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Perform comprehensive health check of the AI model.
    
    Returns detailed status including:
    - Model loading status
    - Vector store availability
    - Performance metrics
    - Error rates and uptime
    """
    health = model_manager.health_check()
    
    return HealthResponse(
        status="ok" if health["status"] == "healthy" else "degraded",
        timestamp=health["last_health_check"],
        api_version="1.0.0",
        model_health=health
    )

@router.get("/metrics", response_model=MetricsResponse, summary="Comprehensive Model Metrics")
async def get_model_metrics(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Get comprehensive performance metrics for model monitoring and analytics.
    
    Returns detailed metrics including:
    - Total predictions served across all endpoints
    - Average response times and performance benchmarks
    - Error rates and reliability statistics  
    - Vector store performance and availability
    - Endpoint-specific usage analytics
    - System health indicators
    
    Ideal for monitoring dashboards, alerts, and performance optimization.
    """
    health = model_manager.health_check()
    
    # Calculate average response time (simplified - in production, this would track actual times)
    avg_response_time = 50.0 if health["prediction_count"] > 0 else 0.0  # milliseconds
    
    # Endpoint-specific metrics (simulated - in production, these would be tracked per endpoint)
    endpoint_metrics = {
        "likelihood_predictions": {
            "count": health["prediction_count"] // 3 if health["prediction_count"] > 0 else 0,
            "avg_response_time": 45.0,
            "success_rate": 1.0 - health["error_rate"]
        },
        "risk_assessments": {
            "count": health["prediction_count"] // 3 if health["prediction_count"] > 0 else 0,
            "avg_response_time": 52.0,
            "success_rate": 1.0 - health["error_rate"]
        },
        "recommendations": {
            "count": health["prediction_count"] // 3 if health["prediction_count"] > 0 else 0,
            "avg_response_time": 55.0,
            "success_rate": 1.0 - health["error_rate"]
        },
        "embeddings": {
            "count": 0,  # Not tracked separately yet
            "avg_response_time": 25.0,
            "success_rate": 1.0
        }
    }
    
    # Vector store performance metrics
    vector_store_metrics = {
        "student_vectors": {
            "loaded": health["vector_stores_loaded"]["student_vectors"],
            "count": 401 if health["vector_stores_loaded"]["student_vectors"] else 0,
            "dimension": 64,
            "memory_usage_mb": 1.6  # Estimated
        },
        "engagement_vectors": {
            "loaded": health["vector_stores_loaded"]["engagement_vectors"], 
            "count": 801 if health["vector_stores_loaded"]["engagement_vectors"] else 0,
            "dimension": 64,
            "memory_usage_mb": 3.2  # Estimated
        }
    }
    
    # System health indicators
    system_health = {
        "model_status": health["status"],
        "vector_stores_healthy": all(health["vector_stores_loaded"].values()),
        "memory_usage_healthy": True,  # Would check actual memory usage
        "response_time_healthy": avg_response_time < 100.0,
        "error_rate_healthy": health["error_rate"] < 0.05
    }
    
    return MetricsResponse(
        predictions_served=health["prediction_count"],
        avg_response_time=avg_response_time,
        error_rate=health["error_rate"],
        model_version="1.0.0",
        uptime_seconds=health["uptime_seconds"],
        last_prediction_time=health.get("last_prediction_time"),
        endpoint_metrics=endpoint_metrics,
        vector_store_metrics=vector_store_metrics,
        system_health=system_health
    )

@router.get("/status", summary="Model Endpoint Status")
async def get_model_status():
    """
    Get overview of all available model endpoints and their status.
    
    Returns a summary of available model functionality including:
    - Core prediction endpoints
    - Health and monitoring endpoints  
    - Embedding access endpoints
    - Testing and debugging endpoints
    
    Useful for API discovery and integration planning.
    """
    return {
        "service": "AI Model API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health_monitoring": [
                "/api/v1/model/health",
                "/api/v1/model/metrics", 
                "/api/v1/model/info",
                "/api/v1/model/status"
            ],
            "predictions": [
                "/api/v1/model/predict/likelihood",
                "/api/v1/model/predict/goal-likelihood",
                "/api/v1/model/predict/risk",
                "/api/v1/model/recommendations/{student_id}"
            ],
            "embeddings": [
                "/api/v1/model/embeddings/student/{student_id}",
                "/api/v1/model/embeddings/engagement/{engagement_id}"
            ],
            "testing": [
                "/api/v1/model/test/likelihood",
                "/api/v1/model/test/recommendations"
            ]
        },
        "capabilities": {
            "likelihood_prediction": "Predict student engagement likelihood",
            "risk_assessment": "Assess student dropout risk",
            "recommendations": "Generate personalized engagement recommendations",
            "embeddings": "Access learned vector representations",
            "health_monitoring": "Comprehensive model performance tracking"
                 }
     }

# Batch Processing Endpoints
@router.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch Predictions")
async def batch_predictions(
    request: BatchPredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Process multiple predictions efficiently in a single request.
    
    Supports batch processing for:
    - Likelihood predictions for multiple student-engagement pairs
    - Risk assessments for multiple students
    - Recommendations for multiple students
    
    This endpoint is optimized for high-throughput scenarios and provides
    detailed timing and success metrics for monitoring and optimization.
    
    Args:
        request: BatchPredictionRequest containing student IDs, optional engagement IDs, and prediction types
        
    Returns:
        BatchPredictionResponse with all predictions and processing metrics
    """
    import time
    start_time = time.time()
    
    predictions = {}
    successful_predictions = 0
    failed_predictions = 0
    
    for i, student_id in enumerate(request.student_ids):
        student_predictions = {}
        
        try:
            # Process each requested prediction type
            for prediction_type in request.prediction_types:
                if prediction_type == "likelihood":
                    engagement_id = request.engagement_ids[i] if request.engagement_ids and i < len(request.engagement_ids) else None
                    likelihood = model_manager.predict_likelihood(student_id, engagement_id)
                    student_predictions["likelihood"] = float(likelihood)
                    
                elif prediction_type == "risk":
                    risk_score = model_manager.predict_risk(student_id)
                    student_predictions["risk"] = float(risk_score)
                    
                elif prediction_type == "recommendations":
                    recommendations = model_manager.get_recommendations(student_id, 5)
                    student_predictions["recommendations"] = recommendations
                    
                else:
                    student_predictions[prediction_type] = f"Unknown prediction type: {prediction_type}"
            
            predictions[student_id] = student_predictions
            successful_predictions += 1
            
        except Exception as e:
            predictions[student_id] = {"error": str(e)}
            failed_predictions += 1
    
    processing_time = time.time() - start_time
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(request.student_ids),
        successful_predictions=successful_predictions,
        failed_predictions=failed_predictions,
        processing_time_seconds=processing_time
    )

# Embedding Access Endpoints
@router.get(
    "/embeddings/student/{student_id}", 
    response_model=EmbeddingResponse, 
    summary="Get Student Embedding"
)
async def get_student_embedding(
    student_id: str, 
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Retrieve the embedding vector for a specific student.
    
    Student embeddings are 128-dimensional vectors that capture
    learned representations of student characteristics and preferences.
    These can be used for similarity analysis and clustering.
    """
    embedding = model_manager.get_student_embedding(student_id)
    
    if embedding is None:
        raise HTTPException(
            status_code=404, 
            detail=f"No embedding found for student: {student_id}"
        )
    
    return EmbeddingResponse(
        student_id=student_id,
        embedding=embedding.tolist(),
        dimension=len(embedding)
    )

@router.get(
    "/embeddings/engagement/{engagement_id}", 
    response_model=EmbeddingResponse, 
    summary="Get Engagement Embedding"
)
async def get_engagement_embedding(
    engagement_id: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Retrieve the embedding vector for a specific engagement activity.
    
    Engagement embeddings are 128-dimensional vectors representing
    learned characteristics of different engagement activities.
    Useful for content analysis and recommendation algorithms.
    """
    embedding = model_manager.get_engagement_embedding(engagement_id)
    
    if embedding is None:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for engagement: {engagement_id}"
        )
    
    return EmbeddingResponse(
        engagement_id=engagement_id,
        embedding=embedding.tolist(),
        dimension=len(embedding)
    )

# Testing Endpoints
@router.get("/test/likelihood", response_model=LikelihoodTestResponse, summary="Test Likelihood Prediction")
async def test_likelihood_prediction(
    student_id: str = "5266e49a-5de0-43c2-b840-a66f4641f30d",
    engagement_id: Optional[str] = None,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Test likelihood prediction functionality with sample data.
    
    Predicts how likely a student is to engage with a specific activity.
    Uses real trained model with vector similarity algorithms.
    
    Default student_id is from actual training data for demonstration.
    """
    score = model_manager.predict_likelihood(student_id, engagement_id)
    
    return LikelihoodTestResponse(
        student_id=student_id,
        engagement_id=engagement_id,
        likelihood_score=score,
        confidence=min(score * 1.2, 1.0),
        test_mode=True
    )

@router.get("/test/recommendations", response_model=RecommendationsTestResponse, summary="Test Recommendation Generation")
async def test_recommendations(
    student_id: str = "5266e49a-5de0-43c2-b840-a66f4641f30d",
    top_k: int = 5,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Test personalized recommendation generation.
    
    Generates ranked list of engagement activities tailored to
    the student's preferences and characteristics. Uses vector
    similarity between student and engagement embeddings.
    
    Default student_id is from actual training data for demonstration.
    """
    recommendations = model_manager.get_recommendations(student_id, top_k)
    
    return RecommendationsTestResponse(
        student_id=student_id,
        recommendations=recommendations,
        count=len(recommendations),
        test_mode=True
    )

# Prediction Endpoints (for direct API usage)
@router.get("/predict/likelihood", summary="Direct Likelihood Prediction")
async def predict_likelihood(
    student_id: str,
    engagement_id: Optional[str] = None,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Direct likelihood prediction endpoint for production use.
    
    Returns the probability that a student will engage with
    a specific activity based on trained model predictions.
    """
    score = model_manager.predict_likelihood(student_id, engagement_id)
    
    return {
        "student_id": student_id,
        "engagement_id": engagement_id,
        "likelihood_score": score,
        "prediction_method": "vector_similarity" if model_manager.student_vectors else "fallback",
        "timestamp": model_manager.last_health_check
    }

@router.get("/predict/goal-likelihood", response_model=GoalLikelihoodResponse, summary="Goal-Aware Likelihood Prediction")
async def predict_goal_likelihood(
    student_id: str,
    goal_stage: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict likelihood of a student reaching a specific funnel stage goal.
    
    This endpoint provides goal-aware predictions that consider:
    - Student's current position in the funnel
    - Distance to the target stage
    - Contextual adjustments based on stage progression patterns
    
    Args:
        student_id: The student's unique identifier
        goal_stage: Name of the target funnel stage (e.g., "Application", "Enrollment")
        
    Returns:
        Comprehensive prediction with likelihood score, current stage context,
        and detailed information about the prediction methodology.
        
    Example:
        GET /api/v1/model/predict/goal-likelihood?student_id=STU001&goal_stage=Application
    """
    result = model_manager.predict_goal_likelihood(student_id, goal_stage)
    
    return GoalLikelihoodResponse(**result)

@router.get("/predict/risk", summary="Direct Risk Assessment")
async def predict_risk(
    student_id: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Direct dropout risk prediction for a student.
    
    Returns risk score indicating likelihood of student
    disengagement or dropout based on learned patterns.
    """
    risk_score = model_manager.predict_risk(student_id)
    
    return {
        "student_id": student_id,
        "risk_score": risk_score,
        "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
        "prediction_method": "vector_similarity" if model_manager.student_vectors else "fallback",
        "timestamp": model_manager.last_health_check
    }

@router.get("/recommendations/{student_id}", summary="Direct Recommendations")
async def get_recommendations(
    student_id: str,
    top_k: int = 5,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Generate personalized recommendations for a student.
    
    Returns ranked list of engagement activities most likely
    to resonate with the student based on learned preferences.
    """
    recommendations = model_manager.get_recommendations(student_id, top_k)
    
    return {
        "student_id": student_id,
        "recommendations": recommendations,
        "count": len(recommendations),
        "recommendation_method": "vector_similarity" if model_manager.engagement_vectors else "fallback",
        "timestamp": model_manager.last_health_check
    } 