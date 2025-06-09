from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from uuid import UUID
import logging
import json

from database.session import get_db
from api.services.success_factor_analyzer import SuccessFactorAnalyzer
from data.models.funnel_stage import FunnelStage

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/success-factors",
    tags=["success-factors"]
)

@router.get("/summary")
async def get_success_factors_summary(
    db: Session = Depends(get_db)
):
    """
    Get a summary of success factors across all stages.
    """
    try:
        # Get all active stages
        stages = (
            db.query(FunnelStage)
            .filter(FunnelStage.is_active == True)
            .order_by(FunnelStage.stage_order)
            .all()
        )
        
        logger.info(f"Found {len(stages)} active stages")
        
        if not stages:
            raise HTTPException(
                status_code=404,
                detail="No active stages found"
            )
        
        # Create analyzer
        analyzer = SuccessFactorAnalyzer(db)
        
        # Analyze each stage
        stage_analyses = {}
        for stage in stages:
            try:
                logger.info(f"Analyzing stage: {stage.stage_name}")
                analysis = analyzer.analyze_success_factors(stage)
                stage_analyses[stage.stage_name] = analysis
                logger.info(f"Successfully analyzed stage {stage.stage_name}")
            except Exception as e:
                logger.error(f"Error analyzing stage {stage.stage_name}: {str(e)}", exc_info=True)
                continue
        
        if not stage_analyses:
            logger.error("No stages were successfully analyzed")
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze any stages"
            )
        
        # Calculate overall summary
        logger.info("Calculating overall factors")
        overall_factors = _calculate_overall_factors(stage_analyses)
        
        summary = {
            "stage_analyses": stage_analyses,
            "overall_factors": overall_factors
        }
        
        # Log the response data
        logger.info(f"Success factors summary response: {json.dumps(summary, default=str)}")
        
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_success_factors_summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/{stage_id}", response_model=Dict)
async def analyze_success_factors(
    stage_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Analyze factors contributing to success for a specific stage.
    """
    try:
        # Get the goal stage
        goal_stage = (
            db.query(FunnelStage)
            .filter(
                FunnelStage.id == stage_id,
                FunnelStage.is_active == True
            )
            .first()
        )
        
        if not goal_stage:
            raise HTTPException(
                status_code=404,
                detail="Goal stage not found"
            )
        
        # Create analyzer and get analysis
        analyzer = SuccessFactorAnalyzer(db)
        analysis = analyzer.analyze_success_factors(goal_stage)
        
        # Log the response data
        logger.info(f"Stage analysis response: {json.dumps(analysis, default=str)}")
        
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_success_factors: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

def _calculate_overall_factors(stage_analyses: Dict) -> Dict:
    """Calculate overall success factors across all stages."""
    try:
        # Combine all factors
        all_factors = []
        for stage_name, analysis in stage_analyses.items():
            for factor_type in ["engagement_factors", "timing_factors", "stage_factors", "response_factors"]:
                for factor in analysis.get(factor_type, []):
                    all_factors.append({
                        "stage": stage_name,
                        "type": factor_type.replace("_factors", ""),
                        "factor": factor,
                        "impact": factor.get("impact", {}).get("value", 0)
                    })
        
        # Sort by impact
        all_factors.sort(key=lambda x: x["impact"], reverse=True)
        
        # Get top factors by type
        top_factors = {}
        for factor_type in ["engagement", "timing", "stage", "response"]:
            type_factors = [f for f in all_factors if f["type"] == factor_type]
            top_factors[factor_type] = type_factors[:3]  # Top 3 factors per type
        
        result = {
            "top_factors": top_factors,
            "all_factors": all_factors[:10]  # Top 10 factors overall
        }
        
        # Log the calculated factors
        logger.info(f"Calculated overall factors: {json.dumps(result, default=str)}")
        
        return result
    except Exception as e:
        logger.error(f"Error in _calculate_overall_factors: {str(e)}")
        return {
            "top_factors": {},
            "all_factors": []
        } 