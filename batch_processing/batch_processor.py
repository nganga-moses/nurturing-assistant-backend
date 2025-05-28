import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from ..data.data_processor import DataProcessor
from ..data.models import get_session, StudentProfile, EngagementHistory
from .model_pipeline import ModelUpdatePipeline

logger = logging.getLogger(__name__)

class BatchDataProcessor:
    def __init__(self):
        self.session = get_session()
        self.data_processor = DataProcessor()
        self.model_pipeline = ModelUpdatePipeline()
        
    def validate_crm_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the CRM data before processing
        
        Args:
            df: DataFrame containing CRM data
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        required_columns = [
            'student_id',
            'status',
            'funnel_stage',
            'last_interaction_date',
            'engagement_count'
        ]
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check data types
        try:
            df['last_interaction_date'] = pd.to_datetime(df['last_interaction_date'])
        except Exception as e:
            logger.error(f"Error converting last_interaction_date: {e}")
            return False
            
        return True
        
    def process_batch_update(self, csv_path: str, update_type: str):
        """
        Process a batch update from CRM CSV
        
        Args:
            csv_path: Path to the CSV file from CRM
            update_type: Type of update ('daily' or 'weekly')
        """
        logger.info(f"Processing {update_type} update from {csv_path}")
        
        try:
            # Load and validate CSV
            df = pd.read_csv(csv_path)
            if not self.validate_crm_data(df):
                raise ValueError("CRM data validation failed")
            
            # Process based on update type
            if update_type == 'daily':
                self.process_daily_update(df)
            else:
                self.process_weekly_update(df)
                
            # Trigger model update if needed
            self.model_pipeline.schedule_retraining(update_type)
            
            logger.info(f"Successfully processed {update_type} update")
            
        except Exception as e:
            logger.error(f"Error processing batch update: {e}")
            raise
    
    def process_daily_update(self, df: pd.DataFrame):
        """Process daily updates (faster, less intensive)"""
        logger.info("Processing daily update")
        
        try:
            # Update student statuses
            self.update_student_statuses(df)
            
            # Update engagement metrics
            self.update_engagement_metrics(df)
            
            # Update funnel stages
            self.update_funnel_stages(df)
            
            self.session.commit()
            logger.info("Daily update completed successfully")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error in daily update: {e}")
            raise
    
    def process_weekly_update(self, df: pd.DataFrame):
        """Process weekly updates (more comprehensive)"""
        logger.info("Processing weekly update")
        
        try:
            # Do everything in daily update
            self.process_daily_update(df)
            
            # Additional weekly tasks
            self.recalculate_application_likelihood()
            self.update_risk_scores()
            self.generate_weekly_reports()
            
            self.session.commit()
            logger.info("Weekly update completed successfully")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error in weekly update: {e}")
            raise
    
    def update_student_statuses(self, df: pd.DataFrame):
        """Update student statuses from CRM data"""
        for _, row in df.iterrows():
            student = self.session.query(StudentProfile).filter_by(
                student_id=row['student_id']
            ).first()
            
            if student:
                # Update basic information
                student.funnel_stage = row['funnel_stage']
                student.last_interaction_date = row['last_interaction_date']
                student.interaction_count = row['engagement_count']
                
                # Update application status if provided
                if 'application_status' in row:
                    student.application_status = row['application_status']
    
    def update_engagement_metrics(self, df: pd.DataFrame):
        """Update engagement metrics from CRM data"""
        if 'engagement_metrics' not in df.columns:
            return
            
        for _, row in df.iterrows():
            engagement = self.session.query(EngagementHistory).filter_by(
                student_id=row['student_id'],
                timestamp=row['last_interaction_date']
            ).first()
            
            if engagement:
                engagement.engagement_metrics = row['engagement_metrics']
    
    def update_funnel_stages(self, df: pd.DataFrame):
        """Update funnel stages from CRM data"""
        for _, row in df.iterrows():
            student = self.session.query(StudentProfile).filter_by(
                student_id=row['student_id']
            ).first()
            
            if student and 'funnel_stage' in row:
                student.funnel_stage = row['funnel_stage']
    
    def recalculate_application_likelihood(self):
        """Recalculate application likelihood scores for all students"""
        # Get all students
        students = self.session.query(StudentProfile).all()
        
        for student in students:
            # Get student's engagements
            engagements = self.session.query(EngagementHistory).filter_by(
                student_id=student.student_id
            ).all()
            
            # Calculate new likelihood score based on engagements
            new_score = self.calculate_likelihood_score(student, engagements)
            student.application_likelihood_score = new_score
    
    def calculate_likelihood_score(self, student: StudentProfile, 
                                 engagements: List[EngagementHistory]) -> float:
        """
        Calculate application likelihood score for a student based on:
        1. Engagement patterns and frequency
        2. Funnel stage progression
        3. Content interaction quality
        4. Time-based factors
        5. Response patterns
        """
        if not engagements:
            return 0.0
            
        # 1. Engagement Pattern Analysis
        engagement_scores = []
        for engagement in engagements:
            # Base score from engagement type
            type_scores = {
                'view': 0.3,
                'click': 0.4,
                'form_submit': 0.8,
                'document_upload': 0.9,
                'application_submit': 1.0
            }
            base_score = type_scores.get(engagement.engagement_type.lower(), 0.2)
            
            # Adjust based on engagement metrics
            if engagement.engagement_metrics:
                metrics = engagement.engagement_metrics
                # Time spent factor (0-1)
                time_spent = metrics.get('time_spent', 0)
                time_score = min(time_spent / 300, 1.0)  # 5 minutes max
                
                # Scroll depth factor (0-1)
                scroll_depth = metrics.get('scroll_depth', 0)
                scroll_score = min(scroll_depth / 100, 1.0)
                
                # Interaction quality
                interaction_count = metrics.get('interaction_count', 0)
                interaction_score = min(interaction_count / 10, 1.0)
                
                # Combine metrics
                engagement_score = (base_score * 0.4 + 
                                  time_score * 0.2 + 
                                  scroll_score * 0.2 + 
                                  interaction_score * 0.2)
                engagement_scores.append(engagement_score)
        
        # Calculate average engagement score
        avg_engagement_score = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
        
        # 2. Funnel Stage Analysis
        funnel_stages = {
            'awareness': 0.2,
            'interest': 0.4,
            'consideration': 0.6,
            'decision': 0.8,
            'application': 1.0
        }
        stage_score = funnel_stages.get(student.funnel_stage.lower(), 0.0)
        
        # 3. Time-based Analysis
        if student.first_interaction_date and student.last_interaction_date:
            total_days = (student.last_interaction_date - student.first_interaction_date).days
            if total_days > 0:
                # Calculate engagement velocity (engagements per day)
                engagement_velocity = student.interaction_count / total_days
                # Normalize velocity (0.5 engagements per day is considered good)
                velocity_score = min(engagement_velocity / 0.5, 1.0)
            else:
                velocity_score = 0.0
        else:
            velocity_score = 0.0
            
        # 4. Response Pattern Analysis
        response_scores = []
        for engagement in engagements:
            if engagement.engagement_response:
                # Analyze response quality
                response = engagement.engagement_response.lower()
                if 'completed' in response or 'submitted' in response:
                    response_scores.append(1.0)
                elif 'started' in response or 'in_progress' in response:
                    response_scores.append(0.7)
                elif 'viewed' in response or 'clicked' in response:
                    response_scores.append(0.4)
                else:
                    response_scores.append(0.2)
        
        avg_response_score = sum(response_scores) / len(response_scores) if response_scores else 0
        
        # 5. Combine all factors with weights
        final_score = (
            avg_engagement_score * 0.35 +  # Engagement quality
            stage_score * 0.25 +          # Funnel stage
            velocity_score * 0.20 +        # Engagement velocity
            avg_response_score * 0.20      # Response patterns
        )
        
        return round(final_score, 2)
    
    def update_risk_scores(self):
        """Update risk scores for all students"""
        # Get all students
        students = self.session.query(StudentProfile).all()
        
        for student in students:
            # Get student's engagements
            engagements = self.session.query(EngagementHistory).filter_by(
                student_id=student.student_id
            ).all()
            
            # Calculate new risk score
            new_score = self.calculate_risk_score(student, engagements)
            student.dropout_risk_score = new_score
    
    def calculate_risk_score(self, student: StudentProfile, 
                           engagements: List[EngagementHistory]) -> float:
        """
        Calculate dropout risk score based on:
        1. Engagement frequency and recency
        2. Funnel stage stagnation
        3. Response patterns
        4. Content interaction quality
        5. Time-based factors
        """
        if not engagements:
            return 0.8  # High risk if no engagements
            
        risk_factors = []
        
        # 1. Engagement Frequency and Recency
        if student.last_interaction_date:
            days_since_last = (datetime.now() - student.last_interaction_date).days
            # Higher risk if no engagement in last 30 days
            recency_risk = min(days_since_last / 30, 1.0)
            risk_factors.append(recency_risk)
            
            # Calculate engagement frequency
            if student.first_interaction_date:
                total_days = (student.last_interaction_date - student.first_interaction_date).days
                if total_days > 0:
                    engagement_frequency = student.interaction_count / total_days
                    # Risk increases if frequency is below 0.2 engagements per day
                    frequency_risk = max(0, 1 - (engagement_frequency / 0.2))
                    risk_factors.append(frequency_risk)
        
        # 2. Funnel Stage Stagnation
        stage_risks = {
            'awareness': 0.8,  # High risk of dropping out in awareness stage
            'interest': 0.6,
            'consideration': 0.4,
            'decision': 0.2,
            'application': 0.1
        }
        stage_risk = stage_risks.get(student.funnel_stage.lower(), 0.5)
        risk_factors.append(stage_risk)
        
        # 3. Response Pattern Analysis
        response_risks = []
        for engagement in engagements:
            if engagement.engagement_response:
                response = engagement.engagement_response.lower()
                if 'abandoned' in response or 'failed' in response:
                    response_risks.append(1.0)
                elif 'incomplete' in response:
                    response_risks.append(0.8)
                elif 'started' in response:
                    response_risks.append(0.6)
                elif 'viewed' in response:
                    response_risks.append(0.4)
                else:
                    response_risks.append(0.2)
        
        if response_risks:
            avg_response_risk = sum(response_risks) / len(response_risks)
            risk_factors.append(avg_response_risk)
        
        # 4. Content Interaction Quality
        interaction_risks = []
        for engagement in engagements:
            if engagement.engagement_metrics:
                metrics = engagement.engagement_metrics
                
                # Time spent risk (too short or too long)
                time_spent = metrics.get('time_spent', 0)
                if time_spent < 30:  # Less than 30 seconds
                    time_risk = 0.8
                elif time_spent > 600:  # More than 10 minutes
                    time_risk = 0.6
                else:
                    time_risk = 0.3
                interaction_risks.append(time_risk)
                
                # Scroll depth risk
                scroll_depth = metrics.get('scroll_depth', 0)
                if scroll_depth < 20:  # Less than 20% scrolled
                    scroll_risk = 0.9
                elif scroll_depth < 50:  # Less than 50% scrolled
                    scroll_risk = 0.7
                else:
                    scroll_risk = 0.3
                interaction_risks.append(scroll_risk)
        
        if interaction_risks:
            avg_interaction_risk = sum(interaction_risks) / len(interaction_risks)
            risk_factors.append(avg_interaction_risk)
        
        # 5. Time-based Factors
        if student.first_interaction_date:
            days_since_first = (datetime.now() - student.first_interaction_date).days
            # Risk increases if student has been in the funnel for too long
            time_risk = min(days_since_first / 180, 1.0)  # 6 months max
            risk_factors.append(time_risk)
        
        # Calculate final risk score
        if risk_factors:
            final_risk = sum(risk_factors) / len(risk_factors)
            return round(final_risk, 2)
        else:
            return 0.5  # Default medium risk if no factors available
    
    def generate_weekly_reports(self):
        """Generate weekly reports"""
        # Implement your report generation logic here
        pass 