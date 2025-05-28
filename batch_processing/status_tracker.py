import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from ..data.models import get_session, StudentProfile, StatusChange

logger = logging.getLogger(__name__)

class BatchStatusTracker:
    def __init__(self):
        self.session = get_session()
        
    def track_status_changes(self, df: pd.DataFrame, batch_id: str):
        """
        Track status changes from batch update
        
        Args:
            df: DataFrame containing CRM data
            batch_id: Unique identifier for this batch update
        """
        logger.info(f"Tracking status changes for batch {batch_id}")
        
        try:
            for _, row in df.iterrows():
                student = self.session.query(StudentProfile).filter_by(
                    student_id=row['student_id']
                ).first()
                
                if student:
                    # Track funnel stage changes
                    if 'funnel_stage' in row and row['funnel_stage'] != student.funnel_stage:
                        self._record_status_change(
                            student.student_id,
                            'funnel_stage',
                            student.funnel_stage,
                            row['funnel_stage'],
                            batch_id
                        )
                    
                    # Track application status changes
                    if 'application_status' in row and row['application_status'] != student.application_status:
                        self._record_status_change(
                            student.student_id,
                            'application_status',
                            student.application_status,
                            row['application_status'],
                            batch_id
                        )
            
            self.session.commit()
            logger.info(f"Status changes tracked for batch {batch_id}")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error tracking status changes: {e}")
            raise
    
    def _record_status_change(self, student_id: str, field: str, 
                            old_value: str, new_value: str, batch_id: str):
        """
        Record a status change in the database
        
        Args:
            student_id: ID of the student
            field: Field that changed
            old_value: Previous value
            new_value: New value
            batch_id: ID of the batch update
        """
        change = StatusChange(
            student_id=student_id,
            field=field,
            old_value=old_value,
            new_value=new_value,
            batch_id=batch_id,
            timestamp=datetime.now()
        )
        self.session.add(change)
    
    def get_status_changes(self, batch_id: Optional[str] = None, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get status changes for reporting
        
        Args:
            batch_id: Optional batch ID to filter by
            start_date: Optional start date to filter by
            end_date: Optional end date to filter by
            
        Returns:
            DataFrame containing status changes
        """
        query = self.session.query(StatusChange)
        
        if batch_id:
            query = query.filter_by(batch_id=batch_id)
            
        if start_date:
            query = query.filter(StatusChange.timestamp >= start_date)
            
        if end_date:
            query = query.filter(StatusChange.timestamp <= end_date)
            
        changes = query.all()
        
        # Convert to DataFrame
        data = []
        for change in changes:
            data.append({
                'student_id': change.student_id,
                'field': change.field,
                'old_value': change.old_value,
                'new_value': change.new_value,
                'batch_id': change.batch_id,
                'timestamp': change.timestamp
            })
            
        return pd.DataFrame(data)
    
    def generate_status_report(self, batch_id: str) -> Dict:
        """
        Generate a report of status changes for a batch
        
        Args:
            batch_id: ID of the batch to report on
            
        Returns:
            Dictionary containing report data
        """
        changes_df = self.get_status_changes(batch_id=batch_id)
        
        if changes_df.empty:
            return {
                'batch_id': batch_id,
                'total_changes': 0,
                'changes_by_field': {},
                'changes_by_student': {}
            }
        
        # Calculate statistics
        total_changes = len(changes_df)
        changes_by_field = changes_df['field'].value_counts().to_dict()
        changes_by_student = changes_df['student_id'].value_counts().to_dict()
        
        # Get most recent changes
        recent_changes = changes_df.sort_values('timestamp', ascending=False).head(10)
        recent_changes_list = recent_changes.to_dict('records')
        
        return {
            'batch_id': batch_id,
            'total_changes': total_changes,
            'changes_by_field': changes_by_field,
            'changes_by_student': changes_by_student,
            'recent_changes': recent_changes_list
        } 