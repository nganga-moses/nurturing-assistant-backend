#!/usr/bin/env python
"""
Status tracking utility for monitoring data updates and model training.

This module provides functionality to:
1. Track student status changes
2. Monitor engagement updates
3. Generate status reports
4. Track model training status
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import os

logger = logging.getLogger(__name__)

class StatusTracker:
    def __init__(self, status_dir: str = "data/status"):
        """
        Initialize the status tracker.
        
        Args:
            status_dir: Directory to store status files
        """
        self.status_dir = status_dir
        os.makedirs(status_dir, exist_ok=True)
        
    def track_student_status(self, student_id: str, status: Dict) -> None:
        """
        Track changes in student status.
        
        Args:
            student_id: Student identifier
            status: Dictionary containing status information
        """
        status_file = os.path.join(self.status_dir, f"student_{student_id}.json")
        
        # Load existing status if available
        current_status = {}
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                current_status = json.load(f)
        
        # Update status
        current_status.update({
            'last_updated': datetime.now().isoformat(),
            'status': status
        })
        
        # Save updated status
        with open(status_file, 'w') as f:
            json.dump(current_status, f, indent=2)
            
        logger.info(f"Updated status for student {student_id}")
    
    def track_batch_status(self, batch_id: str, status: Dict) -> None:
        """
        Track batch processing status.
        
        Args:
            batch_id: Batch identifier
            status: Dictionary containing batch status
        """
        status_file = os.path.join(self.status_dir, f"batch_{batch_id}.json")
        
        # Update status
        status.update({
            'last_updated': datetime.now().isoformat()
        })
        
        # Save status
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
            
        logger.info(f"Updated status for batch {batch_id}")
    
    def generate_status_report(self, batch_id: str) -> Dict:
        """
        Generate a status report for a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Dictionary containing status report
        """
        status_file = os.path.join(self.status_dir, f"batch_{batch_id}.json")
        
        if not os.path.exists(status_file):
            return {
                'status': 'not_found',
                'message': f'No status found for batch {batch_id}'
            }
        
        with open(status_file, 'r') as f:
            status = json.load(f)
            
        return {
            'status': 'success',
            'batch_id': batch_id,
            'last_updated': status['last_updated'],
            'details': status
        }
    
    def get_student_status(self, student_id: str) -> Optional[Dict]:
        """
        Get current status for a student.
        
        Args:
            student_id: Student identifier
            
        Returns:
            Dictionary containing student status or None if not found
        """
        status_file = os.path.join(self.status_dir, f"student_{student_id}.json")
        
        if not os.path.exists(status_file):
            return None
            
        with open(status_file, 'r') as f:
            return json.load(f)
    
    def cleanup_old_status(self, days: int = 30) -> None:
        """
        Clean up status files older than specified days.
        
        Args:
            days: Number of days to keep status files
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for filename in os.listdir(self.status_dir):
            filepath = os.path.join(self.status_dir, filename)
            if os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
                logger.info(f"Removed old status file: {filename}") 