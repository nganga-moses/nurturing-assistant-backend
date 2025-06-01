#!/usr/bin/env python
"""
Script for scheduling automated data updates and model training.

This script:
1. Schedules daily and weekly updates
2. Handles status tracking and reporting
3. Manages model retraining triggers
4. Provides manual update capabilities

Usage:
    # Start the scheduler
    python schedule_updates.py start

    # Run a manual update
    python schedule_updates.py manual --csv-path data/crm_update.csv --type daily

    # Stop the scheduler
    python schedule_updates.py stop
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import uuid
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.status_tracker import StatusTracker
from scripts.ingest_engagements import main as ingest_engagements
from scripts.ingest_students import main as ingest_students

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpdateScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.status_tracker = StatusTracker()
        
    def start(self):
        """Start the scheduler"""
        logger.info("Starting update scheduler")
        
        # Schedule daily update (runs at 1 AM every day)
        self.scheduler.add_job(
            self.run_daily_update,
            CronTrigger(hour=1, minute=0),
            id='daily_update',
            name='Daily Data Update'
        )
        
        # Schedule weekly update (runs at 2 AM every Sunday)
        self.scheduler.add_job(
            self.run_weekly_update,
            CronTrigger(day_of_week='sun', hour=2, minute=0),
            id='weekly_update',
            name='Weekly Data Update'
        )
        
        self.scheduler.start()
        logger.info("Update scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping update scheduler")
        self.scheduler.shutdown()
        logger.info("Update scheduler stopped")
    
    def run_daily_update(self):
        """Run daily update"""
        batch_id = f"daily_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Starting daily update {batch_id}")
        
        try:
            # Run engagement updates
            ingest_engagements(['--days', '1'])
            
            # Run student updates
            ingest_students(['--days', '1'])
            
            # Track status changes
            self.status_tracker.track_status_changes(batch_id)
            
            # Generate report
            report = self.status_tracker.generate_status_report(batch_id)
            logger.info(f"Daily update completed: {report}")
            
        except Exception as e:
            logger.error(f"Error in daily update: {e}")
            raise
    
    def run_weekly_update(self):
        """Run weekly update"""
        batch_id = f"weekly_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Starting weekly update {batch_id}")
        
        try:
            # Run engagement updates
            ingest_engagements(['--days', '7'])
            
            # Run student updates
            ingest_students(['--days', '7', '--force-retrain'])
            
            # Track status changes
            self.status_tracker.track_status_changes(batch_id)
            
            # Generate report
            report = self.status_tracker.generate_status_report(batch_id)
            logger.info(f"Weekly update completed: {report}")
            
        except Exception as e:
            logger.error(f"Error in weekly update: {e}")
            raise
    
    def run_manual_update(self, csv_path: str, update_type: str):
        """
        Run a manual update
        
        Args:
            csv_path: Path to the update CSV file
            update_type: Type of update ('daily' or 'weekly')
        """
        batch_id = f"manual_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting manual update {batch_id}")
        
        try:
            if update_type == 'daily':
                # Run daily update
                ingest_engagements(['--engagements-csv', csv_path])
                ingest_students(['--students-csv', csv_path])
            else:
                # Run weekly update
                ingest_engagements(['--engagements-csv', csv_path, '--force-retrain'])
                ingest_students(['--students-csv', csv_path, '--force-retrain'])
            
            # Track status changes
            self.status_tracker.track_status_changes(batch_id)
            
            # Generate report
            report = self.status_tracker.generate_status_report(batch_id)
            logger.info(f"Manual update completed: {report}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in manual update: {e}")
            raise

def main():
    """Main function to run the scheduler"""
    parser = argparse.ArgumentParser(description="Schedule automated data updates")
    parser.add_argument('action', choices=['start', 'stop', 'manual'],
                       help="Action to perform")
    parser.add_argument('--csv-path', type=str,
                       help="Path to CSV file for manual update")
    parser.add_argument('--type', choices=['daily', 'weekly'],
                       help="Type of manual update")
    args = parser.parse_args()
    
    scheduler = UpdateScheduler()
    
    if args.action == 'start':
        scheduler.start()
    elif args.action == 'stop':
        scheduler.stop()
    elif args.action == 'manual':
        if not args.csv_path or not args.type:
            parser.error("Manual update requires --csv-path and --type")
        scheduler.run_manual_update(args.csv_path, args.type)

if __name__ == "__main__":
    main() 