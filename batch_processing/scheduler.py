import logging
from datetime import datetime
import uuid
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from .batch_processor import BatchDataProcessor
from .status_tracker import BatchStatusTracker

logger = logging.getLogger(__name__)

class BatchProcessingScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.batch_processor = BatchDataProcessor()
        self.status_tracker = BatchStatusTracker()
        
    def start(self):
        """Start the scheduler"""
        logger.info("Starting batch processing scheduler")
        
        # Schedule daily update (runs at 1 AM every day)
        self.scheduler.add_job(
            self.run_daily_update,
            CronTrigger(hour=1, minute=0),
            id='daily_update',
            name='Daily CRM Update'
        )
        
        # Schedule weekly update (runs at 2 AM every Sunday)
        self.scheduler.add_job(
            self.run_weekly_update,
            CronTrigger(day_of_week='sun', hour=2, minute=0),
            id='weekly_update',
            name='Weekly CRM Update'
        )
        
        self.scheduler.start()
        logger.info("Batch processing scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping batch processing scheduler")
        self.scheduler.shutdown()
        logger.info("Batch processing scheduler stopped")
    
    def run_daily_update(self):
        """Run daily batch update"""
        batch_id = f"daily_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Starting daily update {batch_id}")
        
        try:
            # Process the daily update
            self.batch_processor.process_batch_update(
                csv_path='data/crm_daily_update.csv',
                update_type='daily'
            )
            
            # Track status changes
            df = self.batch_processor.load_latest_crm_data()
            self.status_tracker.track_status_changes(df, batch_id)
            
            # Generate report
            report = self.status_tracker.generate_status_report(batch_id)
            logger.info(f"Daily update completed: {report}")
            
        except Exception as e:
            logger.error(f"Error in daily update: {e}")
            raise
    
    def run_weekly_update(self):
        """Run weekly batch update"""
        batch_id = f"weekly_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Starting weekly update {batch_id}")
        
        try:
            # Process the weekly update
            self.batch_processor.process_batch_update(
                csv_path='data/crm_weekly_update.csv',
                update_type='weekly'
            )
            
            # Track status changes
            df = self.batch_processor.load_latest_crm_data()
            self.status_tracker.track_status_changes(df, batch_id)
            
            # Generate report
            report = self.status_tracker.generate_status_report(batch_id)
            logger.info(f"Weekly update completed: {report}")
            
        except Exception as e:
            logger.error(f"Error in weekly update: {e}")
            raise
    
    def run_manual_update(self, csv_path: str, update_type: str):
        """
        Run a manual batch update
        
        Args:
            csv_path: Path to the CRM CSV file
            update_type: Type of update ('daily' or 'weekly')
        """
        batch_id = f"manual_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting manual update {batch_id}")
        
        try:
            # Process the update
            self.batch_processor.process_batch_update(
                csv_path=csv_path,
                update_type=update_type
            )
            
            # Track status changes
            df = self.batch_processor.load_latest_crm_data()
            self.status_tracker.track_status_changes(df, batch_id)
            
            # Generate report
            report = self.status_tracker.generate_status_report(batch_id)
            logger.info(f"Manual update completed: {report}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in manual update: {e}")
            raise 