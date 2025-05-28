import logging
from datetime import datetime, timedelta
from typing import Optional
from ..models.collaborative_filtering import CollaborativeFilteringModel
from ..data.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class ModelUpdatePipeline:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = CollaborativeFilteringModel()
        self.last_training_time: Optional[datetime] = None
        
    def schedule_retraining(self, update_type: str):
        """
        Schedule model retraining based on update type and last training time
        
        Args:
            update_type: Type of update ('daily' or 'weekly')
        """
        logger.info(f"Scheduling retraining for {update_type} update")
        
        # Determine if retraining is needed
        if self._should_retrain(update_type):
            logger.info("Retraining conditions met, starting retraining")
            self.retrain_model()
        else:
            logger.info("Retraining conditions not met, skipping retraining")
    
    def _should_retrain(self, update_type: str) -> bool:
        """
        Determine if model should be retrained based on update type and last training time
        
        Args:
            update_type: Type of update ('daily' or 'weekly')
            
        Returns:
            bool: True if model should be retrained, False otherwise
        """
        if not self.last_training_time:
            return True
            
        time_since_last_training = datetime.now() - self.last_training_time
        
        if update_type == 'daily':
            # Retrain if it's been more than 3 days since last training
            return time_since_last_training > timedelta(days=3)
        else:  # weekly
            # Retrain if it's been more than 7 days since last training
            return time_since_last_training > timedelta(days=7)
    
    def retrain_model(self):
        """Retrain the collaborative filtering model"""
        try:
            logger.info("Starting model retraining")
            
            # Prepare data
            data = self.data_processor.prepare_data()
            
            # Train model
            self.model.train(data)
            
            # Update last training time
            self.last_training_time = datetime.now()
            
            logger.info("Model retraining completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            raise
    
    def get_model_status(self) -> dict:
        """
        Get current model status
        
        Returns:
            dict: Model status information
        """
        return {
            'last_training_time': self.last_training_time,
            'model_type': 'collaborative_filtering',
            'is_trained': self.last_training_time is not None
        } 