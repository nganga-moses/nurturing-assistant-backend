import logging
from batch_processing import BatchProcessingScheduler
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    scheduler.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start scheduler
    scheduler = BatchProcessingScheduler()
    
    try:
        logger.info("Starting batch processing scheduler")
        scheduler.start()
        
        # Keep the main thread alive
        signal.pause()
        
    except Exception as e:
        logger.error(f"Error running batch processor: {e}")
        scheduler.stop()
        sys.exit(1) 