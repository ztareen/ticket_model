import schedule
import time
import subprocess
import sys
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers= [
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

def run_ticketmaster_script():
    """Run the ticketmaster event getter script"""
    try:
        logging.info("Starting ticketmaster event collection...")
        
        # Run the script
        result = subprocess.run([sys.executable, 'ticketmaster_eventgetter.py'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            logging.info("Ticketmaster script completed successfully")
            logging.info(f"Output: {result.stdout}")
        else:
            logging.error(f"Ticketmaster script failed with return code {result.returncode}")
            logging.error(f"Error: {result.stderr}")
            
    except Exception as e:
        logging.error(f"Error running ticketmaster script: {e}")

def main():
    """Main scheduler function"""
    logging.info("Starting ticket data collection scheduler...")
    
    # Schedule the script to run daily at 9:00 AM
    schedule.every().day.at("09:00").do(run_ticketmaster_script)
    
    # You can also schedule it to run multiple times per day
    # schedule.every(6).hours.do(run_ticketmaster_script)  # Every 6 hours
    # schedule.every().monday.at("09:00").do(run_ticketmaster_script)  # Every Monday at 9 AM
    
    # For testing, you can run it immediately
    # run_ticketmaster_script()
    
    logging.info("Scheduler started. Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")

if __name__ == "__main__":
    main() 