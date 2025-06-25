import requests
import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
import time
import threading
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ticketmaster_auto.log'),
        logging.StreamHandler()
    ]
)

class TicketmasterAutoCollector:
    def __init__(self, run_interval_hours=24):
        self.run_interval_hours = run_interval_hours
        self.running = False
        self.thread = None
        
    def collect_event_data(self):
        """Main function to collect event data from Ticketmaster"""
        event_info_list = []

        uselessKeywords = [
            "Pinstripe Pass", "Premium Seating", "Group Tickets",
            "Suite", "Club Access", "Ballpark Tours", "Pregame Tours", "Non-Gameday"
        ]

        keyword = "seattle_mariners"
        start_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        end_utc = (start_utc + timedelta(days=7)).replace(hour=23, minute=59, second=59)

        start_utc_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc_str = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        url = (
            f'https://app.ticketmaster.com/discovery/v2/events.json'
            f'?apikey=KntBcmnLD8BBG0DpsV1GMMakFSFcoaof'
            f'&keyword=Seattle%20Mariners'
            f'&startDateTime={start_utc_str}'
            f'&endDateTime={end_utc_str}'
            f'&size=20'
        )

        try:
            response = requests.get(url)
            if response.status_code != 200:
                logging.error(f"Request failed with status code {response.status_code}")
                return

            data = response.json()
            events = data["_embedded"]["events"]
            
            logging.info(f"Found {len(events)} events to process")

            for i, event in enumerate(events, start=1):
                name = event["name"]
                event_id = event["id"]

                if any(kw.lower() in name.lower() for kw in uselessKeywords):
                    continue

                logging.info(f"Processing Event {i}: {name}")

                venue = event.get("_embedded", {}).get("venues", [{}])[0]

                event_info = {
                    "event_id": event_id,
                    "event_name": name,
                    "url": event.get("url", ""),
                    "start_date": event["dates"]["start"].get("localDate", ""),
                    "start_time": event["dates"]["start"].get("localTime", ""),
                    "datetime_utc": event["dates"]["start"].get("dateTime", ""),
                    "timezone": event["dates"].get("timezone", ""),
                    "ticket_status": event["dates"]["status"].get("code", ""),
                    "public_sale_start": event["sales"]["public"].get("startDateTime", ""),
                    "public_sale_end": event["sales"]["public"].get("endDateTime", ""),
                    "presales": json.dumps([
                        {
                            "name": p.get("name", ""),
                            "start": p.get("startDateTime", ""),
                            "end": p.get("endDateTime", "")
                        } for p in event["sales"].get("presales", [])
                    ]),
                    "image": event["images"][0]["url"] if event.get("images") else "",
                    "info": event.get("info", ""),
                    "please_note": event.get("pleaseNote", ""),
                    "seatmap_url": event.get("seatmap", {}).get("staticUrl", ""),
                    "accessibility_info": event.get("accessibility", {}).get("info", ""),
                    "ticket_limit": event.get("ticketLimit", {}).get("info", ""),

                    # Venue Info
                    "venue_id": venue.get("id", ""),
                    "venue_name": venue.get("name", ""),
                    "city": venue.get("city", {}).get("name", ""),
                    "state": venue.get("state", {}).get("name", ""),
                    "country": venue.get("country", {}).get("name", ""),
                    "venue_timezone": venue.get("timezone", ""),

                    # Classification
                    "segment": event["classifications"][0]["segment"]["name"] if "classifications" in event else "",
                    "genre": event["classifications"][0]["genre"]["name"] if "classifications" in event else "",
                    "subGenre": event["classifications"][0]["subGenre"]["name"] if "classifications" in event else "",
                }

                # Get detailed event information
                url_EventDetail = (
                    f'https://app.ticketmaster.com/discovery/v2/events/{event_id}.json?apikey=KntBcmnLD8BBG0DpsV1GMMakFSFcoaof'
                    f'&id={event_id}'
                )

                detail_resp = requests.get(url_EventDetail)
                if detail_resp.status_code == 200:
                    detail_data = detail_resp.json()
                    
                    # Safely pull optional or nested fields
                    event_info.update({
                        "type": detail_data.get("type", ""),
                        "location": json.dumps(detail_data.get("location", {})),
                        "name": detail_data.get("name", ""),
                        "dates": json.dumps(detail_data.get("dates", {})),
                        "sales": json.dumps(detail_data.get("sales", {})),
                        "priceRanges": json.dumps(detail_data.get("priceRanges", [])),
                        "promoter": json.dumps(detail_data.get("promoter", {})),
                        "promoters": json.dumps(detail_data.get("promoters", [])),
                        "seatmap": json.dumps(detail_data.get("seatmap", {})),
                        "accessibility": json.dumps(detail_data.get("accessibility", {})),
                        "ticketLimit": json.dumps(detail_data.get("ticketLimit", {})),
                        "classifications": json.dumps(detail_data.get("classifications", [])),
                        "externalLinks": json.dumps(detail_data.get("externalLinks", {})),
                    })

                event_info_list.append(event_info)

            # Save to CSV
            if event_info_list:
                df = pd.DataFrame(event_info_list)
                filename = f"event_data_{datetime.now().strftime('%Y-%m-%d')}.csv"
                df.to_csv(filename, index=False)
                logging.info(f"Saved {len(df)} events to {filename}")
            else:
                logging.warning("No events found to save")

        except Exception as e:
            logging.error(f"Error collecting event data: {e}")

    def run_continuously(self):
        """Run the collector continuously at specified intervals"""
        while self.running:
            try:
                logging.info("Starting scheduled data collection...")
                self.collect_event_data()
                logging.info(f"Data collection completed. Next run in {self.run_interval_hours} hours.")
                
                # Sleep for the specified interval
                time.sleep(self.run_interval_hours * 3600)  # Convert hours to seconds
                
            except Exception as e:
                logging.error(f"Error in continuous run: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def start(self):
        """Start the automatic collection"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_continuously)
            self.thread.daemon = True
            self.thread.start()
            logging.info(f"Auto collector started. Running every {self.run_interval_hours} hours.")

    def stop(self):
        """Stop the automatic collection"""
        self.running = False
        if self.thread:
            self.thread.join()
        logging.info("Auto collector stopped.")

def main():
    """Main function to run the collector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Ticketmaster Event Collector')
    parser.add_argument('--auto', action='store_true', help='Run in automatic mode')
    parser.add_argument('--interval', type=int, default=24, help='Interval in hours between runs (default: 24)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    collector = TicketmasterAutoCollector(run_interval_hours=args.interval)
    
    if args.once:
        logging.info("Running single collection...")
        collector.collect_event_data()
    elif args.auto:
        logging.info("Starting automatic collection...")
        collector.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Stopping automatic collection...")
            collector.stop()
    else:
        # Default: run once
        logging.info("Running single collection...")
        collector.collect_event_data()

if __name__ == "__main__":
    main() 