@echo off
cd /d "C:\Users\zarak\OneDrive\Documents\GitHub\ticket_model"
python ticketmaster_eventgetter.py
echo Script completed at %date% %time% >> run_log.txt 