# Ticketmaster Event Data Collection - Automation Guide

This guide explains how to automatically run your `ticketmaster_eventgetter.py` script using different methods.

## Prerequisites

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Automation Options

### Option 1: Python Scheduler (Recommended)

Use the `scheduler.py` script which uses the `schedule` library:

```bash
python scheduler.py
```

**Features:**
- Runs daily at 9:00 AM by default
- Logs all activities to `scheduler.log`
- Easy to modify schedule (every 6 hours, weekly, etc.)
- Can be stopped with Ctrl+C

**To modify the schedule, edit `scheduler.py`:**
```python
# Daily at 9 AM (default)
schedule.every().day.at("09:00").do(run_ticketmaster_script)

# Every 6 hours
schedule.every(6).hours.do(run_ticketmaster_script)

# Every Monday at 9 AM
schedule.every().monday.at("09:00").do(run_ticketmaster_script)

# Every 30 minutes
schedule.every(30).minutes.do(run_ticketmaster_script)
```

### Option 2: Enhanced Auto Script

Use the `auto_ticketmaster.py` script which has built-in scheduling:

```bash
# Run once
python auto_ticketmaster.py

# Run once and exit
python auto_ticketmaster.py --once

# Run automatically every 24 hours
python auto_ticketmaster.py --auto

# Run automatically every 6 hours
python auto_ticketmaster.py --auto --interval 6
```

**Features:**
- No external dependencies needed
- Built-in logging to `ticketmaster_auto.log`
- Configurable intervals
- Error handling and retry logic

### Option 3: Windows Task Scheduler

1. **Create the batch file** (already created as `run_ticketmaster.bat`)

2. **Set up Windows Task Scheduler:**
   - Open "Task Scheduler" (search in Start menu)
   - Click "Create Basic Task"
   - Name: "Ticketmaster Data Collection"
   - Trigger: Daily (or your preferred frequency)
   - Action: Start a program
   - Program: `C:\Users\zarak\OneDrive\Documents\GitHub\ticket_model\run_ticketmaster.bat`
   - Finish

3. **Advanced settings:**
   - Right-click the task → Properties
   - General tab: Check "Run whether user is logged on or not"
   - Settings tab: Configure retry options

### Option 4: PowerShell Script

Create a PowerShell script for more control:

```powershell
# run_ticketmaster.ps1
Set-Location "C:\Users\zarak\OneDrive\Documents\GitHub\ticket_model"
python ticketmaster_eventgetter.py
Write-Output "Script completed at $(Get-Date)" | Out-File -Append run_log.txt
```

## Running as a Windows Service

To run the script as a Windows service (runs even when not logged in):

1. **Install NSSM (Non-Sucking Service Manager):**
   - Download from: https://nssm.cc/
   - Extract to a folder

2. **Create the service:**
   ```cmd
   nssm install TicketmasterCollector "C:\Python39\python.exe" "C:\Users\zarak\OneDrive\Documents\GitHub\ticket_model\auto_ticketmaster.py --auto"
   nssm set TicketmasterCollector AppDirectory "C:\Users\zarak\OneDrive\Documents\GitHub\ticket_model"
   nssm start TicketmasterCollector
   ```

3. **Manage the service:**
   ```cmd
   nssm start TicketmasterCollector
   nssm stop TicketmasterCollector
   nssm remove TicketmasterCollector
   ```

## Monitoring and Logs

All automation methods create log files:
- `scheduler.log` - For the scheduler script
- `ticketmaster_auto.log` - For the auto script
- `run_log.txt` - For batch file runs

Check these files to monitor script execution and troubleshoot issues.

## Recommended Setup

For most users, I recommend **Option 2** (`auto_ticketmaster.py`) because:
- No external dependencies
- Built-in error handling
- Easy to configure
- Runs continuously without external schedulers

**To start:**
```bash
python auto_ticketmaster.py --auto --interval 24
```

This will run the script every 24 hours and keep running until you stop it with Ctrl+C.

## Troubleshooting

1. **Script not running:** Check the log files for error messages
2. **Permission issues:** Run as administrator or check file permissions
3. **Python not found:** Make sure Python is in your PATH or use full path
4. **API rate limits:** The script includes delays, but you may need to adjust intervals

## Customization

You can modify any of these scripts to:
- Change the search keywords
- Adjust the date range
- Add email notifications
- Integrate with other data processing scripts
- Send data to databases instead of CSV files 