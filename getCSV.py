import time
import os
import glob
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import zipfile
import shutil

#issue - it doesn't auto put in the login credentilas
# go thru the CSS of the login page and see if there is a way to auto fill the login credentials
# if not, then we need to use the selenium to fill in the login credentials

class SeatDataScraper:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.driver = None
        self.wait = None
        
    def setup_driver(self):
        """Setup Chrome WebDriver with appropriate options"""
        chrome_options = Options()
        
        # Set download directory to current directory
        current_dir = os.getcwd()
        prefs = {
            "download.default_directory": current_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Add other useful options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 20)
        
    def login(self):
        """Login to seatdata.io"""
        try:
            print("Navigating to login page...")
            self.driver.get("https://seatdata.io/login/")
            
            # Wait for login form to load
            email_field = self.wait.until(EC.presence_of_element_located((By.NAME, "email")))
            password_field = self.driver.find_element(By.NAME, "password")
            
            # Enter credentials
            print("Entering login credentials...")
            email_field.clear()
            email_field.send_keys(self.username)
            password_field.clear()
            password_field.send_keys(self.password)
            
            # Find and click login button
            login_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            # Wait for redirect to dashboard
            print("Waiting for login to complete...")
            self.wait.until(EC.url_contains("dashboard"))
            print("Successfully logged in!")
            
        except Exception as e:
            print(f"Login failed: {str(e)}")
            raise
    
    def navigate_to_sold_listings(self):
        """Navigate to the sold listings page"""
        try:
            print("Navigating to sold listings page...")
            self.driver.get("https://seatdata.io/dashboard/sold-listings")
            
            # Wait for page to load
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            print("Successfully navigated to sold listings page")
            
        except Exception as e:
            print(f"Navigation failed: {str(e)}")
            raise
    
    def search_mariners(self):
        """Search for 'mariners' in the search bar"""
        try:
            print("Searching for 'mariners'...")
            
            # Wait for search input to be available
            search_input = self.wait.until(EC.presence_of_element_located((
                By.XPATH, "//input[@placeholder='Search...' or @type='search' or contains(@class, 'search')]"
            )))
            
            # Clear and enter search term
            search_input.clear()
            search_input.send_keys("mariners")
            search_input.send_keys(Keys.RETURN)
            
            # Wait for search results to load
            time.sleep(3)
            print("Search completed")
            
        except Exception as e:
            print(f"Search failed: {str(e)}")
            raise
    
    def download_csv_files(self, num_files=7):
        """Download the first N CSV files"""
        try:
            print(f"Attempting to download {num_files} CSV files...")
            
            # Wait for results to load
            time.sleep(5)
            
            # Look for download buttons or links
            download_buttons = self.driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Download') or contains(@class, 'download') or contains(@aria-label, 'download')]")
            
            if not download_buttons:
                # Try alternative selectors
                download_buttons = self.driver.find_elements(By.XPATH, 
                    "//a[contains(@href, '.csv') or contains(text(), 'CSV')]")
            
            if not download_buttons:
                # Look for any clickable elements that might trigger downloads
                download_buttons = self.driver.find_elements(By.XPATH, 
                    "//*[contains(@class, 'btn') or contains(@class, 'button')]")
            
            print(f"Found {len(download_buttons)} potential download elements")
            
            # Download the first N files
            downloaded_count = 0
            for i, button in enumerate(download_buttons[:num_files]):
                try:
                    print(f"Attempting to download file {i+1}...")
                    
                    # Scroll to button if needed
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                    time.sleep(1)
                    
                    # Click the download button
                    button.click()
                    time.sleep(3)  # Wait for download to start
                    
                    downloaded_count += 1
                    print(f"Download initiated for file {i+1}")
                    
                except Exception as e:
                    print(f"Failed to download file {i+1}: {str(e)}")
                    continue
            
            print(f"Download process completed. {downloaded_count} files downloaded.")
            
        except Exception as e:
            print(f"Download process failed: {str(e)}")
            raise
    
    def wait_for_downloads(self, timeout=60):
        """Wait for downloads to complete"""
        print("Waiting for downloads to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for .crdownload files (Chrome download in progress)
            crdownload_files = glob.glob("*.crdownload")
            if not crdownload_files:
                print("All downloads appear to be complete")
                break
            time.sleep(2)
        
        # List downloaded files
        csv_files = glob.glob("*.csv")
        print(f"Downloaded CSV files: {csv_files}")
    
    def cleanup(self):
        """Clean up the WebDriver"""
        if self.driver:
            self.driver.quit()
            print("Browser closed")
    
    def run(self, num_files=7):
        """Run the complete scraping process"""
        try:
            self.setup_driver()
            self.login()
            self.navigate_to_sold_listings()
            self.search_mariners()
            self.download_csv_files(num_files)
            self.wait_for_downloads()
            
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
        finally:
            self.cleanup()

def main():
    # Credentials
    username = "ztareen@purdue.edu"
    password = "q3yS$$%h8yp51eXK"
    
    # Create scraper instance
    scraper = SeatDataScraper(username, password)
    
    # Run the scraping process
    scraper.run(num_files=7)

if __name__ == "__main__":
    main()

