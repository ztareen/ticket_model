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
        
        try:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.wait = WebDriverWait(self.driver, 20)
            print("Chrome WebDriver setup completed successfully")
        except Exception as e:
            print(f"Failed to setup Chrome WebDriver: {str(e)}")
            raise
    
    def check_driver(self):
        """Check if driver is properly initialized"""
        if self.driver is None or self.wait is None:
            raise Exception("WebDriver not initialized. Call setup_driver() first.")
        
    def login(self):
        """Login to seatdata.io"""
        try:
            # Ensure driver is initialized
            self.check_driver()
            
            print("Navigating to login page...")
            self.driver.get("https://seatdata.io/login/")
            
            # Wait for page to fully load
            time.sleep(3)
            
            # Try multiple strategies to find the email field - FIXED ORDER based on HTML
            email_field = None
            email_selectors = [
                (By.NAME, "login"),  # This is the actual name from HTML
                (By.ID, "inputEmail"),  # This is the actual ID from HTML
                (By.NAME, "email"),
                (By.ID, "email"),
                (By.XPATH, "//input[@type='email']"),
                (By.XPATH, "//input[contains(@placeholder, 'email') or contains(@placeholder, 'Email')]"),
                (By.CSS_SELECTOR, "input[type='email']"),
                (By.CSS_SELECTOR, "input[name*='email']"),
                (By.CSS_SELECTOR, "input[id*='email']")
            ]
            
            for selector_type, selector_value in email_selectors:
                try:
                    email_field = self.wait.until(EC.presence_of_element_located((selector_type, selector_value)))
                    print(f"Found email field using {selector_type}: {selector_value}")
                    break
                except TimeoutException:
                    continue
            
            if not email_field:
                raise Exception("Could not find email field with any selector")
            
            # Try multiple strategies to find the password field - FIXED ORDER based on HTML
            password_field = None
            password_selectors = [
                (By.NAME, "password"),  # This matches the HTML
                (By.ID, "inputPassword"),  # This is the actual ID from HTML
                (By.ID, "password"),
                (By.XPATH, "//input[@type='password']"),
                (By.XPATH, "//input[contains(@placeholder, 'password') or contains(@placeholder, 'Password')]"),
                (By.CSS_SELECTOR, "input[type='password']"),
                (By.CSS_SELECTOR, "input[name*='password']"),
                (By.CSS_SELECTOR, "input[id*='password']")
            ]
            
            for selector_type, selector_value in password_selectors:
                try:
                    password_field = self.driver.find_element(selector_type, selector_value)
                    print(f"Found password field using {selector_type}: {selector_value}")
                    break
                except NoSuchElementException:
                    continue
            
            if not password_field:
                raise Exception("Could not find password field with any selector")
            
            # Enter credentials with explicit waits and clearing
            print("Entering login credentials...")
            
            # Clear and enter email
            self.driver.execute_script("arguments[0].scrollIntoView(true);", email_field)
            time.sleep(1)
            
            # Use JavaScript to clear and set value to avoid issues
            self.driver.execute_script("arguments[0].value = '';", email_field)
            time.sleep(0.5)
            self.driver.execute_script("arguments[0].value = arguments[1];", email_field, self.username)
            time.sleep(0.5)
            
            # Also try with send_keys as backup
            email_field.clear()
            email_field.send_keys(self.username)
            print(f"Entered email: {self.username}")
            
            # Clear and enter password
            self.driver.execute_script("arguments[0].scrollIntoView(true);", password_field)
            time.sleep(1)
            
            # Use JavaScript to clear and set value to avoid issues
            self.driver.execute_script("arguments[0].value = '';", password_field)
            time.sleep(0.5)
            self.driver.execute_script("arguments[0].value = arguments[1];", password_field, self.password)
            time.sleep(0.5)
            
            # Also try with send_keys as backup
            password_field.clear()
            password_field.send_keys(self.password)
            print("Entered password")
            
            # Try multiple strategies to find the login button - UPDATED based on HTML
            login_button = None
            button_selectors = [
                (By.XPATH, "//button[@type='submit' and contains(text(), 'Sign in')]"),  # Specific to HTML
                (By.CSS_SELECTOR, "button.btn.btn-lg.btn-primary[type='submit']"),  # Specific to HTML classes
                (By.XPATH, "//button[@type='submit']"),
                (By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Sign In') or contains(text(), 'Log In') or contains(text(), 'Sign in')]"),
                (By.XPATH, "//input[@type='submit']"),
                (By.CSS_SELECTOR, "button[type='submit']"),
                (By.CSS_SELECTOR, "input[type='submit']"),
                (By.CSS_SELECTOR, ".login-button, .signin-button, .btn-primary"),
                (By.XPATH, "//*[contains(@class, 'btn') and contains(@class, 'primary')]")
            ]
            
            for selector_type, selector_value in button_selectors:
                try:
                    login_button = self.driver.find_element(selector_type, selector_value)
                    print(f"Found login button using {selector_type}: {selector_value}")
                    break
                except NoSuchElementException:
                    continue
            
            if not login_button:
                raise Exception("Could not find login button with any selector")
            
            # Click the login button
            self.driver.execute_script("arguments[0].scrollIntoView(true);", login_button)
            time.sleep(1)
            
            # Try JavaScript click first, then regular click
            try:
                self.driver.execute_script("arguments[0].click();", login_button)
                print("Clicked login button using JavaScript")
            except:
                login_button.click()
                print("Clicked login button using regular click")
            
            # Wait for redirect to dashboard or check for errors
            print("Waiting for login to complete...")
            try:
                self.wait.until(EC.url_contains("dashboard"))
                print("Successfully logged in!")
            except TimeoutException:
                # Check if we're still on login page or if there's an error
                current_url = self.driver.current_url
                print(f"Current URL after login attempt: {current_url}")
                
                # Check if there's an error message
                error_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'error') or contains(text(), 'Error') or contains(text(), 'invalid') or contains(text(), 'Invalid') or contains(text(), 'incorrect') or contains(text(), 'failed')]")
                if error_elements:
                    error_text = error_elements[0].text
                    print(f"Login failed with error: {error_text}")
                    raise Exception(f"Login failed: {error_text}")
                else:
                    # Wait a bit more and check URL again
                    time.sleep(5)
                    current_url = self.driver.current_url
                    if "dashboard" in current_url or "login" not in current_url:
                        print("Login appears to have succeeded (delayed redirect)")
                    else:
                        print("Login may have failed - no dashboard redirect detected")
                        # Continue anyway to see what happens
            
        except Exception as e:
            print(f"Login failed: {str(e)}")
            # Take a screenshot for debugging
            try:
                self.driver.save_screenshot("login_error.png")
                print("Screenshot saved as login_error.png")
                
                # Also print page source for debugging
                print("Current page title:", self.driver.title)
                print("Current URL:", self.driver.current_url)
            except:
                pass
            raise
    
    def navigate_to_sold_listings(self):
        """Navigate to the sold listings page by direct URL"""
        try:
            # Ensure driver is initialized
            self.check_driver()
            
            print("Navigating directly to sold listings page...")
            
            # Direct navigation to the sold listings URL
            self.driver.get("https://seatdata.io/dashboard/sold-listings")
            
            # Wait for page to load
            time.sleep(3)
            
            # Verify we're on the correct page
            current_url = self.driver.current_url
            print(f"Current URL: {current_url}")
            
            if "sold-listings" in current_url:
                print("✅ Successfully navigated to sold listings page!")
            else:
                print(f"⚠️ Unexpected URL: {current_url}")
            
            # Wait for page content to load
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            print("✅ Page loaded successfully")
            
        except Exception as e:
            print(f"❌ Navigation failed: {str(e)}")
            raise
    
    def search_mariners(self):
        """Search for 'mariners' in the search bar"""
        try:
            # Ensure driver is initialized
            self.check_driver()
            
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
            # Ensure driver is initialized
            self.check_driver()
            
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
    
    try:
        # Run the scraping process
        scraper.run(num_files=7)
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        # Keep browser open for debugging if there's an error
        input("Press Enter to close the browser...")
    finally:
        scraper.cleanup()

def test_login_only():
    """Test only the login functionality"""
    username = "ztareen@purdue.edu"
    password = "q3yS$$%h8yp51eXK"
    
    scraper = SeatDataScraper(username, password)
    
    try:
        print("Testing login functionality...")
        scraper.setup_driver()
        scraper.login()
        print("Login test completed successfully!")
        input("Press Enter to close the browser...")
    except Exception as e:
        print(f"Login test failed: {str(e)}")
        input("Press Enter to close the browser...")
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    # Uncomment the line below to test only login functionality
    test_login_only()
    
    # Run the full scraping process
    # main()


    #not able to just go to the other page?