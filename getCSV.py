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

class SeatDataScraper:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.driver = None
        self.wait = None
        
    def setup_driver(self):
        chrome_options = Options()
        current_dir = os.getcwd()
        prefs = {
            "download.default_directory": current_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
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
        if self.driver is None or self.wait is None:
            raise Exception("WebDriver not initialized. Call setup_driver() first.")
        
    def login(self):
        try:
            self.check_driver()
            print("Navigating to login page...")
            self.driver.get("https://seatdata.io/login/")
            time.sleep(3)
            
            email_field = None
            email_selectors = [
                (By.NAME, "login"),
                (By.ID, "inputEmail"),
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
            
            password_field = None
            password_selectors = [
                (By.NAME, "password"),
                (By.ID, "inputPassword"),
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
            
            self.driver.execute_script("arguments[0].scrollIntoView(true);", email_field)
            time.sleep(1)
            self.driver.execute_script("arguments[0].value = '';", email_field)
            time.sleep(0.5)
            self.driver.execute_script("arguments[0].value = arguments[1];", email_field, self.username)
            time.sleep(0.5)
            email_field.clear()
            email_field.send_keys(self.username)
            print(f"Entered email: {self.username}")
            
            self.driver.execute_script("arguments[0].scrollIntoView(true);", password_field)
            time.sleep(1)
            self.driver.execute_script("arguments[0].value = '';", password_field)
            time.sleep(0.5)
            self.driver.execute_script("arguments[0].value = arguments[1];", password_field, self.password)
            time.sleep(0.5)
            password_field.clear()
            password_field.send_keys(self.password)
            print("Entered password")
            
            login_button = None
            button_selectors = [
                (By.XPATH, "//button[@type='submit' and contains(text(), 'Sign in')]"),
                (By.CSS_SELECTOR, "button.btn.btn-lg.btn-primary[type='submit']"),
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
            
            self.driver.execute_script("arguments[0].scrollIntoView(true);", login_button)
            time.sleep(1)
            try:
                self.driver.execute_script("arguments[0].click();", login_button)
                print("Clicked login button using JavaScript")
            except:
                login_button.click()
                print("Clicked login button using regular click")
            
            print("Waiting for login to complete...")
            try:
                self.wait.until(EC.url_contains("dashboard"))
                print("Successfully logged in!")
            except TimeoutException:
                current_url = self.driver.current_url
                print(f"Current URL after login attempt: {current_url}")
                error_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'error') or contains(text(), 'invalid') or contains(text(), 'incorrect') or contains(text(), 'failed')]")
                if error_elements:
                    error_text = error_elements[0].text
                    print(f"Login failed with error: {error_text}")
                    raise Exception(f"Login failed: {error_text}")
                else:
                    time.sleep(5)
                    current_url = self.driver.current_url
                    if "dashboard" in current_url or "login" not in current_url:
                        print("Login appears to have succeeded (delayed redirect)")
                    else:
                        print("Login may have failed - no dashboard redirect detected")
        except Exception as e:
            print(f"Login failed: {str(e)}")
            try:
                self.driver.save_screenshot("login_error.png")
                print("Screenshot saved as login_error.png")
                print("Current page title:", self.driver.title)
                print("Current URL:", self.driver.current_url)
            except:
                pass
            raise

    def navigate_to_sold_listings(self):
        try:
            self.check_driver()
            print("Looking for 'Sold Tickets by Event' link in sidebar...")
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "sidebar")))
            sold_listings_selectors = [
                (By.XPATH, "//a[@href='/dashboard/sold-listings']"),
                (By.XPATH, "//a[contains(@href, 'sold-listings')]"),
                (By.XPATH, "//a[contains(text(), 'Sold Tickets by Event')]"),
                (By.XPATH, "//span[contains(text(), 'Sold Tickets by Event')]/parent::a"),
                (By.CSS_SELECTOR, "a.sidebar-link[href*='sold-listings']"),
                (By.XPATH, "//li[contains(@class, 'sidebar-item')]//a[contains(@href, 'sold-listings')]"),
            ]
            sold_listings_link = None
            for selector_type, selector_value in sold_listings_selectors:
                try:
                    sold_listings_link = self.wait.until(EC.element_to_be_clickable((selector_type, selector_value)))
                    print(f"Found sold listings link using {selector_type}: {selector_value}")
                    break
                except TimeoutException:
                    print(f"Selector failed: {selector_type}: {selector_value}")
                    continue
            if not sold_listings_link:
                print("Trying to find all sidebar links...")
                sidebar_links = self.driver.find_elements(By.XPATH, "//li[contains(@class, 'sidebar-item')]//a")
                for link in sidebar_links:
                    try:
                        link_text = link.text.strip()
                        link_href = link.get_attribute('href')
                        print(f"Found sidebar link: '{link_text}' -> {link_href}")
                        if 'sold-listings' in str(link_href) or 'Sold Tickets' in link_text:
                            sold_listings_link = link
                            print(f"Found target link: {link_text}")
                            break
                    except:
                        continue
            if not sold_listings_link:
                raise Exception("Could not find the sold listings link in the sidebar")
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", sold_listings_link)
            time.sleep(2)
            current_url_before = self.driver.current_url
            print(f"Current URL before click: {current_url_before}")
            try:
                sold_listings_link.click()
                print("Clicked sold listings link using regular click")
            except Exception as e:
                print(f"Regular click failed: {e}")
                self.driver.execute_script("arguments[0].click();", sold_listings_link)
                print("Clicked sold listings link using JavaScript click")
            print("Waiting for navigation to sold listings page...")
            try:
                self.wait.until(lambda driver: driver.current_url != current_url_before)
                print("URL changed, navigation in progress...")
            except TimeoutException:
                print("URL didn't change immediately, checking current URL...")
            time.sleep(5)
            final_url = self.driver.current_url
            print(f"Final URL: {final_url}")
            if "sold-listings" in final_url:
                print("✅ Successfully navigated to sold listings page!")
            else:
                print(f"⚠️ May not have reached sold listings page. Current URL: {final_url}")
                print("Trying direct navigation as backup...")
                self.driver.get("https://seatdata.io/dashboard/sold-listings")
                time.sleep(3)
                final_url = self.driver.current_url
                if "sold-listings" in final_url:
                    print("✅ Direct navigation to sold listings successful!")
                else:
                    print(f"❌ Direct navigation also failed. Final URL: {final_url}")
            try:
                self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                print("✅ Page content loaded successfully")
            except TimeoutException:
                print("⚠️ Page content may not have loaded completely")
        except Exception as e:
            print(f"❌ Navigation failed: {str(e)}")
            try:
                self.driver.save_screenshot("navigation_error.png")
                print("Screenshot saved as navigation_error.png")
            except:
                pass
            raise

    def search_mariners(self):
        """Search for 'seattle mariners' in the search bar on the sold listings page"""
        try:
            self.check_driver()
            print("Searching for 'seattle mariners'...")
            # Wait for the search input with id 'dt-search-1' and class 'dt-input'
            search_input = self.wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, "input#dt-search-1.dt-input"
            )))
            search_input.clear()
            search_input.send_keys("seattle mariners")
            search_input.send_keys(Keys.RETURN)
            time.sleep(3)
            print("Search for 'seattle mariners' completed")
        except Exception as e:
            print(f"Search failed: {str(e)}")
            raise

    def download_csv_files(self, num_files=7):
        try:
            self.check_driver()
            print(f"Downloading CSV files, limiting to {num_files} files")
            csv_links = []
            for i in range(1, num_files + 1):
                try:
                    link = self.wait.until(EC.element_to_be_clickable((
                        By.XPATH, f"//table[@id='example']/tbody/tr[{i}]/td[7]/a"
                    )))
                    csv_links.append(link)
                    print(f"Found CSV link for row {i}")
                except Exception as e:
                    print(f"Could not find CSV link for row {i}: {str(e)}")
                    continue
            for link in csv_links:
                try:
                    link.click()
                    print("Clicked on CSV link")
                    time.sleep(2)
                except Exception as e:
                    print(f"Error clicking on CSV link: {str(e)}")
                    continue
            print("CSV download initiated")
        except Exception as e:
            print(f"Error during CSV download: {str(e)}")
            raise

    def wait_for_downloads(self):
        try:
            self.check_driver()
            print("Waiting for downloads to complete...")
            time.sleep(5)
            downloads_dir = os.getcwd()
            while True:
                time.sleep(5)
                if all(not fnmatch.fnmatch(x, '*.crdownload') for x in os.listdir(downloads_dir)):
                    print("All downloads completed")
                    break
                else:
                    print("Waiting for downloads to finish...")
        except Exception as e:
            print(f"Error while waiting for downloads: {str(e)}")
            raise

    def cleanup(self):
        if self.driver:
            self.driver.quit()
            print("Browser closed")

    def run(self, num_files=7):
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
    username = "ztareen@purdue.edu"
    password = "q3yS$$%h8yp51eXK"
    scraper = SeatDataScraper(username, password)
    try:
        scraper.run(num_files=7)
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        input("Press Enter to close the browser...")
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    # test_login_only()
    main()

    #make it click on event
    #download csv
    #rename csv to: like "Kansas City Royals at Seattle Mariners - 2023-09-30.csv"
    # thats where 2023-09-30 is the date of the event