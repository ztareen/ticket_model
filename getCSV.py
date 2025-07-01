import time
import os
import glob
import fnmatch
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
    def __init__(self, username, password, search_term="seattle mariners"):
        self.username = username
        self.password = password
        self.search_term = search_term
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

    def search_events(self):
        """Search for events using the search term"""
        try:
            self.check_driver()
            print(f"Searching for '{self.search_term}'...")
            # Wait for the search input with id 'dt-search-1' and class 'dt-input'
            search_input = self.wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, "input#dt-search-1.dt-input"
            )))
            search_input.clear()
            search_input.send_keys(self.search_term)
            search_input.send_keys(Keys.RETURN)
            time.sleep(3)
            print(f"Search for '{self.search_term}' completed")
        except Exception as e:
            print(f"Search failed: {str(e)}")
            raise

    def get_event_rows(self):
        """Get all event rows from the table"""
        try:
            self.check_driver()
            print("Getting event rows from table...")
            # Wait longer for the table to load and try alternative selectors
            try:
                table = WebDriverWait(self.driver, 40).until(
                    EC.presence_of_element_located((By.ID, "example"))
                )
            except TimeoutException:
                print("Table with id 'example' not found, trying alternative selectors...")
                # Try to find any table on the page
                tables = self.driver.find_elements(By.TAG_NAME, "table")
                if tables:
                    table = tables[0]
                    print("Using first table found on the page.")
                else:
                    print("No tables found on the page.")
                    return []
            # Get all rows from tbody (excluding header)
            rows = table.find_elements(By.XPATH, ".//tbody/tr")
            print(f"Found {len(rows)} event rows")
            return rows
        except Exception as e:
            print(f"Error getting event rows: {str(e)}")
            return []

    def extract_event_info(self, row):
        """Extract event information from a table row"""
        try:
            # Get event name from the first column
            name_cell = row.find_element(By.XPATH, ".//td[1]")
            event_name = name_cell.text.strip()
            
            # Get date from the second column
            date_cell = row.find_element(By.XPATH, ".//td[2]")
            event_date = date_cell.text.strip()
            
            return event_name, event_date
        except Exception as e:
            print(f"Error extracting event info: {str(e)}")
            return "Unknown Event", "Unknown Date"

    def click_event_and_download_csv(self, row, event_name, event_date):
        """Click on an event row to open sidebar and download CSV"""
        try:
            print(f"Processing event: {event_name} - {event_date}")
            
            # Scroll the row into view
            self.driver.execute_script("arguments[0].scrollIntoView(true);", row)
            time.sleep(1)
            
            # Check if this row has the expandable class
            row_classes = row.get_attribute('class') or ''
            print(f"Row classes: {row_classes}")
            
            # Look for the expand button or clickable element within the row
            # Try to find the element that actually triggers the sidebar
            expand_element = None
            
            # Try different selectors for the expand trigger
            expand_selectors = [
                (By.CSS_SELECTOR, "td.dt-control"),  # DataTables control column
                (By.CSS_SELECTOR, ".dtr-control"),   # Responsive control
                (By.XPATH, ".//td[contains(@class, 'control')]"),
                (By.XPATH, ".//button[contains(@class, 'dtr-control')]"),
                (By.XPATH, ".//span[contains(@class, 'dtr-control')]"),
            ]
            
            for selector_type, selector_value in expand_selectors:
                try:
                    expand_element = row.find_element(selector_type, selector_value)
                    print(f"Found expand element using {selector_type}: {selector_value}")
                    break
                except NoSuchElementException:
                    continue
            
            # If no specific expand element found, but row has dt-hasChild class, click the row
            if not expand_element and 'dt-hasChild' in row_classes:
                expand_element = row
                print("Using the row itself as expand element")
            
            # If still no expand element and row is expandable, try first td
            if not expand_element:
                try:
                    # Sometimes the first cell is the expand trigger
                    first_td = row.find_element(By.XPATH, ".//td[1]")
                    expand_element = first_td
                    print("Using first td as expand element")
                except NoSuchElementException:
                    pass
            
            if expand_element:
                try:
                    # Try regular click first
                    expand_element.click()
                    print("Clicked expand element with regular click")
                except Exception as e:
                    print(f"Regular click failed: {e}, trying JavaScript click")
                    self.driver.execute_script("arguments[0].click();", expand_element)
                    print("Clicked expand element using JavaScript")
                
                # Wait for the sidebar/details to appear
                time.sleep(3)
                
                # Check if sidebar or details panel appeared
                sidebar_appeared = False
                sidebar_selectors = [
                    (By.CSS_SELECTOR, ".sidebar"),
                    (By.CSS_SELECTOR, ".dtr-details"),
                    (By.CSS_SELECTOR, "[class*='detail']"),
                    (By.CSS_SELECTOR, "[class*='sidebar']"),
                    (By.ID, "download-csv"),  # Direct check for download button
                ]
                
                for selector_type, selector_value in sidebar_selectors:
                    try:
                        sidebar = self.driver.find_element(selector_type, selector_value)
                        if sidebar.is_displayed():
                            sidebar_appeared = True
                            print(f"Sidebar/details panel found using {selector_type}: {selector_value}")
                            break
                    except NoSuchElementException:
                        continue
                
                if not sidebar_appeared:
                    print("Sidebar may not have appeared, checking for any new elements...")
                    # Take screenshot for debugging
                    self.driver.save_screenshot(f"after_click_{event_name.replace(' ', '_')}.png")
            
            else:
                print("No expand element found, trying to click the row directly")
                try:
                    row.click()
                    print("Clicked on row directly")
                    time.sleep(3)
                except Exception as e:
                    print(f"Direct row click failed: {e}")
                    self.driver.execute_script("arguments[0].click();", row)
                    print("Clicked row using JavaScript")
                    time.sleep(3)
            
            # Look for the download CSV button
            download_button_selectors = [
                (By.ID, "download-csv"),
                (By.XPATH, "//button[contains(text(), 'Download CSV')]"),
                (By.XPATH, "//button[@id='download-csv']"),
                (By.CSS_SELECTOR, "button#download-csv"),
                (By.XPATH, "//button[contains(@class, 'btn') and contains(text(), 'CSV')]"),
                (By.XPATH, "//a[contains(text(), 'Download CSV')]"),  # In case it's a link
                (By.XPATH, "//a[contains(@href, 'csv')]"),
            ]
            
            download_button = None
            for selector_type, selector_value in download_button_selectors:
                try:
                    download_button = self.wait.until(EC.element_to_be_clickable((selector_type, selector_value)))
                    print(f"Found download CSV button using {selector_type}: {selector_value}")
                    break
                except TimeoutException:
                    continue
            
            if not download_button:
                print("Could not find download CSV button, searching all buttons...")
                # Try to find any button with "CSV" in the text
                all_buttons = self.driver.find_elements(By.TAG_NAME, "button")
                all_links = self.driver.find_elements(By.TAG_NAME, "a")
                
                for element in all_buttons + all_links:
                    try:
                        element_text = element.text.lower()
                        if "csv" in element_text or "download" in element_text:
                            download_button = element
                            print(f"Found potential download element with text: {element.text}")
                            break
                    except:
                        continue
            
            if download_button:
                try:
                    # Scroll download button into view
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
                    time.sleep(1)
                    
                    download_button.click()
                    print("Clicked download CSV button")
                    time.sleep(3)  # Wait for download to start
                    
                    # Try to rename the downloaded file
                    self.rename_downloaded_file(event_name, event_date)
                    
                except Exception as e:
                    print(f"Error clicking download button: {str(e)}")
                    try:
                        self.driver.execute_script("arguments[0].click();", download_button)
                        print("Used JavaScript click for download button")
                        time.sleep(3)
                        self.rename_downloaded_file(event_name, event_date)
                    except Exception as e2:
                        print(f"JavaScript click also failed: {str(e2)}")
            else:
                print("Could not find download CSV button")
                # Take screenshot for debugging
                self.driver.save_screenshot(f"no_download_button_{event_name.replace(' ', '_')}.png")
                print("Available elements on page:")
                try:
                    all_clickable = self.driver.find_elements(By.XPATH, "//button | //a | //input[@type='submit']")
                    for elem in all_clickable[:10]:  # Show first 10
                        try:
                            print(f"  - {elem.tag_name}: '{elem.text}' (class: {elem.get_attribute('class')})")
                        except:
                            pass
                except:
                    pass
            
            # Close the sidebar/details panel
            self.close_sidebar_or_details()
            
        except Exception as e:
            print(f"Error processing event {event_name}: {str(e)}")
            # Take screenshot for debugging
            try:
                self.driver.save_screenshot(f"error_{event_name.replace(' ', '_')}.png")
            except:
                pass

    def close_sidebar_or_details(self):
        """Close sidebar or details panel"""
        try:
            # Try different methods to close the sidebar/details
            close_selectors = [
                (By.XPATH, "//button[contains(@class, 'close')]"),
                (By.XPATH, "//button[@aria-label='Close']"),
                (By.XPATH, "//button[contains(text(), '×')]"),
                (By.CSS_SELECTOR, ".close"),
                (By.CSS_SELECTOR, "[data-dismiss]"),
                (By.XPATH, "//button[contains(@class, 'btn-close')]"),
            ]
            
            for selector_type, selector_value in close_selectors:
                try:
                    close_button = self.driver.find_element(selector_type, selector_value)
                    if close_button.is_displayed():
                        close_button.click()
                        print("Closed sidebar/details using close button")
                        time.sleep(1)
                        return
                except NoSuchElementException:
                    continue
            
            # If no close button found, try clicking on the row again to collapse
            print("No close button found, trying alternative methods...")
            
            # Try pressing Escape key
            try:
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                print("Pressed Escape key to close")
                time.sleep(1)
            except:
                pass
            
        except Exception as e:
            print(f"Error closing sidebar: {str(e)}")

    def rename_downloaded_file(self, event_name, event_date):
        """Rename the most recently downloaded CSV file"""
        try:
            downloads_dir = os.getcwd()
            # Get the most recent CSV file
            csv_files = glob.glob(os.path.join(downloads_dir, "*.csv"))
            if csv_files:
                latest_file = max(csv_files, key=os.path.getctime)
                
                # Clean up the event name and date for filename
                clean_event_name = event_name.replace("/", "-").replace("\\", "-").replace(":", "-")
                clean_date = event_date.replace("/", "-").replace("\\", "-")
                
                # Create new filename
                new_filename = f"{clean_event_name} - {clean_date}.csv"
                new_filepath = os.path.join(downloads_dir, new_filename)
                
                # Rename the file
                os.rename(latest_file, new_filepath)
                print(f"Renamed file to: {new_filename}")
            else:
                print("No CSV files found to rename")
        except Exception as e:
            print(f"Error renaming file: {str(e)}")

    def download_csv_files(self, num_files=7):
        """Download CSV files for the specified number of events"""
        try:
            self.check_driver()
            print(f"Starting CSV download process for up to {num_files} events")
            # Get all event rows with the correct class (dt-hasChild dtr-expanded)
            event_rows = self.driver.find_elements(By.CSS_SELECTOR, "tr.dt-hasChild.dtr-expanded")
            if not event_rows:
                print("No event rows found with class 'dt-hasChild dtr-expanded'")
                return
            rows_to_process = event_rows[:num_files]
            for i, row in enumerate(rows_to_process, 1):
                try:
                    print(f"\n--- Processing event {i}/{len(rows_to_process)} ---")
                    event_name, event_date = self.extract_event_info(row)
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", row)
                    time.sleep(1)
                    try:
                        row.click()
                        print("Clicked event row to ensure sidebar is open")
                    except Exception as e:
                        print(f"Regular click failed: {e}, trying JavaScript click")
                        self.driver.execute_script("arguments[0].click();", row)
                        print("Clicked event row using JavaScript")
                    time.sleep(2)
                    download_button = None
                    download_button_selectors = [
                        (By.ID, "download-csv"),
                        (By.XPATH, "//button[contains(text(), 'Download CSV') or contains(@class, 'csv') or contains(@id, 'download-csv') or contains(@class, 'btn')]")
                    ]
                    for selector_type, selector_value in download_button_selectors:
                        try:
                            download_button = self.wait.until(EC.element_to_be_clickable((selector_type, selector_value)))
                            print(f"Found download CSV button using {selector_type}: {selector_value}")
                            break
                        except TimeoutException:
                            continue
                    if download_button:
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
                        time.sleep(1)
                        try:
                            download_button.click()
                            print("Clicked download CSV button")
                        except Exception as e:
                            print(f"Regular click failed: {e}, trying JavaScript click")
                            self.driver.execute_script("arguments[0].click();", download_button)
                            print("Clicked download CSV button using JavaScript")
                        time.sleep(3)
                        self.rename_downloaded_file(event_name, event_date)
                    else:
                        print("Could not find download CSV button for this event")
                        self.driver.save_screenshot(f"no_download_button_{event_name.replace(' ', '_')}.png")
                    self.close_sidebar_or_details()
                    time.sleep(1)
                except Exception as e:
                    print(f"Error processing row {i}: {str(e)}")
                    continue
            print(f"\nCompleted processing {len(rows_to_process)} events")
        except Exception as e:
            print(f"Error during CSV download process: {str(e)}")
            raise

    def wait_for_downloads(self):
        try:
            self.check_driver()
            print("Waiting for downloads to complete...")
            time.sleep(5)
            downloads_dir = os.getcwd()
            max_wait = 60  # Maximum wait time in seconds
            waited = 0
            
            while waited < max_wait:
                time.sleep(5)
                waited += 5
                
                # Check if there are any .crdownload files (Chrome incomplete downloads)
                crdownload_files = glob.glob(os.path.join(downloads_dir, "*.crdownload"))
                if not crdownload_files:
                    print("All downloads completed")
                    break
                else:
                    print(f"Waiting for downloads to finish... ({len(crdownload_files)} files remaining)")
            
            if waited >= max_wait:
                print("Download wait timeout reached")
                
        except Exception as e:
            print(f"Error while waiting for downloads: {str(e)}")

    def cleanup(self):
        if self.driver:
            self.driver.quit()
            print("Browser closed")

    def run(self, num_files=7):
        try:
            self.setup_driver()
            self.login()
            self.navigate_to_sold_listings()
            self.search_events()
            self.download_csv_files(num_files)
            self.wait_for_downloads()
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
        finally:
            self.cleanup()

def main():
    username = "ztareen@purdue.edu"
    password = "q3yS$$%h8yp51eXK"
    search_term = "seattle mariners"  # Now configurable
    
    scraper = SeatDataScraper(username, password, search_term)
    try:
        scraper.run(num_files=7)
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        input("Press Enter to close the browser...")
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main()