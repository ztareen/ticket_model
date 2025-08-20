import time
import os
import glob
import fnmatch
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

#next thing is clicking page 2 and doing it all over again same w/ maybe page 3

class SeatDataScraper:
    def __init__(self, username="", password="", search_term="seattle mariners"):
        self.username = ""
        self.password = ""
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
            self.driver.execute_script("arguments[0].value = arguments[1];", email_field, "")
            time.sleep(0.5)
            email_field.clear()
            email_field.send_keys("")
            print(f"Entered email: [BLANK]")
            
            self.driver.execute_script("arguments[0].scrollIntoView(true);", password_field)
            time.sleep(1)
            self.driver.execute_script("arguments[0].value = '';", password_field)
            time.sleep(0.5)
            self.driver.execute_script("arguments[0].value = arguments[1];", password_field, "")
            time.sleep(0.5)
            password_field.clear()
            password_field.send_keys("")
            print("Entered password: [BLANK]")
            
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

            # Set entries per page to 50
            print("Setting entries per page to 50...")
            try:
                entries_dropdown = self.wait.until(EC.presence_of_element_located((
                    By.CSS_SELECTOR, "select#dt-length-1.dt-input"
                )))
                self.driver.execute_script("arguments[0].scrollIntoView(true);", entries_dropdown)
                time.sleep(1)
                entries_dropdown.click()
                time.sleep(0.5)
                # Select the option with value '50'
                for option in entries_dropdown.find_elements(By.TAG_NAME, "option"):
                    if option.get_attribute("value") == "50":
                        option.click()
                        print("Selected 50 entries per page.")
                        break
                time.sleep(3)  # Wait for table to reload
            except Exception as e:
                print(f"Could not set entries per page: {str(e)}")
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
            # Get event name from the correct column (skip first 2 td elements which are buttons)
            tds = row.find_elements(By.TAG_NAME, "td")
            event_name = None
            event_date = None
            
            # Skip first 2 td elements (pin button and other buttons)
            content_tds = tds[2:] if len(tds) > 2 else tds
            
            for td in content_tds:
                text = td.text.strip()
                # Event name is the first non-empty cell that doesn't look like a date
                if text and not event_name and not (len(text) >= 10 and text[4] == '-' and text[7] == '-'):
                    event_name = text
                # Event date is the first cell that looks like a date (YYYY-MM-DD or similar)
                if text and not event_date and len(text) >= 10 and text[4] == '-' and text[7] == '-':
                    event_date = text
                    
            if not event_name:
                event_name = "Unknown-Event"
            if not event_date:
                event_date = "Unknown-Date"
            return event_name, event_date
        except Exception as e:
            print(f"Error extracting event info: {str(e)}")
            return "Unknown-Event", "Unknown-Date"

    def click_event_row_robust(self, row):
        """Robustly click on an event row to open sidebar"""
        try:
            # Scroll the row into view
            self.driver.execute_script("arguments[0].scrollIntoView(true);", row)
            time.sleep(1)
            
            # Try clicking on different parts of the row (skip first 2 td elements)
            click_methods = [
                # Method 1: Click on the third td (skip first 2 button columns)
                lambda: row.find_element(By.XPATH, ".//td[3]").click(),
                # Method 2: Click on the fourth td 
                lambda: row.find_element(By.XPATH, ".//td[4]").click(),
                # Method 3: Click on the fifth td
                lambda: row.find_element(By.XPATH, ".//td[5]").click(),
                # Method 4: Click on the row itself
                lambda: row.click(),
                # Method 5: JavaScript click on third td
                lambda: self.driver.execute_script("arguments[0].click();", row.find_element(By.XPATH, ".//td[3]")),
                # Method 6: JavaScript click on fourth td
                lambda: self.driver.execute_script("arguments[0].click();", row.find_element(By.XPATH, ".//td[4]")),
                # Method 7: JavaScript click on the row
                lambda: self.driver.execute_script("arguments[0].click();", row),
            ]
            
            for i, click_method in enumerate(click_methods, 1):
                try:
                    print(f"Trying click method {i}...")
                    click_method()
                    time.sleep(2)  # Wait a bit for sidebar to appear
                    
                    # Check if sidebar appeared by looking for download button
                    try:
                        download_button = self.driver.find_element(By.ID, "download-csv")
                        if download_button.is_displayed():
                            print(f"✅ Click method {i} successful - sidebar opened!")
                            return True
                    except NoSuchElementException:
                        pass
                    
                    # Alternative check - look for any sidebar-like element
                    sidebar_selectors = [
                        (By.CSS_SELECTOR, ".sidebar"),
                        (By.CSS_SELECTOR, ".dtr-details"),
                        (By.CSS_SELECTOR, "[class*='detail']"),
                        (By.CSS_SELECTOR, "[class*='sidebar']"),
                    ]
                    
                    for selector_type, selector_value in sidebar_selectors:
                        try:
                            sidebar = self.driver.find_element(selector_type, selector_value)
                            if sidebar.is_displayed():
                                print(f"✅ Click method {i} successful - sidebar found!")
                                return True
                        except NoSuchElementException:
                            continue
                    
                    print(f"Click method {i} didn't open sidebar, trying next method...")
                    
                except Exception as e:
                    print(f"Click method {i} failed: {str(e)}")
                    continue
            
            print("❌ All click methods failed to open sidebar")
            return False
            
        except Exception as e:
            print(f"Error in robust click: {str(e)}")
            return False

    def download_csv_files(self, num_files=14):
        """Download CSV files for the specified number of events"""
        try:
            self.check_driver()
            print(f"Starting CSV download process for up to {num_files} events")
            
            # Get all event rows from the table
            event_rows = self.get_event_rows()
            
            if not event_rows:
                print("No event rows found in table")
                return
            
            print(f"Found {len(event_rows)} total event rows")
            rows_to_process = event_rows[:num_files]
            
            for i, row in enumerate(rows_to_process, 1):
                try:
                    print(f"\n--- Processing event {i}/{len(rows_to_process)} ---")
                    
                    # Extract event info first
                    event_name, event_date = self.extract_event_info(row)
                    print(f"Event: {event_name} - {event_date}")
                    
                    # Use robust clicking method
                    if not self.click_event_row_robust(row):
                        print(f"❌ Could not open sidebar for event {i}, skipping...")
                        continue
                    
                    # Now look for download button
                    download_button = None
                    download_button_selectors = [
                        (By.ID, "download-csv"),
                        (By.XPATH, "//button[contains(text(), 'Download CSV')]"),
                        (By.XPATH, "//button[@id='download-csv']"),
                        (By.CSS_SELECTOR, "button#download-csv"),
                        (By.XPATH, "//button[contains(@class, 'btn') and contains(text(), 'CSV')]"),
                        (By.XPATH, "//a[contains(text(), 'Download CSV')]"),
                        (By.XPATH, "//a[contains(@href, 'csv')]"),
                    ]
                    
                    for selector_type, selector_value in download_button_selectors:
                        try:
                            download_button = WebDriverWait(self.driver, 10).until(
                                EC.element_to_be_clickable((selector_type, selector_value))
                            )
                            print(f"Found download CSV button using {selector_type}: {selector_value}")
                            break
                        except TimeoutException:
                            continue
                    
                    if download_button:
                        try:
                            # Scroll download button into view
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
                            time.sleep(1)
                            
                            # Try clicking the download button
                            try:
                                download_button.click()
                                print("✅ Clicked download CSV button")
                            except Exception as e:
                                print(f"Regular click failed: {e}, trying JavaScript click")
                                self.driver.execute_script("arguments[0].click();", download_button)
                                print("✅ Clicked download CSV button using JavaScript")
                            
                            time.sleep(3)  # Wait for download to start
                            
                            # Rename the downloaded file
                            self.rename_downloaded_file(event_name, event_date)
                            
                        except Exception as e:
                            print(f"❌ Error clicking download button: {str(e)}")
                    else:
                        print("❌ Could not find download CSV button")
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
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Error processing row {i}: {str(e)}")
                    continue
            
            print(f"\n✅ Completed processing {len(rows_to_process)} events")
            
        except Exception as e:
            print(f"❌ Error during CSV download process: {str(e)}")
            raise

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

    def reorder_event_name_for_mariners(self, event_name):
        """Reorder event name to ensure Seattle Mariners appears first"""
        try:
            # Convert to lowercase for matching
            event_lower = event_name.lower()
            
            # Common patterns for Seattle Mariners
            mariners_variations = [
                'seattle mariners',
                'mariners',
                'sea mariners'
            ]
            
            # Check if this is a Mariners game
            has_mariners = any(variation in event_lower for variation in mariners_variations)
            
            if not has_mariners:
                # If no Mariners reference found, return original name
                return event_name
            
            # Clean the event name
            clean_event = event_name.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "-")
            
            # Split by common separators to find team names
            separators = [' at ', ' vs ', ' v ', '@', '-at-', '-vs-', '-v-']
            parts = [clean_event]
            
            # Try to split by separators
            for sep in separators:
                for i, part in enumerate(parts):
                    if sep in part.lower():
                        split_parts = part.split(sep, 1)
                        parts = parts[:i] + split_parts + parts[i+1:]
                        break
                if len(parts) > 1:
                    break
            
            if len(parts) >= 2:
                team1 = parts[0].strip()
                team2 = parts[1].strip()
                
                # Check which team is the Mariners
                team1_lower = team1.lower()
                team2_lower = team2.lower()
                
                mariners_is_team1 = any(variation in team1_lower for variation in mariners_variations)
                mariners_is_team2 = any(variation in team2_lower for variation in mariners_variations)
                
                if mariners_is_team2 and not mariners_is_team1:
                    # Swap teams so Mariners come first
                    print(f"Reordering teams: '{team1}' and '{team2}' -> 'Seattle-Mariners' first")
                    return f"Seattle-Mariners-vs-{team1}"
                elif mariners_is_team1:
                    # Mariners already first, but ensure consistent naming
                    return f"Seattle-Mariners-vs-{team2}"
            
            # If we can't parse teams properly, ensure Seattle-Mariners is at the start
            if not clean_event.lower().startswith('seattle-mariners'):
                # Try to extract the opponent team name
                # Remove common Mariners references to isolate opponent
                opponent_name = clean_event
                for variation in ['seattle-mariners', 'mariners', 'sea-mariners']:
                    # Remove the variation and common connectors
                    opponent_name = re.sub(rf'{variation}[-\s]*(?:at|vs|v|@)[-\s]*', '', opponent_name, flags=re.IGNORECASE)
                    opponent_name = re.sub(rf'[-\s]*(?:at|vs|v|@)[-\s]*{variation}', '', opponent_name, flags=re.IGNORECASE)
                    opponent_name = re.sub(rf'{variation}', '', opponent_name, flags=re.IGNORECASE)
                
                # Clean up any leftover separators
                opponent_name = re.sub(r'^[-\s]+|[-\s]+$', '', opponent_name)
                opponent_name = re.sub(r'[-\s]+', '-', opponent_name)
                
                if opponent_name and opponent_name != clean_event:
                    return f"Seattle-Mariners-vs-{opponent_name}"
                else:
                    return f"Seattle-Mariners-{clean_event}"
            
            return clean_event
            
        except Exception as e:
            print(f"Error reordering event name: {str(e)}")
            # Fallback: just ensure Seattle-Mariners is first
            clean_event = event_name.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "-")
            if not clean_event.lower().startswith('seattle-mariners'):
                return f"Seattle-Mariners-{clean_event}"
            return clean_event

    def rename_downloaded_file(self, event_name, event_date):
        """Rename the most recently downloaded CSV file and move it to the target folder"""
        try:
            # Set the target directory
            target_dir = r"C:\Users\zarak\Downloads\TestData_Mariners"
            os.makedirs(target_dir, exist_ok=True)
            # Get the most recent CSV file in the current directory
            downloads_dir = os.getcwd()
            csv_files = glob.glob(os.path.join(downloads_dir, "*.csv"))
            if csv_files:
                latest_file = max(csv_files, key=os.path.getctime)
                
                # Reorder event name to ensure Seattle Mariners comes first
                reordered_event_name = self.reorder_event_name_for_mariners(event_name)
                
                # Remove day of week in parenthesis from event_date, e.g. '2025-06-24 (Tue)' -> '2025-06-24'
                clean_date = event_date.split(' (')[0].replace("/", "-").replace("\\", "-").replace(" ", "-")
                
                # Create new filename (no extension in name, add .csv)
                new_filename = f"{reordered_event_name}-{clean_date}.csv"
                new_filepath = os.path.join(target_dir, new_filename)
                
                # Move and rename the file
                os.rename(latest_file, new_filepath)
                print(f"Moved and renamed file to: {new_filepath}")
            else:
                print("No CSV files found to rename/move")
        except Exception as e:
            print(f"Error renaming/moving file: {str(e)}")

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

    def run(self, num_files=14):
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
    password = "q3yS$$%h8yp51eXK"  # Note: Consider using environment variables for credentials
    search_term = "seattle mariners"  # Now configurable
    
    scraper = SeatDataScraper(username, password, search_term)
    try:
        scraper.run(num_files=50)
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        input("Press Enter to close the browser...")
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main()