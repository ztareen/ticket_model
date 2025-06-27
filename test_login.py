import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

def test_login():
    """Simple test to debug login functionality"""
    username = "ztareen@purdue.edu"
    password = "q3yS$$%h8yp51eXK"
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Initialize driver
    driver = None
    try:
        print("Setting up Chrome WebDriver...")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        wait = WebDriverWait(driver, 20)
        
        print("Navigating to login page...")
        driver.get("https://seatdata.io/login/")
        
        # Wait for page to load
        time.sleep(5)
        
        print("Current URL:", driver.current_url)
        print("Page title:", driver.title)
        
        # Take a screenshot to see what we're working with
        driver.save_screenshot("login_page.png")
        print("Screenshot saved as login_page.png")
        
        # Try to find email field with multiple strategies
        email_field = None
        email_selectors = [
            (By.NAME, "email"),
            (By.ID, "email"),
            (By.XPATH, "//input[@type='email']"),
            (By.XPATH, "//input[contains(@placeholder, 'email') or contains(@placeholder, 'Email')]"),
            (By.CSS_SELECTOR, "input[type='email']"),
            (By.CSS_SELECTOR, "input[name*='email']"),
            (By.CSS_SELECTOR, "input[id*='email']"),
            (By.XPATH, "//input[@type='text']"),  # Sometimes email fields are just text inputs
        ]
        
        print("Looking for email field...")
        for selector_type, selector_value in email_selectors:
            try:
                email_field = wait.until(EC.presence_of_element_located((selector_type, selector_value)))
                print(f"✓ Found email field using {selector_type}: {selector_value}")
                break
            except TimeoutException:
                print(f"✗ Not found with {selector_type}: {selector_value}")
                continue
        
        if not email_field:
            print("❌ Could not find email field!")
            # List all input elements on the page
            all_inputs = driver.find_elements(By.TAG_NAME, "input")
            print(f"Found {len(all_inputs)} input elements:")
            for i, inp in enumerate(all_inputs):
                try:
                    print(f"  {i+1}. type={inp.get_attribute('type')}, name={inp.get_attribute('name')}, id={inp.get_attribute('id')}, placeholder={inp.get_attribute('placeholder')}")
                except:
                    print(f"  {i+1}. Could not get attributes")
            return
        
        # Try to find password field
        password_field = None
        password_selectors = [
            (By.NAME, "password"),
            (By.ID, "password"),
            (By.XPATH, "//input[@type='password']"),
            (By.XPATH, "//input[contains(@placeholder, 'password') or contains(@placeholder, 'Password')]"),
            (By.CSS_SELECTOR, "input[type='password']"),
            (By.CSS_SELECTOR, "input[name*='password']"),
            (By.CSS_SELECTOR, "input[id*='password']")
        ]
        
        print("Looking for password field...")
        for selector_type, selector_value in password_selectors:
            try:
                password_field = driver.find_element(selector_type, selector_value)
                print(f"✓ Found password field using {selector_type}: {selector_value}")
                break
            except NoSuchElementException:
                print(f"✗ Not found with {selector_type}: {selector_value}")
                continue
        
        if not password_field:
            print("❌ Could not find password field!")
            return
        
        # Enter credentials
        print("Entering credentials...")
        
        # Clear and enter email
        driver.execute_script("arguments[0].scrollIntoView(true);", email_field)
        time.sleep(1)
        email_field.clear()
        time.sleep(0.5)
        email_field.send_keys(username)
        time.sleep(0.5)
        print(f"✓ Entered email: {username}")
        
        # Clear and enter password
        driver.execute_script("arguments[0].scrollIntoView(true);", password_field)
        time.sleep(1)
        password_field.clear()
        time.sleep(0.5)
        password_field.send_keys(password)
        time.sleep(0.5)
        print("✓ Entered password")
        
        # Take screenshot after entering credentials
        driver.save_screenshot("credentials_entered.png")
        print("Screenshot saved as credentials_entered.png")
        
        # Look for login button
        login_button = None
        button_selectors = [
            (By.XPATH, "//button[@type='submit']"),
            (By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Sign In') or contains(text(), 'Log In')]"),
            (By.XPATH, "//input[@type='submit']"),
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.CSS_SELECTOR, "input[type='submit']"),
            (By.CSS_SELECTOR, ".login-button, .signin-button, .btn-primary"),
            (By.XPATH, "//*[contains(@class, 'btn') and contains(@class, 'primary')]")
        ]
        
        print("Looking for login button...")
        for selector_type, selector_value in button_selectors:
            try:
                login_button = driver.find_element(selector_type, selector_value)
                print(f"✓ Found login button using {selector_type}: {selector_value}")
                print(f"  Button text: {login_button.text}")
                break
            except NoSuchElementException:
                print(f"✗ Not found with {selector_type}: {selector_value}")
                continue
        
        if not login_button:
            print("❌ Could not find login button!")
            # List all buttons on the page
            all_buttons = driver.find_elements(By.TAG_NAME, "button")
            print(f"Found {len(all_buttons)} buttons:")
            for i, btn in enumerate(all_buttons):
                try:
                    print(f"  {i+1}. text='{btn.text}', type={btn.get_attribute('type')}, class={btn.get_attribute('class')}")
                except:
                    print(f"  {i+1}. Could not get attributes")
            return
        
        # Click login button
        print("Clicking login button...")
        driver.execute_script("arguments[0].scrollIntoView(true);", login_button)
        time.sleep(1)
        login_button.click()
        print("✓ Clicked login button")
        
        # Wait and check result
        time.sleep(5)
        print(f"Current URL after login: {driver.current_url}")
        print(f"Page title after login: {driver.title}")
        
        # Take final screenshot
        driver.save_screenshot("after_login.png")
        print("Screenshot saved as after_login.png")
        
        if "dashboard" in driver.current_url:
            print("✅ Login successful!")
        else:
            print("❌ Login may have failed - not redirected to dashboard")
            # Check for error messages
            error_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'error') or contains(text(), 'Error') or contains(text(), 'invalid') or contains(text(), 'Invalid')]")
            if error_elements:
                print(f"Error message found: {error_elements[0].text}")
        
        input("Press Enter to close browser...")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        if driver:
            driver.save_screenshot("error_screenshot.png")
            print("Error screenshot saved as error_screenshot.png")
        input("Press Enter to close browser...")
    finally:
        if driver:
            driver.quit()
            print("Browser closed")

if __name__ == "__main__":
    test_login() 