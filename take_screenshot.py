#!/usr/bin/env python3
"""
Screenshot script for DreamWalk demo
Takes a screenshot of the web dashboard
"""

import time
import webbrowser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def take_screenshot():
    """Take a screenshot of the DreamWalk dashboard"""
    print("Taking screenshot of DreamWalk dashboard...")
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1200,800")
    
    try:
        # Initialize driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to dashboard
        driver.get("http://localhost:8000")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "connectionStatus"))
        )
        
        # Wait a bit for data to load
        time.sleep(3)
        
        # Take screenshot
        driver.save_screenshot("docs/screenshots/dashboard.png")
        print("Screenshot saved to docs/screenshots/dashboard.png")
        
        driver.quit()
        
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        print("Make sure Chrome is installed and the web demo is running")

if __name__ == "__main__":
    take_screenshot()
