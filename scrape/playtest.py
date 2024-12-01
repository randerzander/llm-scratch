from playwright.sync_api import sync_playwright

def scrape_webpage(url):
    with sync_playwright() as p:
        # Launch Firefox browser
        browser = p.firefox.launch(headless=True)
        
        # Create a new page
        page = browser.new_page()
        
        # Navigate to the URL
        page.goto(url)
        
        # Extract the page title
        title = page.title()
        
        # Extract all text from the body
        body_text = page.inner_text('body')
        
        # Close the browser
        browser.close()
        
        return title, body_text

# Example usage
url = "https://www.notebookcheck.net/"
title, content = scrape_webpage(url)

print(f"Title: {title}")
print(f"Content: {content[:200]}...")  # Print first 200 characters
