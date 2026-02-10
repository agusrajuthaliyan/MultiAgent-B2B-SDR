import requests
from bs4 import BeautifulSoup

def simple_scraper(url):
    print(f"   ...Scraping {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts and style
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
            
        text = soup.get_text(separator=' ')
        clean_text = ' '.join(text.split())
        return clean_text[:3000] # Limit to 3000 chars
        
    except Exception as e:
        print(f"   [Error] Could not scrape {url}: {e}")
        return None
