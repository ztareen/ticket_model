from bs4 import BeautifulSoup
import requests

# https://www.ticketmaster.com/ken-carson-tickets/artist/2804916

url = 'https://www.ticketmaster.com/ken-carson-tickets/artist/2804916'

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/5  7.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

page = requests.get(url, headers=headers)
soup = BeautifulSoup(page.text, 'html.parser')
print(soup.prettify())
soup = BeautifulSoup(page.text, 'html.parser')

print(soup)