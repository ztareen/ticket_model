from bs4 import BeautifulSoup
import requests

# https://www.ticketmaster.com/ken-carson-tickets/artist/2804916

url = 'https://www.ticketmaster.com/ken-carson-the-lord-of-chaos-boston-massachusetts-07-29-2025/event/01006298C1E14C04'

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/5  7.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

page = requests.get(url, headers=headers)
soup = BeautifulSoup(page.text, 'html.parser')
print(soup.prettify())
soup = BeautifulSoup(page.text, 'html.parser')

print(soup)


#have two types of scraping
# one takes in ticketmaster for a full artist (Ex: https://www.ticketmaster.com/ken-carson-tickets/artist/2804916)
# and gives back a data table that is
# Location , Date, URL
# Ex: London, May 15th, www.kencarson.com/londonticket

# the next one takes in the specific URL given
# EX: https://www.ticketmaster.com/ken-carson-the-lord-of-chaos-boston-massachusetts-07-29-2025/event/01006298C1E14C04
# and returns back a data table of each ticket price as well as the seciton and row and ticket type
# Ex: Section 310, Row C, Standard Ticket, $130
# days until event?
# create a gradient of times to buy? Blue is perfect day, green is ideal, yellow is good, orange is meh, red is bad?
# Popularity -> Demand
# Stadium type????
# How popular the artist is/ how many monthly listeners they have (scrape from spotify)
# NY ticket is way more expensive than west lafayette
# Demand???? Number of ppl visiting the site.
# they use dynamic pricing
# reverse engineer their model???
# the model is the hardest part
# input ticketmaster of ur event, input seatgeek of ur concert, insert row and seat type u want and the
# model gives it back
# once we get that dataset my model can categorize by cheapest to most expensive and make categories for stuff
