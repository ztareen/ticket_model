def ticketInfoGetter(event):
    event_id = event.get("id")
    title = event.get("title")
    date = event.get("datetime_local")
    url = event.get("url")

    venue = event.get("venue", {})
    venue_name = venue.get("name")
    city = venue.get("city")
    state = venue.get("state")

    promo = event.get("event_promotion") or {}
    promo_headline = promo.get("headline", "No promotion")

    stats = event.get("stats", {})
    lowest = stats.get("lowest_price")
    average = stats.get("average_price")
    highest = stats.get("highest_price")
    listings = stats.get("listing_count")
    event_count = stats.get("event_count")

    print(f"Lowest Price: {lowest}")
    print(f"Highest Price: {highest}")
    print(f"Average Price: {average}")
    print(f"Number of Listings: {listings}")

    teams = [p["name"] for p in event.get("performers", [])]

    print(f"Event ID: {event_id}")
    print(f"Title: {title}")
    print(f"Date: {date}")
    print(f"Teams: {' vs '.join(teams)}")
    print(f"Venue: {venue_name} ({city}, {state})")
    print(f"Promo: {promo_headline}")
    print(f"URL: {url}")
    print("-" * 60)

    return event_id