def seatingInfoGetter(event, event_IDs):
    url_second = "https://api.seatgeek.com/2/events/section_info/:event_IDs"
    section = event.get("sections")

    print(f"Section: {section}")



    #what do i wanna get?
    # section
    # array for all the seat numbers within that section
    #real question is HOW tf do i relate this to the price, if the price is not
