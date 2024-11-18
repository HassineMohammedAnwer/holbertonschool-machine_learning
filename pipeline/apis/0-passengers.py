#!/usr/bin/env python3
"""0. Can I join?"""
import requests


def availableShips(passengerCount):
    """returns the list of ships that can hold a given number of passengers:
    Prototype: def availableShips(passengerCount):
    Don’t forget the pagination
    If no ship available, return an empty list."""
    url = 'https://swapi-api.hbtn.io/api/starships/'
    req = requests.get(url).json()
    starships = []
    while req.get("next"):
        starship = req.get("results")
        for ship in starship:
            passenger = ship.get("passengers")
            if passenger == "n/a" or passenger == "unknown":
                continue
            if int(passenger.replace(",", "")) >= passengerCount:
                starships.append(ship.get("name"))
        req = requests.get(req.get("next")).json()
    return starships
