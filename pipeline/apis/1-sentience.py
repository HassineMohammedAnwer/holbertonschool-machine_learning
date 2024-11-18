#!/usr/bin/env python3
"""1. Where I am?"""
import requests


def sentientPlanets():
    """returns the list of names of the home planets of all sentient species
    Prototype: def sentientPlanets():
    Don’t forget the pagination
    sentient type is either in the classification or designation attributes."""
    url = 'https://swapi-api.hbtn.io/api/species/'
    sp=[]
    home_pl=[]
    while url:
        res = requests.get(url).json()
        url = res.get('next')
        sp +=res.get('results')
    for species in sp:
            # Check if the species is sentient
            if species.get('designation') == 'sentient' or \
                species.get('classification') == 'sentient':
                homeworld_url = species.get("homeworld")
                if homeworld_url:
                    # Fetch the home planet's name
                    planet_response = requests.get(homeworld_url).json()
                    home_pl.append(planet_response["name"])


    return home_pl
