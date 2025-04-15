#!/usr/bin/env python3
"""2. Rate me is you can!"""
import requests


if __name__ == '__main__':
    """displays the first launch with these information:
    Name of the launch
    The date (in local time)
    The rocket name
    The name (with the locality) of the launchpad
    <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
    use the date_unix for sorting it - and if 2 launches have the same date, use
    __the first one in the API result."""
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    res = requests.get(url).json()
    if res is None:
        exit(99)
    rockets = []
    for launch in res:
        rocket_url = 'https://api.spacexdata.com/v4/rockets/{}'.format(
            launch.get('rocket'))
        roc = requests.get(rocket_url).json()
        roc_name = roc.get('name')
        if rockets.get(roc_name) is None:
            rockets[roc_name] = 1
            continue
        rockets[roc_name] += 1
    o_rockets = sorted(rockets.items(),
                           key=lambda kv: kv[1],
                           reverse=True)
    for rocket, count in o_rockets:
        print("{}: {}".format(rocket, count))