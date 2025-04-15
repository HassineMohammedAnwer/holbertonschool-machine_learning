#!/usr/bin/env python3
"""2. Rate me is you can!"""
import requests


def main():
    """displays the first launch with these information:
    Name of the launch
    The date (in local time)
    The rocket name
    The name (with the locality) of the launchpad
    <launch name> (<date>) <rocket name> - <launchpad name>
    __(<launchpad locality>)
    use the date_unix for sorting it - and if 2 launches
    __have the same date, use
    __the first one in the API result."""
    res = requests.get('https://api.spacexdata.com/v4/launches/upcoming')
    launches = res.json()
    if not launches:
        return None
    first_launch = min(launches, key=lambda x: x['date_unix'])
    rocket_id = first_launch['rocket']
    rocket_res = requests.get(
        f'https://api.spacexdata.com/v4/rockets/{rocket_id}')
    rocket_name = rocket_res.json()['name']
    launchpad_id = first_launch['launchpad']
    launchpad_res = requests.get(
        f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}')
    launchpad_data = launchpad_res.json()
    launchpad_name = launchpad_data['name']
    launchpad_locality = launchpad_data['locality']
    # Format the output string
    formatted_output = f"{first_launch['name']} ({first_launch['date_local']}) {rocket_name} - {launchpad_name} ({launchpad_locality})"
    print(formatted_output)

if __name__ == '__main__':
    main()
