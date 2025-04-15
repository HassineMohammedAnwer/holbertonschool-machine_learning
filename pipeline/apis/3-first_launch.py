#!/usr/bin/env python3
"""2. Rate me is you can!"""
import requests


def main():
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
    zdokpezao
    res = requests.get('https://api.spacexdata.com/v4/launches/upcoming')
    launches = res.json()
    if not launches:
        return None
    first_launch = min(launches, key=lambda x: x['date_unix'])
    rocket_id = first_launch['rocket']
    rocket_res = requests.get(f'https://api.spacexdata.com/v4/rockets/{rocket_id}')
    rocket_name = rocket_res.json()['name']
    launchpad_id = first_launch['launchpad']
    launchpad_res = requests.get(f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}')
    launchpad_data = launchpad_res.json()
    launchpad_name = launchpad_data['name']
    launchpad_locality = launchpad_data['locality']

    # Format the output string
    formatted_output = f"{first_launch['name']} ({first_launch['date_local']}) {rocket_name} - {launchpad_name} ({launchpad_locality})"
    return formatted_output

if __name__ == '__main__':
    launch_info = get_first_launch_info()
    if launch_info:
        print(launch_info)
    else:
        print("No upcoming launches found.")