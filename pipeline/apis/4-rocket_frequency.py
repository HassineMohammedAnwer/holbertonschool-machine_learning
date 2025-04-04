#!/usr/bin/env python3
"""2. Rate me is you can!"""
import requests
from collections import Counter


if __name__ == '__main__':
    """displays the first launch with these information:
    Name of the launch
    The date (in local time)
    The rocket name
    The name (with the locality) of the launchpad
    <launch name> (<date>) <rocket name> - <launchpad name>
    __(<launchpad locality>)
    use the date_unix for sorting it - and if 2 launches have
    __the same date, use
    __the first one in the API result."""
    url = 'https://api.spacexdata.com/v4/launches'
    res = requests.get(url).json()
    if res is None:
        exit(99)
    rocket_launch_counts = Counter(launch["rocket"] for launch in res)
    rocket_url = 'https://api.spacexdata.com/v4/rockets/'
    roc_res = requests.get(rocket_url).json()
    rocket_id_to_name = {rocket["id"]: rocket["name"] for rocket in roc_res}
    rocket_launch_list = [
        (rocket_id_to_name[rocket_id], count)
        for rocket_id, count in rocket_launch_counts.items()
        if rocket_id in rocket_id_to_name
    ]
    sorted_rocket_launches = sorted(
        rocket_launch_list, key=lambda x: (-x[1], x[0])
    )
    for rocket_name, count in sorted_rocket_launches:
        print(f"{rocket_name}: {count}")
