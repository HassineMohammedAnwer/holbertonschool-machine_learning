#!/usr/bin/env python3
"""2. Rate me is you can!"""
import requests
import sys


if __name__ == '__main__':
    if len(sys.argv) == 2:
        url = sys.argv[1]
        res = requests.get(url, headers='Accept: application/vnd.github+json')
        if res.status_code == 200:
            print(res.json()["location"])
        elif res.status_code == 404:
            print("Not found")