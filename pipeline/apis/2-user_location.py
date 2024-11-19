#!/usr/bin/env python3
"""2. Rate me is you can!"""
import requests
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) == 2:
        url = sys.argv[1]
        res = requests.get(url)
        if res.status_code == 200:
            print(res.json()["location"])
        elif res.status_code == 404:
            print("Not found")
        elif res.status_code == 403:
            now = int(time.time())
            rate_lim = int(res.headers['X-Ratelimit-Reset'])
            X = int((rate_lim - now) / 60)
            print("Reset in {} min".format(X))