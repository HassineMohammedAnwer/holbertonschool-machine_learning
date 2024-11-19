#!/usr/bin/env python3
"""2. Rate me is you can!"""
import requests
import sys
import time
import os
import stat


def get_file_permissions(filename):
    # Get the file's mode
    mode = os.stat(filename).st_mode
    
    # Translate the mode into readable permissions
    permissions = [
        "r" if mode & stat.S_IRUSR else "-",
        "w" if mode & stat.S_IWUSR else "-",
        "x" if mode & stat.S_IXUSR else "-",
        "r" if mode & stat.S_IRGRP else "-",
        "w" if mode & stat.S_IWGRP else "-",
        "x" if mode & stat.S_IXGRP else "-",
        "r" if mode & stat.S_IROTH else "-",
        "w" if mode & stat.S_IWOTH else "-",
        "x" if mode & stat.S_IXOTH else "-"
    ]
    
    return "".join(permissions)

if __name__ == '__main__':
    current_file = __file__  # Get the current file's name
    permissions = get_file_permissions(current_file)
    print(f"Permissions for {current_file}: {permissions}")
    if len(sys.argv) == 2:
        url = sys.argv[1]
        res = requests.get(url,
                           headers={"Authorization": "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"})
        if res.status_code == 200:
            print(res.json()["location"])
        elif res.status_code == 404:
            print("Not found")
        elif res.status_code == 403:
            now = int(time.time())
            rate_lim = int(res.headers['X-Ratelimit-Reset'])
            X = int((rate_lim - now) / 60)
            print("Reset in {} min".format(X))
