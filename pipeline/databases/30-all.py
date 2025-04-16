#!/usr/bin/env python3
"""30. List all documents in Python"""


def list_all(mongo_collection):
    """
    lists all documents in a collection:
    Prototype: def list_all(mongo_collection):
    Return an empty list if no document in the collection
    mongo_collection will be the pymongo collection object
    """
    return list(mongo_collection.find())
