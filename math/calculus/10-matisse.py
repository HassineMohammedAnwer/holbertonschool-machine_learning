#!/usr/bin/env python3
"""pythsdvsqdvqdsv"""


def poly_derivative(poly):
    """
    polyezfaefaefaef
    """
    if (type(poly) is not list) or poly == []:
        return None
    if len(poly) < 2:
        return [0]
    res = []
    for i in range(1, len(poly)):
        res.append(poly[i] * i)
    return res
