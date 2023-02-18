#!/usr/bin/env python3
"""integrate.py"""


def poly_integral(poly, C=0):
    if not isinstance(poly, list) or len(poly) == 0 or poly is None:
        return None
    if len(poly) == 1:
        return [C]
    new_poly = [C]
    for i, coeff in enumerate(poly):
        if not isinstance(coeff, (int, float)):
            return None
        res = coeff / (i + 1)
        if res.is_integer():
            new_poly.append(int(res))
        else:
            new_poly.append(res)
    return new_poly
