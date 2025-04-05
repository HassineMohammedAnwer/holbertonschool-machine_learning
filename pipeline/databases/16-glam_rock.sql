-- lists all bands with Glam rock as their main style, ranked by their longevity
-- Column names must be:
-- band_name
-- lifespan until 2020 (in years)
-- You should use attributes formed and split for computing the lifespan
-- select band name and calculate lifespan.
-- lifespan is calculated as difference between split year and formed year.
-- If band hasn't split (split is NULL), use 2020 as end year.
SELECT
    band_name,
    (COALESCE(split, 2020) - formed) AS lifespan
FROM
    metal_bands
WHERE
    style LIKE '%Glam rock%'
ORDER BY
    lifespan DESC;