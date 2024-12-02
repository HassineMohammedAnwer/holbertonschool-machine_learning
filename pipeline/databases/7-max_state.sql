-- Write a script displays the max temperature of each state (ordered by State name).
SELECT state, avg(value) as max_temp
FROM temperatures
GROUP BY state
ORDER BY state ASC