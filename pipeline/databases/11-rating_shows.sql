-- lists all shows from hbtn_0d_tvshows_rate by their rating.
-- Each record should display: tv_shows.title - rating sum
-- Results must be sorted in descending order by the rating
-- You can use only one SELECT statement
-- The database name will be passed as an argument of the mysql command
SELECT tv_shows.title , SUM(tv_show_genres.rate)
FROM tv_show
LEFT JOIN tv_show_ratings
ON tv_shows.id = tv_show_genres.show_id
GROUP BY tv_shows.title
ORDER BY SUM(tv_show_genres.rate) DESC
