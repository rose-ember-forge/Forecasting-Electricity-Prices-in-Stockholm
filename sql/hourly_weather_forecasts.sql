-- Archived weather forecasts in local time, as they stood before the day-ahead auction.
-- Bids for day T close at 12:00 on day T-1. The forecast issued 24 h before an hour was
-- already out by then for hours 00-11, but not for hours 12-23, which get the 48 h old one.
WITH local AS (
    SELECT
        variable,
        station,
        CAST(timezone('Europe/Stockholm', time_utc) AS DATE) AS date,
        hour(timezone('Europe/Stockholm', time_utc))        AS hour,
        CASE WHEN hour(timezone('Europe/Stockholm', time_utc)) < 12 THEN lead_1d ELSE lead_2d END AS value
    FROM read_parquet($forecasts)
),
per_station AS (
    SELECT variable, station, date, hour, avg(value) AS value
    FROM local
    WHERE value IS NOT NULL
    GROUP BY ALL
)
SELECT
    date,
    hour,
    avg(value) FILTER (WHERE variable = 'temperature') AS temp_sthlm_fc,
    avg(value) FILTER (WHERE variable = 'wind_speed')  AS wind_index_fc
FROM per_station
GROUP BY ALL
ORDER BY date, hour
