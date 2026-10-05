-- Hourly weather in local time: Stockholm air temperature and a wind index,
-- the mean wind speed over six stations near Sweden's main wind power areas.
WITH local AS (
    SELECT
        variable,
        station,
        CAST(timezone('Europe/Stockholm', time_utc) AS DATE) AS date,
        hour(timezone('Europe/Stockholm', time_utc))        AS hour,
        value
    FROM read_parquet($weather)
    WHERE value IS NOT NULL
),
per_station AS (
    SELECT variable, station, date, hour, avg(value) AS value
    FROM local
    GROUP BY ALL
)
SELECT
    date,
    hour,
    avg(value) FILTER (WHERE variable = 'temperature') AS temp_sthlm,
    avg(value) FILTER (WHERE variable = 'wind_speed')  AS wind_index,
    count(*)   FILTER (WHERE variable = 'wind_speed')  AS n_wind_stations
FROM per_station
GROUP BY ALL
ORDER BY date, hour
