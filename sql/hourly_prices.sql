-- Hourly day-ahead prices per bidding zone, in local (Swedish) delivery time.
-- Since 1 October 2025 the market clears in 15-minute steps; those are averaged to the hour
-- so the whole history has one resolution. On the autumn DST day the repeated 02:00 hour
-- is averaged too, and the missing spring hour is filled in Python.
SELECT
    area,
    CAST(timezone('Europe/Stockholm', CAST(time_start AS TIMESTAMPTZ)) AS DATE) AS date,
    hour(timezone('Europe/Stockholm', CAST(time_start AS TIMESTAMPTZ)))        AS hour,
    avg(EUR_per_kWh) * 1000                                                    AS eur_mwh,
    avg(SEK_per_kWh) * 100                                                     AS ore_kwh,
    count(*)                                                                   AS n_intervals
FROM read_parquet($prices)
GROUP BY ALL
ORDER BY area, date, hour
