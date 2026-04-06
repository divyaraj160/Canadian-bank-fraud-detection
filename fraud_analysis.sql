-- ============================================================
-- Canadian Bank Transaction Fraud Detection
-- SQL Analysis Queries
-- Author: Divyaraj Jadeja | github.com/divyaraj160
-- ============================================================
-- These queries are designed to run against the transactions
-- table generated from transactions.csv (load into any SQL DB)
-- ============================================================


-- ── TABLE STRUCTURE ──────────────────────────────────────────
-- CREATE TABLE transactions (
--     transaction_id       VARCHAR(10) PRIMARY KEY,
--     transaction_date     DATE,
--     transaction_amount   DECIMAL(10,2),
--     hour_of_day          INT,
--     country              VARCHAR(50),
--     merchant_category    VARCHAR(50),
--     days_since_last_txn  INT,
--     txn_velocity_24h     INT,
--     distance_from_home_km DECIMAL(8,1),
--     new_merchant         TINYINT,
--     card_present         TINYINT,
--     city                 VARCHAR(50),
--     bank                 VARCHAR(50),
--     province             VARCHAR(5),
--     is_fraud             TINYINT
-- );


-- ── Q1: OVERALL FRAUD RATE ───────────────────────────────────
SELECT
    COUNT(*)                                    AS total_transactions,
    SUM(is_fraud)                               AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2)              AS fraud_rate_pct,
    ROUND(AVG(CASE WHEN is_fraud = 0 THEN transaction_amount END), 2) AS avg_legit_amount,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN transaction_amount END), 2) AS avg_fraud_amount
FROM transactions;


-- ── Q2: FRAUD RATE BY MERCHANT CATEGORY ─────────────────────
SELECT
    merchant_category,
    COUNT(*)                               AS total_txns,
    SUM(is_fraud)                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(transaction_amount), 2)      AS avg_amount
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;


-- ── Q3: FRAUD BY HOUR OF DAY (TIME-BASED PATTERNS) ──────────
SELECT
    hour_of_day,
    COUNT(*)                               AS total_txns,
    SUM(is_fraud)                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    CASE
        WHEN hour_of_day BETWEEN 0 AND 4   THEN 'Late Night'
        WHEN hour_of_day BETWEEN 5 AND 11  THEN 'Morning'
        WHEN hour_of_day BETWEEN 12 AND 17 THEN 'Afternoon'
        WHEN hour_of_day BETWEEN 18 AND 21 THEN 'Evening'
        ELSE 'Night'
    END AS time_of_day
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC;


-- ── Q4: HIGH-RISK COMBINATION — LATE NIGHT + LARGE AMOUNT ───
SELECT
    COUNT(*)                                AS flagged_transactions,
    SUM(is_fraud)                           AS confirmed_fraud,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(transaction_amount), 2)       AS avg_amount
FROM transactions
WHERE hour_of_day IN (0, 1, 2, 3, 22, 23)
  AND transaction_amount > 500;


-- ── Q5: FOREIGN TRANSACTION RISK ────────────────────────────
SELECT
    country,
    COUNT(*)                               AS total_txns,
    SUM(is_fraud)                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(transaction_amount), 2)      AS avg_amount
FROM transactions
GROUP BY country
ORDER BY fraud_rate_pct DESC;


-- ── Q6: HIGH-VELOCITY TRANSACTION ANALYSIS ──────────────────
-- Flag accounts making 5+ transactions in 24 hours
SELECT
    txn_velocity_24h,
    COUNT(*)                               AS total_txns,
    SUM(is_fraud)                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY txn_velocity_24h
ORDER BY txn_velocity_24h DESC;


-- ── Q7: CARD NOT PRESENT + FOREIGN = VERY HIGH RISK ─────────
SELECT
    card_present,
    country = 'Canada'                     AS is_domestic,
    COUNT(*)                               AS total_txns,
    SUM(is_fraud)                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY card_present, is_domestic
ORDER BY fraud_rate_pct DESC;


-- ── Q8: BANK-LEVEL FRAUD SUMMARY ────────────────────────────
SELECT
    bank,
    COUNT(*)                               AS total_txns,
    SUM(is_fraud)                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN transaction_amount ELSE 0 END), 2) AS fraud_loss_cad
FROM transactions
GROUP BY bank
ORDER BY fraud_loss_cad DESC;


-- ── Q9: PROVINCE-LEVEL RISK HEATMAP ─────────────────────────
SELECT
    province,
    COUNT(*)                               AS total_txns,
    SUM(is_fraud)                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(transaction_amount), 2)      AS avg_amount
FROM transactions
GROUP BY province
ORDER BY fraud_rate_pct DESC;


-- ── Q10: COMPOSITE RISK SCORE SEGMENTATION ──────────────────
-- Segments transactions into risk tiers using multiple signals
SELECT
    risk_tier,
    COUNT(*)                               AS txn_count,
    SUM(is_fraud)                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM (
    SELECT
        is_fraud,
        CASE
            WHEN (CASE WHEN hour_of_day IN (0,1,2,3,22,23) THEN 1 ELSE 0 END
                + CASE WHEN country != 'Canada' THEN 1 ELSE 0 END
                + CASE WHEN transaction_amount > 500 THEN 1 ELSE 0 END
                + CASE WHEN txn_velocity_24h >= 5 THEN 1 ELSE 0 END
                + new_merchant) >= 4 THEN 'CRITICAL'
            WHEN (CASE WHEN hour_of_day IN (0,1,2,3,22,23) THEN 1 ELSE 0 END
                + CASE WHEN country != 'Canada' THEN 1 ELSE 0 END
                + CASE WHEN transaction_amount > 500 THEN 1 ELSE 0 END
                + CASE WHEN txn_velocity_24h >= 5 THEN 1 ELSE 0 END
                + new_merchant) = 3 THEN 'HIGH'
            WHEN (CASE WHEN hour_of_day IN (0,1,2,3,22,23) THEN 1 ELSE 0 END
                + CASE WHEN country != 'Canada' THEN 1 ELSE 0 END
                + CASE WHEN transaction_amount > 500 THEN 1 ELSE 0 END
                + CASE WHEN txn_velocity_24h >= 5 THEN 1 ELSE 0 END
                + new_merchant) = 2 THEN 'MEDIUM'
            ELSE 'LOW'
        END AS risk_tier
    FROM transactions
) risk_segments
GROUP BY risk_tier
ORDER BY FIELD(risk_tier, 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW');


-- ── Q11: ROLLING 7-DAY FRAUD TREND ──────────────────────────
SELECT
    transaction_date,
    COUNT(*)        AS daily_txns,
    SUM(is_fraud)   AS daily_fraud,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS daily_fraud_rate,
    ROUND(AVG(SUM(is_fraud)) OVER (
        ORDER BY transaction_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2)           AS rolling_7day_avg
FROM transactions
GROUP BY transaction_date
ORDER BY transaction_date;


-- ── Q12: NEW MERCHANT RISK BY CATEGORY ──────────────────────
SELECT
    merchant_category,
    new_merchant,
    COUNT(*)        AS txn_count,
    SUM(is_fraud)   AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY merchant_category, new_merchant
ORDER BY merchant_category, new_merchant;
