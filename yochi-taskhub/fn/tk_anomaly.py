# -*- coding: utf-8 -*-
"""Universal anomaly layer: every venue x metric, latest trading day vs the same
weekday over the trailing 8 weeks (its OWN baseline). Extreme z-scores surface
as cockpit signals - no rule authoring, no tasks, no email. The hand-written
watch rules stay for the specific, actionable cases."""
import datetime as dt
import logging

import tk_db
import lake_reader

LOG = logging.getLogger("tk_anomaly")

Z_THRESHOLD = 3.0
MAX_SIGNALS = 3
SOURCE = "anomaly layer"


def run():
    lake_reader.sync(log=LOG.info)
    res = lake_reader.query("""
        WITH d AS (SELECT venue, CAST("date" AS DATE) dd, net_sales, txns, atv
                   FROM mart_venue_daily),
        mx AS (SELECT max(dd) m FROM d),
        cur AS (SELECT * FROM d WHERE dd = (SELECT m FROM mx)),
        hist AS (SELECT venue,
                        avg(net_sales) mu_s, stddev_samp(net_sales) sd_s,
                        avg(txns) mu_t, stddev_samp(txns) sd_t,
                        avg(atv) mu_a, stddev_samp(atv) sd_a, count(*) n
                 FROM d
                 WHERE dayofweek(dd) = (SELECT dayofweek(m) FROM mx)
                   AND dd < (SELECT m FROM mx) AND dd >= (SELECT m FROM mx) - INTERVAL 56 DAY
                 GROUP BY 1 HAVING count(*) >= 5)
        SELECT * FROM (
          SELECT c.venue, c.dd,
                 'net_sales' AS metric, c.net_sales AS val, h.mu_s AS baseline,
                 (c.net_sales - h.mu_s)/nullif(h.sd_s,0) AS z
          FROM cur c JOIN hist h USING (venue) WHERE abs((c.net_sales - h.mu_s)/nullif(h.sd_s,0)) >= %f
          UNION ALL
          SELECT c.venue, c.dd, 'txns', c.txns, h.mu_t,
                 (c.txns - h.mu_t)/nullif(h.sd_t,0)
          FROM cur c JOIN hist h USING (venue) WHERE abs((c.txns - h.mu_t)/nullif(h.sd_t,0)) >= %f
          UNION ALL
          SELECT c.venue, c.dd, 'atv', c.atv, h.mu_a,
                 (c.atv - h.mu_a)/nullif(h.sd_a,0)
          FROM cur c JOIN hist h USING (venue) WHERE abs((c.atv - h.mu_a)/nullif(h.sd_a,0)) >= %f
        ) ORDER BY abs(z) DESC LIMIT 20""" % (Z_THRESHOLD, Z_THRESHOLD, Z_THRESHOLD),
        max_rows=20)

    # prune old anomaly signals, then surface the top few
    try:
        import urllib.parse
        import urllib.request
        import os
        cutoff = (dt.datetime.utcnow() - dt.timedelta(days=7)).isoformat() + "Z"
        qs = urllib.parse.urlencode({"source": "eq." + SOURCE,
                                     "created_at": "lt." + cutoff})
        req = urllib.request.Request(
            "%s/rest/v1/signals?%s" % (os.environ["SUPABASE_URL"].rstrip("/"), qs),
            headers={"apikey": os.environ["SUPABASE_SERVICE_ROLE_KEY"],
                     "Authorization": "Bearer " + os.environ["SUPABASE_SERVICE_ROLE_KEY"],
                     "Accept-Profile": "taskapp", "Content-Profile": "taskapp"},
            method="DELETE")
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        LOG.exception("anomaly signal prune failed (non-fatal)")

    labels = {"net_sales": "sales", "txns": "transactions", "atv": "avg ticket"}
    inserted = 0
    for row in res["rows"][:MAX_SIGNALS]:
        venue, day, metric, value, baseline, z = row
        direction = "above" if z > 0 else "below"
        headline = "Anomaly: %s %s %.1fσ %s its normal %s" % (
            venue, labels.get(metric, metric), abs(z), direction,
            str(day)[:10])
        existing = tk_db.get("signals", {"headline": "eq." + headline,
                                         "select": "id", "limit": "1"})
        if existing:
            continue
        tk_db.insert("signals", [{
            "kind": "business", "headline": headline[:200],
            "detail": "%s on %s: %s %.0f vs same-weekday 8-week average %.0f (z=%+.1f)."
                      % (venue, str(day)[:10], labels.get(metric, metric),
                         float(value or 0), float(baseline or 0), z),
            "source": SOURCE}])
        inserted += 1
    return {"candidates": len(res["rows"]), "signals_added": inserted}
