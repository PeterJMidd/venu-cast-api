const SupabaseClient = {
    client: null,

    init(url, key) {
        if (!url || !key) return false;
        const cleanUrl = this.normaliseUrl(url);
        this.client = supabase.createClient(cleanUrl, key);
        return true;
    },

    normaliseUrl(url) {
        return String(url || '')
            .trim()
            .replace(/\/rest\/v1\/?$/i, '')
            .replace(/\/+$/, '');
    },

    async testConnection() {
        if (!this.client) throw new Error('Supabase not initialised');
        const { data, error } = await this.client.from('budget_runs').select('run_id').limit(1);
        if (error) throw error;
        return true;
    },

    async createBudgetRun(name, params) {
        const { data, error } = await this.client
            .from('budget_runs')
            .insert({ run_name: name, parameters: params })
            .select('run_id')
            .single();
        if (error) throw error;
        return data.run_id;
    },

    async upsertVenues(venues) {
        const { error } = await this.client
            .from('venues')
            .upsert(venues, { onConflict: 'venue_name' });
        if (error) throw error;
    },

    async getVenueIdMap() {
        const { data, error } = await this.client
            .from('venues')
            .select('venue_id, venue_name');
        if (error) throw error;

        const map = {};
        for (const venue of data || []) {
            const key = String(venue.venue_name || '').trim().toLowerCase().replace(/\s+/g, ' ');
            map[key] = venue.venue_id;
        }
        return map;
    },

    async upsertVenueRampUp(rows) {
        const { error } = await this.client
            .from('venue_ramp_up')
            .upsert(rows, { onConflict: 'venue_id,month_number' });
        if (error) throw error;
    },

    async writeSalesHistory(rows, onProgress) {
        await this._batchUpsert('sales_history', rows, 'venue_id,sale_date', onProgress);
    },

    async writeWeatherData(rows) {
        await this._batchUpsert('weather_data', rows, 'state,observation_date');
    },

    async writePriorPnl(rows) {
        await this._batchUpsert('prior_pnl', rows, 'venue_id,period_month,line_item');
    },

    async writeAvgTicket(runId, rows) {
        const tagged = rows.map(r => ({ ...r, run_id: runId }));
        await this._batchUpsert('avg_ticket_assumptions', tagged, 'run_id,venue_id,budget_month');
    },

    async writeLabour(runId, rows) {
        const tagged = rows.map(r => ({ ...r, run_id: runId }));
        await this._batchUpsert('labour_assumptions', tagged, 'run_id,venue_id');
    },

    async writeCogs(runId, rows) {
        const tagged = rows.map(r => ({ ...r, run_id: runId }));
        await this._batchUpsert('cogs_assumptions', tagged, 'run_id,venue_id,category');
    },

    async writeRent(runId, rows) {
        const tagged = rows.map(r => ({ ...r, run_id: runId }));
        await this._batchUpsert('rent_assumptions', tagged, 'run_id,venue_id');
    },

    async writeDailyForecast(runId, rows, onProgress) {
        const tagged = rows.map(r => ({ ...r, run_id: runId }));
        await this._batchUpsert('daily_forecast', tagged, 'run_id,venue_id,forecast_date', onProgress);
    },

    async writeMonthlySummary(runId, rows) {
        const tagged = rows.map(r => ({ ...r, run_id: runId }));
        await this._batchUpsert('monthly_summary', tagged, 'run_id,venue_id,budget_month');
    },

    async _batchUpsert(table, rows, onConflict, onProgress) {
        const batchSize = CONFIG.SUPABASE_BATCH_SIZE;
        const conflictColumns = onConflict.split(',').map(c => c.trim()).filter(Boolean);
        const dedupedRows = this._dedupeConflictRows(rows, conflictColumns);
        const total = dedupedRows.length;
        let done = 0;

        for (let i = 0; i < total; i += batchSize) {
            const batch = dedupedRows.slice(i, i + batchSize);
            const { error } = await this.client
                .from(table)
                .upsert(batch, { onConflict });
            if (error) throw new Error(`${table} batch ${i}: ${error.message}`);
            done += batch.length;
            if (onProgress) onProgress(done, total);
        }
    },

    _dedupeConflictRows(rows, conflictColumns) {
        if (!conflictColumns.length) return rows;
        const map = new Map();
        for (const row of rows) {
            const keyParts = conflictColumns.map(col => row[col]);
            if (keyParts.some(v => v === undefined || v === null || v === '')) continue;
            map.set(keyParts.join('|'), row);
        }
        return [...map.values()];
    },

    getSchemaSQL() {
        return `
-- Frozen Yoghurt QSR Budget Model — Supabase Schema

CREATE TABLE IF NOT EXISTS venues (
    venue_id        SERIAL PRIMARY KEY,
    venue_name      TEXT NOT NULL UNIQUE,
    state           TEXT NOT NULL,
    opening_date    DATE NOT NULL,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS venue_ramp_up (
    venue_id        INT REFERENCES venues(venue_id),
    month_number    INT NOT NULL CHECK (month_number BETWEEN 1 AND 18),
    multiplier      NUMERIC(5,4) NOT NULL,
    PRIMARY KEY (venue_id, month_number)
);

CREATE TABLE IF NOT EXISTS budget_runs (
    run_id          SERIAL PRIMARY KEY,
    run_name        TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    parameters      JSONB
);

CREATE TABLE IF NOT EXISTS sales_history (
    venue_id        INT REFERENCES venues(venue_id),
    sale_date       DATE NOT NULL,
    gross_sales     NUMERIC(12,2) NOT NULL,
    transaction_count INT,
    PRIMARY KEY (venue_id, sale_date)
);

CREATE TABLE IF NOT EXISTS weather_data (
    state           TEXT NOT NULL,
    observation_date DATE NOT NULL,
    max_temp_c      NUMERIC(4,1),
    min_temp_c      NUMERIC(4,1),
    rainfall_mm     NUMERIC(6,1),
    sunshine_hours  NUMERIC(4,1),
    PRIMARY KEY (state, observation_date)
);

CREATE TABLE IF NOT EXISTS seasonality_indices (
    run_id          INT REFERENCES budget_runs(run_id),
    venue_id        INT REFERENCES venues(venue_id),
    day_of_week     INT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    month_of_year   INT NOT NULL CHECK (month_of_year BETWEEN 1 AND 12),
    is_school_holiday BOOLEAN DEFAULT FALSE,
    is_public_holiday BOOLEAN DEFAULT FALSE,
    index_value     NUMERIC(8,6) NOT NULL,
    PRIMARY KEY (run_id, venue_id, day_of_week, month_of_year, is_school_holiday, is_public_holiday)
);

CREATE TABLE IF NOT EXISTS prior_pnl (
    venue_id        INT REFERENCES venues(venue_id),
    period_month    DATE NOT NULL,
    line_item       TEXT NOT NULL,
    amount          NUMERIC(12,2) NOT NULL,
    PRIMARY KEY (venue_id, period_month, line_item)
);

CREATE TABLE IF NOT EXISTS avg_ticket_assumptions (
    run_id          INT REFERENCES budget_runs(run_id),
    venue_id        INT REFERENCES venues(venue_id),
    budget_month    DATE NOT NULL,
    avg_ticket      NUMERIC(8,2) NOT NULL,
    PRIMARY KEY (run_id, venue_id, budget_month)
);

CREATE TABLE IF NOT EXISTS labour_assumptions (
    run_id              INT REFERENCES budget_runs(run_id),
    venue_id            INT REFERENCES venues(venue_id),
    sales_per_labour_hr NUMERIC(8,2) NOT NULL,
    avg_hourly_rate     NUMERIC(8,2) NOT NULL,
    oncosts_pct         NUMERIC(5,4) NOT NULL,
    mgmt_salary_monthly NUMERIC(10,2) NOT NULL,
    mgmt_oncosts_pct    NUMERIC(5,4) NOT NULL,
    PRIMARY KEY (run_id, venue_id)
);

CREATE TABLE IF NOT EXISTS cogs_assumptions (
    run_id          INT REFERENCES budget_runs(run_id),
    venue_id        INT REFERENCES venues(venue_id),
    category        TEXT NOT NULL,
    cogs_pct        NUMERIC(5,4) NOT NULL,
    PRIMARY KEY (run_id, venue_id, category)
);

CREATE TABLE IF NOT EXISTS rent_assumptions (
    run_id          INT REFERENCES budget_runs(run_id),
    venue_id        INT REFERENCES venues(venue_id),
    base_rent_monthly    NUMERIC(10,2) NOT NULL,
    outgoings_monthly    NUMERIC(10,2) NOT NULL,
    pct_rent_threshold   NUMERIC(12,2),
    pct_rent_rate        NUMERIC(5,4),
    marketing_levy_pct   NUMERIC(5,4),
    PRIMARY KEY (run_id, venue_id)
);

CREATE TABLE IF NOT EXISTS daily_forecast (
    run_id              INT REFERENCES budget_runs(run_id),
    venue_id            INT REFERENCES venues(venue_id),
    forecast_date       DATE NOT NULL,
    forecast_transactions INT,
    avg_ticket          NUMERIC(8,2),
    gross_sales         NUMERIC(12,2),
    net_sales           NUMERIC(12,2),
    ramp_up_multiplier  NUMERIC(5,4),
    cogs_food           NUMERIC(10,2),
    cogs_packaging      NUMERIC(10,2),
    cogs_retail         NUMERIC(10,2),
    cogs_discounts      NUMERIC(10,2),
    cogs_total          NUMERIC(10,2),
    crew_labour_hours   NUMERIC(8,2),
    crew_labour_cost    NUMERIC(10,2),
    crew_oncosts        NUMERIC(10,2),
    mgmt_labour_cost    NUMERIC(10,2),
    mgmt_oncosts        NUMERIC(10,2),
    labour_total        NUMERIC(10,2),
    rent_base           NUMERIC(10,2),
    rent_outgoings      NUMERIC(10,2),
    rent_percentage     NUMERIC(10,2),
    rent_marketing_levy NUMERIC(10,2),
    occupancy_total     NUMERIC(10,2),
    gross_profit        NUMERIC(12,2),
    venue_contribution  NUMERIC(12,2),
    PRIMARY KEY (run_id, venue_id, forecast_date)
);

CREATE TABLE IF NOT EXISTS monthly_summary (
    run_id          INT REFERENCES budget_runs(run_id),
    venue_id        INT REFERENCES venues(venue_id),
    budget_month    DATE NOT NULL,
    net_sales       NUMERIC(14,2),
    cogs_total      NUMERIC(12,2),
    gross_profit    NUMERIC(14,2),
    labour_total    NUMERIC(12,2),
    occupancy_total NUMERIC(12,2),
    venue_contribution NUMERIC(14,2),
    transaction_count   INT,
    trading_days        INT,
    PRIMARY KEY (run_id, venue_id, budget_month)
);

CREATE INDEX IF NOT EXISTS idx_daily_forecast_date ON daily_forecast(run_id, forecast_date);
CREATE INDEX IF NOT EXISTS idx_daily_forecast_venue ON daily_forecast(run_id, venue_id);
CREATE INDEX IF NOT EXISTS idx_sales_history_date ON sales_history(sale_date);
CREATE INDEX IF NOT EXISTS idx_weather_state_date ON weather_data(state, observation_date);
`;
    }
};
