const PnlBuilder = {
    dailyResults: [],
    monthlySummary: [],
    priorPnlByVenue: {},

    run(data) {
        const {
            venueDetails, salesHistory, avgTicketData, rampUpData,
            labourAssumptions, cogsAssumptions, rentAssumptions,
            monthlyGrowthData, newVenueAssumptions, forecastStart, forecastEnd
        } = data;

        SalesForecastEngine.buildSeasonality(salesHistory, venueDetails);
        const forecasts = SalesForecastEngine.generateForecasts(
            venueDetails, avgTicketData, rampUpData, forecastStart, forecastEnd,
            monthlyGrowthData || {}, newVenueAssumptions || {}
        );

        CogsCalcEngine.calculate(forecasts, cogsAssumptions);
        LabourCalcEngine.calculate(forecasts, labourAssumptions);
        RentCalcEngine.calculate(forecasts, rentAssumptions);

        for (const f of forecasts) {
            f.venue_contribution = Math.round(
                (f.gross_profit - f.labour_total - f.occupancy_total) * 100
            ) / 100;
        }

        this.dailyResults = forecasts;
        this.monthlySummary = this.aggregateMonthly(forecasts);

        return { daily: this.dailyResults, monthly: this.monthlySummary };
    },

    aggregateMonthly(dailyForecasts) {
        const grouped = {};

        for (const f of dailyForecasts) {
            const monthKey = f.forecast_date.substring(0, 7) + '-01';
            const key = `${f.venue_key}_${monthKey}`;

            if (!grouped[key]) {
                grouped[key] = {
                    venue_key: f.venue_key,
                    venue_name: f.venue_name,
                    state: f.state,
                    budget_month: monthKey,
                    net_sales: 0,
                    cogs_food: 0,
                    cogs_packaging: 0,
                    cogs_retail: 0,
                    cogs_discounts: 0,
                    cogs_total: 0,
                    gross_profit: 0,
                    crew_labour_cost: 0,
                    crew_oncosts: 0,
                    mgmt_labour_cost: 0,
                    mgmt_oncosts: 0,
                    labour_total: 0,
                    rent_base: 0,
                    rent_outgoings: 0,
                    rent_percentage: 0,
                    rent_marketing_levy: 0,
                    occupancy_total: 0,
                    venue_contribution: 0,
                    transaction_count: 0,
                    trading_days: 0
                };
            }

            const m = grouped[key];
            m.net_sales += f.net_sales;
            m.cogs_food += f.cogs_food;
            m.cogs_packaging += f.cogs_packaging;
            m.cogs_retail += f.cogs_retail;
            m.cogs_discounts += f.cogs_discounts;
            m.cogs_total += f.cogs_total;
            m.gross_profit += f.gross_profit;
            m.crew_labour_cost += f.crew_labour_cost;
            m.crew_oncosts += f.crew_oncosts;
            m.mgmt_labour_cost += f.mgmt_labour_cost;
            m.mgmt_oncosts += f.mgmt_oncosts;
            m.labour_total += f.labour_total;
            m.rent_base += f.rent_base;
            m.rent_outgoings += f.rent_outgoings;
            m.rent_percentage += f.rent_percentage;
            m.rent_marketing_levy += f.rent_marketing_levy;
            m.occupancy_total += f.occupancy_total;
            m.venue_contribution += f.venue_contribution;
            m.transaction_count += f.forecast_transactions;
            m.trading_days++;
        }

        const result = Object.values(grouped).map(m => {
            const rounded = {};
            for (const [k, v] of Object.entries(m)) {
                rounded[k] = typeof v === 'number' ? Math.round(v * 100) / 100 : v;
            }
            return rounded;
        });

        return result.sort((a, b) =>
            a.venue_key.localeCompare(b.venue_key) || a.budget_month.localeCompare(b.budget_month)
        );
    },

    setPriorPnl(priorPnlRecords) {
        this.priorPnlByVenue = {};
        for (const r of priorPnlRecords) {
            if (!this.priorPnlByVenue[r.venue_key]) this.priorPnlByVenue[r.venue_key] = {};
            const monthKey = r.period_month;
            if (!this.priorPnlByVenue[r.venue_key][monthKey]) this.priorPnlByVenue[r.venue_key][monthKey] = {};
            this.priorPnlByVenue[r.venue_key][monthKey][r.line_item] = r.amount;
        }
    },

    getPnlTable(venueKey, view) {
        const months = this.getMonthsForView(venueKey, view);
        const lineItems = CONFIG.PNL_LINE_ITEMS;

        const rows = lineItems.map(item => {
            const row = { key: item.key, label: item.label, type: item.type, values: {} };
            for (const period of months) {
                const data = this.getPeriodData(venueKey, period);
                row.values[period.key] = data ? (data[item.key] || 0) : 0;
            }
            row.total = Object.values(row.values).reduce((s, v) => s + v, 0);
            return row;
        });

        return { months, rows };
    },

    getMonthsForView(venueKey, view) {
        const summaries = venueKey === '__all__'
            ? this.monthlySummary
            : this.monthlySummary.filter(m => m.venue_key === venueKey);

        const monthSet = new Set(summaries.map(m => m.budget_month));
        const sorted = [...monthSet].sort();

        if (view === 'quarterly') {
            const quarters = {};
            for (const m of sorted) {
                const month = parseInt(m.substring(5, 7));
                const year = parseInt(m.substring(0, 4));
                const q = Math.ceil(month / 3);
                const qKey = `${year}-Q${q}`;
                if (!quarters[qKey]) quarters[qKey] = { key: qKey, label: `Q${q} ${year}`, months: [] };
                quarters[qKey].months.push(m);
            }
            return Object.values(quarters);
        }

        if (view === 'annual') {
            return [{ key: 'annual', label: 'Full Year', months: sorted }];
        }

        return sorted.map(m => {
            const d = new Date(m);
            const label = d.toLocaleDateString('en-AU', { month: 'short', year: '2-digit' });
            return { key: m, label };
        });
    },

    getPeriodData(venueKey, period) {
        if (period.months) {
            const totals = {};
            for (const monthKey of period.months) {
                const monthData = this.getMonthlyData(venueKey, monthKey);
                if (!monthData) continue;
                for (const [k, v] of Object.entries(monthData)) {
                    if (typeof v === 'number') totals[k] = (totals[k] || 0) + v;
                }
            }
            return totals;
        }
        return this.getMonthlyData(venueKey, period.key);
    },

    getMonthlyData(venueKey, monthKey) {
        const summaries = venueKey === '__all__'
            ? this.monthlySummary.filter(m => m.budget_month === monthKey)
            : this.monthlySummary.filter(m => m.venue_key === venueKey && m.budget_month === monthKey);

        if (summaries.length === 0) return null;

        const agg = {};
        for (const s of summaries) {
            for (const [k, v] of Object.entries(s)) {
                if (typeof v === 'number') {
                    agg[k] = (agg[k] || 0) + v;
                }
            }
        }
        return agg;
    },

    getVarianceTable(venueKey) {
        const budget = this.getPnlTable(venueKey, 'annual');
        const priorData = this.getPriorAnnualTotals(venueKey);

        return budget.rows.map(row => ({
            label: row.label,
            type: row.type,
            budget: row.total,
            prior: priorData[row.key] || 0,
            variance: row.total - (priorData[row.key] || 0),
            variancePct: priorData[row.key] ? ((row.total - priorData[row.key]) / Math.abs(priorData[row.key])) * 100 : 0
        }));
    },

    getPriorAnnualTotals(venueKey) {
        const totals = {};
        const venues = venueKey === '__all__'
            ? Object.keys(this.priorPnlByVenue)
            : [venueKey];

        for (const vk of venues) {
            const venueData = this.priorPnlByVenue[vk];
            if (!venueData) continue;
            for (const monthData of Object.values(venueData)) {
                for (const [item, amount] of Object.entries(monthData)) {
                    const mappedKey = this.mapPriorLineItem(item);
                    if (mappedKey) {
                        totals[mappedKey] = (totals[mappedKey] || 0) + amount;
                    }
                }
            }
        }
        return totals;
    },

    mapPriorLineItem(lineItem) {
        const lower = lineItem.toLowerCase().trim();
        const map = {
            'net sales': 'net_sales',
            'cogs - food': 'cogs_food',
            'cogs - packaging': 'cogs_packaging',
            'cogs - retail': 'cogs_retail',
            'cogs - discounts': 'cogs_discounts',
            'total cogs': 'cogs_total',
            'gross profit': 'gross_profit',
            'labour - crew': 'crew_labour_cost',
            'labour - crew oncosts': 'crew_oncosts',
            'labour - management': 'mgmt_labour_cost',
            'labour - mgmt oncosts': 'mgmt_oncosts',
            'total labour': 'labour_total',
            'occupancy - base rent': 'rent_base',
            'occupancy - outgoings': 'rent_outgoings',
            'occupancy - % rent': 'rent_percentage',
            'occupancy - marketing levy': 'rent_marketing_levy',
            'total occupancy': 'occupancy_total',
            'venue contribution': 'venue_contribution'
        };
        return map[lower] || null;
    },

    getNetworkKPIs() {
        const totals = { net_sales: 0, cogs_total: 0, gross_profit: 0, labour_total: 0, occupancy_total: 0, venue_contribution: 0 };
        for (const m of this.monthlySummary) {
            totals.net_sales += m.net_sales;
            totals.cogs_total += m.cogs_total;
            totals.gross_profit += m.gross_profit;
            totals.labour_total += m.labour_total;
            totals.occupancy_total += m.occupancy_total;
            totals.venue_contribution += m.venue_contribution;
        }

        return {
            sales: totals.net_sales,
            gpPct: totals.net_sales ? (totals.gross_profit / totals.net_sales) * 100 : 0,
            labourPct: totals.net_sales ? (totals.labour_total / totals.net_sales) * 100 : 0,
            occupancyPct: totals.net_sales ? (totals.occupancy_total / totals.net_sales) * 100 : 0,
            contributionPct: totals.net_sales ? (totals.venue_contribution / totals.net_sales) * 100 : 0
        };
    },

    getContributionByState() {
        const byState = {};
        for (const m of this.monthlySummary) {
            if (!byState[m.state]) byState[m.state] = { sales: 0, contribution: 0, venues: new Set() };
            byState[m.state].sales += m.net_sales;
            byState[m.state].contribution += m.venue_contribution;
            byState[m.state].venues.add(m.venue_key);
        }
        return Object.entries(byState).map(([state, data]) => ({
            state,
            sales: data.sales,
            contribution: data.contribution,
            venueCount: data.venues.size,
            contributionPct: data.sales ? (data.contribution / data.sales) * 100 : 0
        })).sort((a, b) => b.contribution - a.contribution);
    }
};
