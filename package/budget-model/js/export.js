const ExportEngine = {
    downloadXlsx(data, headers, filename) {
        const ws = XLSX.utils.json_to_sheet(data, { header: headers });
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, 'Data');
        XLSX.writeFile(wb, filename);
    },

    exportDailyForecast() {
        this.downloadXlsx(
            this.getDailyAccountLines(),
            [
                'venue', 'state', 'date', 'account_code', 'account_name', 'account_type',
                'amount', 'transactions', 'avg_ticket', 'ramp_up_multiplier',
                'growth_multiplier', 'source', 'similar_venue_key'
            ],
            'daily_forecast_account_lines.xlsx'
        );
    },

    exportMonthlySummary() {
        this.downloadXlsx(
            this.getMonthlyAccountLines(),
            [
                'venue', 'state', 'budget_month', 'account_code', 'account_name',
                'account_type', 'amount', 'transactions', 'trading_days'
            ],
            'monthly_summary_account_lines.xlsx'
        );
    },

    exportPnl() {
        this.downloadXlsx(
            this.getMonthlyAccountLines(),
            [
                'venue', 'state', 'budget_month', 'account_code', 'account_name',
                'account_type', 'amount', 'transactions', 'trading_days'
            ],
            'budget_pnl_account_lines.xlsx'
        );
    },

    buildPnlSheet(venueKey) {
        const pnl = PnlBuilder.getPnlTable(venueKey, 'monthly');
        const rows = [];
        const headers = ['Line Item', ...pnl.months.map(m => m.label), 'Total'];
        rows.push(headers);

        for (const row of pnl.rows) {
            const r = [row.label];
            for (const month of pnl.months) {
                r.push(Math.round(row.values[month.key] || 0));
            }
            r.push(Math.round(row.total));
            rows.push(r);
        }
        return rows;
    },

    exportVariance() {
        const variance = PnlBuilder.getVarianceTable('__all__');
        const data = variance.map(v => ({
            account_name: v.label,
            budget: Math.round(v.budget),
            prior_year: Math.round(v.prior),
            variance: Math.round(v.variance),
            variance_pct: Math.round(v.variancePct * 10) / 10
        }));
        this.downloadXlsx(data, null, 'variance_account_lines.xlsx');
    },

    getDailyAccountLines() {
        const rows = [];
        for (const f of PnlBuilder.dailyResults) {
            for (const item of CONFIG.PNL_LINE_ITEMS) {
                rows.push({
                    venue: f.venue_name,
                    state: f.state,
                    date: f.forecast_date,
                    account_code: item.key,
                    account_name: item.label,
                    account_type: item.type,
                    amount: Math.round((f[item.key] || 0) * 100) / 100,
                    transactions: item.key === 'net_sales' ? f.forecast_transactions : null,
                    avg_ticket: item.key === 'net_sales' ? f.avg_ticket : null,
                    ramp_up_multiplier: f.ramp_up_multiplier,
                    growth_multiplier: f.growth_multiplier || 1,
                    source: f.source,
                    similar_venue_key: f.similar_venue_key || null
                });
            }
        }
        return rows;
    },

    getMonthlyAccountLines() {
        const rows = [];
        for (const m of PnlBuilder.monthlySummary) {
            for (const item of CONFIG.PNL_LINE_ITEMS) {
                rows.push({
                    venue: m.venue_name,
                    state: m.state,
                    budget_month: m.budget_month,
                    account_code: item.key,
                    account_name: item.label,
                    account_type: item.type,
                    amount: Math.round((m[item.key] || 0) * 100) / 100,
                    transactions: item.key === 'net_sales' ? m.transaction_count : null,
                    trading_days: item.key === 'net_sales' ? m.trading_days : null
                });
            }
        }
        return rows;
    },

    dedupeBy(rows, keyFn) {
        const map = new Map();
        for (const row of rows) {
            const key = keyFn(row);
            if (!key.includes('undefined') && !key.includes('null')) {
                map.set(key, row);
            }
        }
        return [...map.values()];
    },

    generateTemplates() {
        const wb = XLSX.utils.book_new();

        const salesHeaders = [['venue_name']];
        const start = new Date(CONFIG.BUDGET_YEAR_START);
        start.setFullYear(start.getFullYear() - 2);
        const end = new Date(CONFIG.BUDGET_YEAR_START);
        end.setDate(end.getDate() - 1);
        const current = new Date(start);
        while (current <= end) {
            salesHeaders[0].push(current.toISOString().substring(0, 10));
            current.setDate(current.getDate() + 1);
        }
        salesHeaders.push(['Venue 1']);
        const wsSales = XLSX.utils.aoa_to_sheet(salesHeaders);
        XLSX.utils.book_append_sheet(wb, wsSales, 'Daily Sales');

        const wsPnl = XLSX.utils.aoa_to_sheet([
            ['Line Item', 'Jul-25', 'Aug-25', 'Sep-25', 'Oct-25', 'Nov-25', 'Dec-25', 'Jan-26', 'Feb-26', 'Mar-26', 'Apr-26', 'May-26', 'Jun-26', 'Total'],
            ['Net Sales'], ['COGS - Food'], ['COGS - Packaging'], ['COGS - Retail'], ['COGS - Discounts'],
            ['Total COGS'], ['Gross Profit'], ['Labour - Crew'], ['Labour - Crew Oncosts'],
            ['Labour - Management'], ['Labour - Mgmt Oncosts'], ['Total Labour'],
            ['Occupancy - Base Rent'], ['Occupancy - Outgoings'], ['Occupancy - % Rent'],
            ['Occupancy - Marketing Levy'], ['Total Occupancy'], ['Venue Contribution']
        ]);
        XLSX.utils.book_append_sheet(wb, wsPnl, 'Venue 1');

        const wsVenues = XLSX.utils.aoa_to_sheet([
            ['venue_name', 'state', 'opening_date', 'is_active'],
            ['Venue 1', 'NSW', '2020-01-15', 'Y']
        ]);
        XLSX.utils.book_append_sheet(wb, wsVenues, 'Venues');

        const rampHeaders = ['venue_name'];
        for (let i = 1; i <= 18; i++) rampHeaders.push(`month_${i}`);
        const wsRamp = XLSX.utils.aoa_to_sheet([
            rampHeaders,
            ['Venue 1', 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 1.00]
        ]);
        XLSX.utils.book_append_sheet(wb, wsRamp, 'Ramp Up');

        const budgetStart = new Date(CONFIG.BUDGET_YEAR_START);
        const ticketHeaders = ['venue_name'];
        for (let i = 0; i < 12; i++) {
            const d = new Date(budgetStart);
            d.setMonth(d.getMonth() + i);
            ticketHeaders.push(d.toISOString().substring(0, 10));
        }
        const wsTicket = XLSX.utils.aoa_to_sheet([ticketHeaders, ['Venue 1']]);
        XLSX.utils.book_append_sheet(wb, wsTicket, 'Avg Ticket');

        const wsLabour = XLSX.utils.aoa_to_sheet([
            ['venue_name', 'Sales Per Labour Hour', 'Avg Hourly Rate', 'Oncosts %', 'Mgmt Salary Monthly', 'Mgmt Oncosts %'],
            ['Venue 1', 120, 28.50, 12, 6500, 12]
        ]);
        XLSX.utils.book_append_sheet(wb, wsLabour, 'Labour');

        const wsCogs = XLSX.utils.aoa_to_sheet([
            ['venue_name', 'Food %', 'Packaging %', 'Retail %', 'Discount %'],
            ['Venue 1', 25, 3, 2, 5]
        ]);
        XLSX.utils.book_append_sheet(wb, wsCogs, 'COGS');

        const wsRent = XLSX.utils.aoa_to_sheet([
            ['venue_name', 'Base Rent Monthly', 'Outgoings Monthly', '% Rent Threshold', '% Rent Rate', 'Marketing Levy %'],
            ['Venue 1', 8000, 2500, 50000, 8, 2]
        ]);
        XLSX.utils.book_append_sheet(wb, wsRent, 'Rent');

        XLSX.writeFile(wb, 'budget_templates.xlsx');
    },

    async pushToSupabase(onProgress) {
        const url = document.getElementById('supabase-url').value.trim();
        const key = document.getElementById('supabase-key').value.trim();

        if (!SupabaseClient.init(url, key)) throw new Error('Invalid Supabase credentials');

        const runName = document.getElementById('run-name').value || 'Budget Run';
        const runId = await SupabaseClient.createBudgetRun(runName, {
            forecast_start: CONFIG.BUDGET_YEAR_START,
            forecast_end: CONFIG.BUDGET_YEAR_END,
            created: new Date().toISOString()
        });

        onProgress(5, 'Created budget run...');

        const venues = ExcelParser.uploads.venue_details?.venues || [];
        if (venues.length) {
            await SupabaseClient.upsertVenues(venues.map(v => ({
                venue_name: v.venue_name,
                state: v.state,
                opening_date: v.opening_date,
                is_active: v.is_active
            })));
        }
        const venueIdMap = await SupabaseClient.getVenueIdMap();
        const missingVenueIds = venues
            .filter(v => !venueIdMap[v.venue_key])
            .map(v => v.venue_name);
        if (missingVenueIds.length) {
            throw new Error(`Could not resolve Supabase venue IDs for: ${missingVenueIds.slice(0, 10).join(', ')}`);
        }
        onProgress(15, 'Uploaded venues...');

        const daily = PnlBuilder.dailyResults;
        if (daily.length) {
            const dailyRows = this.dedupeBy(daily.map(f => ({
                venue_id: venueIdMap[f.venue_key],
                forecast_date: f.forecast_date,
                forecast_transactions: f.forecast_transactions,
                avg_ticket: f.avg_ticket,
                gross_sales: f.gross_sales,
                net_sales: f.net_sales,
                ramp_up_multiplier: f.ramp_up_multiplier,
                cogs_food: f.cogs_food,
                cogs_packaging: f.cogs_packaging,
                cogs_retail: f.cogs_retail,
                cogs_discounts: f.cogs_discounts,
                cogs_total: f.cogs_total,
                crew_labour_hours: f.crew_labour_hours,
                crew_labour_cost: f.crew_labour_cost,
                crew_oncosts: f.crew_oncosts,
                mgmt_labour_cost: f.mgmt_labour_cost,
                mgmt_oncosts: f.mgmt_oncosts,
                labour_total: f.labour_total,
                rent_base: f.rent_base,
                rent_outgoings: f.rent_outgoings,
                rent_percentage: f.rent_percentage,
                rent_marketing_levy: f.rent_marketing_levy,
                occupancy_total: f.occupancy_total,
                gross_profit: f.gross_profit,
                venue_contribution: f.venue_contribution
            })), r => `${r.venue_id}_${r.forecast_date}`);

            await SupabaseClient.writeDailyForecast(runId, dailyRows, (done, total) => {
                const pct = 15 + (done / total) * 70;
                onProgress(pct, `Uploading forecasts... ${done}/${total}`);
            });
        }
        onProgress(90, 'Uploading monthly summary...');

        const monthly = PnlBuilder.monthlySummary;
        if (monthly.length) {
            const monthlyRows = this.dedupeBy(monthly.map(m => ({
                venue_id: venueIdMap[m.venue_key],
                budget_month: m.budget_month,
                net_sales: m.net_sales,
                cogs_total: m.cogs_total,
                gross_profit: m.gross_profit,
                labour_total: m.labour_total,
                occupancy_total: m.occupancy_total,
                venue_contribution: m.venue_contribution,
                transaction_count: m.transaction_count,
                trading_days: m.trading_days
            })), r => `${r.venue_id}_${r.budget_month}`);

            await SupabaseClient.writeMonthlySummary(runId, monthlyRows);
        }
        onProgress(100, 'Complete!');
        return runId;
    }
};
