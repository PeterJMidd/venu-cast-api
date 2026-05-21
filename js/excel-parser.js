const ExcelParser = {
    uploads: {},

    readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const wb = XLSX.read(e.target.result, { type: 'array', cellDates: true });
                    resolve(wb);
                } catch (err) {
                    reject(err);
                }
            };
            reader.onerror = () => reject(new Error('File read failed'));
            reader.readAsArrayBuffer(file);
        });
    },

    normaliseVenueName(name) {
        if (!name) return '';
        return String(name).trim().toLowerCase().replace(/\s+/g, ' ');
    },

    formatDate(d) {
        if (!d) return null;
        if (d instanceof Date) {
            const y = d.getFullYear();
            const m = String(d.getMonth() + 1).padStart(2, '0');
            const day = String(d.getDate()).padStart(2, '0');
            return `${y}-${m}-${day}`;
        }
        if (typeof d === 'number') {
            const date = XLSX.SSF.parse_date_code(d);
            return `${date.y}-${String(date.m).padStart(2, '0')}-${String(date.d).padStart(2, '0')}`;
        }
        return String(d);
    },

    firstOfMonth(d) {
        if (d instanceof Date) {
            const y = d.getFullYear();
            const m = String(d.getMonth() + 1).padStart(2, '0');
            return `${y}-${m}-01`;
        }
        const str = this.formatDate(d);
        if (str && str.length >= 7) return str.substring(0, 7) + '-01';
        return str;
    },

    sheetToRows(wb, sheetName) {
        const ws = wb.Sheets[sheetName || wb.SheetNames[0]];
        if (!ws) return [];
        return XLSX.utils.sheet_to_json(ws, { defval: null });
    },

    // 1. Sales History: rows=venues, columns=dates
    parseSalesHistory(wb) {
        const ws = wb.Sheets[wb.SheetNames[0]];
        const data = XLSX.utils.sheet_to_json(ws, { header: 1, cellDates: true });
        const errors = [];
        const records = [];

        if (data.length < 2) {
            errors.push('Sales history sheet must have a header row and at least one venue row');
            return { records, errors };
        }

        const headers = data[0];
        const venueCol = 0;
        const dateColumns = [];

        for (let c = 1; c < headers.length; c++) {
            const dateStr = this.formatDate(headers[c]);
            if (dateStr && /^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
                dateColumns.push({ col: c, date: dateStr });
            }
        }

        if (dateColumns.length === 0) {
            errors.push('No valid date columns found. Columns should be dates (e.g. 2024-05-01)');
            return { records, errors };
        }

        for (let r = 1; r < data.length; r++) {
            const row = data[r];
            const venueName = String(row[venueCol] || '').trim();
            if (!venueName) continue;

            for (const dc of dateColumns) {
                const val = row[dc.col];
                if (val == null || val === '') continue;
                const sales = Number(val);
                if (isNaN(sales)) continue;
                records.push({
                    venue_name: venueName,
                    venue_key: this.normaliseVenueName(venueName),
                    sale_date: dc.date,
                    gross_sales: Math.round(sales * 100) / 100
                });
            }
        }

        return { records, errors, dateRange: { start: dateColumns[0].date, end: dateColumns[dateColumns.length - 1].date } };
    },

    // 2. Prior P&L: one sheet per venue, rows=line items, columns=months
    parsePriorPnl(wb) {
        const records = [];
        const errors = [];

        for (const sheetName of wb.SheetNames) {
            if (sheetName.toLowerCase() === 'instructions') continue;
            const venueName = sheetName.trim();
            const ws = wb.Sheets[sheetName];
            const data = XLSX.utils.sheet_to_json(ws, { header: 1, cellDates: true });

            if (data.length < 2) continue;

            const headers = data[0];
            const monthColumns = [];
            for (let c = 1; c < headers.length; c++) {
                const h = headers[c];
                if (!h) continue;
                if (String(h).toLowerCase() === 'total') continue;
                const monthStr = this.firstOfMonth(h);
                if (monthStr) monthColumns.push({ col: c, month: monthStr });
            }

            for (let r = 1; r < data.length; r++) {
                const row = data[r];
                const lineItem = String(row[0] || '').trim();
                if (!lineItem) continue;

                for (const mc of monthColumns) {
                    const val = row[mc.col];
                    if (val == null || val === '') continue;
                    const amount = Number(val);
                    if (isNaN(amount)) continue;
                    records.push({
                        venue_name: venueName,
                        venue_key: this.normaliseVenueName(venueName),
                        period_month: mc.month,
                        line_item: lineItem,
                        amount: Math.round(amount * 100) / 100
                    });
                }
            }
        }

        return { records, errors };
    },

    // 3. Venue Details: "Venues" sheet + "Ramp Up" sheet
    parseVenueDetails(wb) {
        const errors = [];
        const venues = [];
        const rampUp = {};

        const venueRows = this.sheetToRows(wb, 'Venues') || this.sheetToRows(wb);
        for (const row of venueRows) {
            const name = String(row.venue_name || row['Venue Name'] || row['Venue'] || '').trim();
            const state = String(row.state || row['State'] || '').trim().toUpperCase();
            const openingRaw = row.opening_date || row['Opening Date'] || row['opening date'];
            const openingDate = this.formatDate(openingRaw);
            const isActive = row.is_active !== false && row.is_active !== 'N' && row.is_active !== 0;

            if (!name) continue;
            if (!CONFIG.STATES.includes(state)) {
                errors.push(`Venue "${name}": invalid state "${state}"`);
                continue;
            }

            venues.push({
                venue_name: name,
                venue_key: this.normaliseVenueName(name),
                state,
                opening_date: openingDate,
                is_active: isActive
            });
        }

        const rampSheet = wb.Sheets['Ramp Up'] || wb.Sheets['RampUp'] || wb.Sheets['Ramp_Up'];
        if (rampSheet) {
            const rampData = XLSX.utils.sheet_to_json(rampSheet, { header: 1 });
            if (rampData.length >= 2) {
                for (let r = 1; r < rampData.length; r++) {
                    const row = rampData[r];
                    const name = String(row[0] || '').trim();
                    if (!name) continue;
                    const key = this.normaliseVenueName(name);
                    rampUp[key] = [];
                    for (let m = 1; m <= 18; m++) {
                        const val = Number(row[m]);
                        rampUp[key].push(isNaN(val) ? 1.0 : val);
                    }
                }
            }
        }

        return { venues, rampUp, errors };
    },

    // 4. Average Ticket: rows=venues, columns=months
    parseAvgTicket(wb) {
        const ws = wb.Sheets[wb.SheetNames[0]];
        const data = XLSX.utils.sheet_to_json(ws, { header: 1, cellDates: true });
        const errors = [];
        const records = [];

        if (data.length < 2) return { records, errors };

        const headers = data[0];
        const monthColumns = [];
        for (let c = 1; c < headers.length; c++) {
            const monthStr = this.firstOfMonth(headers[c]);
            if (monthStr) monthColumns.push({ col: c, month: monthStr });
        }

        for (let r = 1; r < data.length; r++) {
            const row = data[r];
            const name = String(row[0] || '').trim();
            if (!name) continue;
            const key = this.normaliseVenueName(name);

            for (const mc of monthColumns) {
                const val = Number(row[mc.col]);
                if (isNaN(val) || val <= 0) continue;
                records.push({
                    venue_name: name,
                    venue_key: key,
                    budget_month: mc.month,
                    avg_ticket: Math.round(val * 100) / 100
                });
            }
        }

        return { records, errors };
    },

    // 5. Labour assumptions
    parseLabour(wb) {
        const rows = this.sheetToRows(wb);
        const errors = [];
        const records = [];

        for (const row of rows) {
            const name = String(row.venue_name || row['Venue Name'] || row['Venue'] || '').trim();
            if (!name) continue;

            records.push({
                venue_name: name,
                venue_key: this.normaliseVenueName(name),
                sales_per_labour_hr: Number(row.sales_per_labour_hr || row['Sales Per Labour Hour'] || row['SPLH'] || 0),
                avg_hourly_rate: Number(row.avg_hourly_rate || row['Avg Hourly Rate'] || row['Hourly Rate'] || 0),
                oncosts_pct: Number(row.oncosts_pct || row['Oncosts %'] || row['On Costs %'] || 0) / 100,
                mgmt_salary_monthly: Number(row.mgmt_salary_monthly || row['Mgmt Salary Monthly'] || row['Management Salary'] || 0),
                mgmt_oncosts_pct: Number(row.mgmt_oncosts_pct || row['Mgmt Oncosts %'] || row['Mgmt On Costs %'] || 0) / 100
            });
        }

        return { records, errors };
    },

    // 6. COGS assumptions
    parseCogs(wb) {
        const rows = this.sheetToRows(wb);
        const errors = [];
        const records = [];

        for (const row of rows) {
            const name = String(row.venue_name || row['Venue Name'] || row['Venue'] || '').trim();
            if (!name) continue;
            const key = this.normaliseVenueName(name);

            const categories = [
                { field: 'food_pct', alt: ['Food %', 'Food'], category: 'food' },
                { field: 'packaging_pct', alt: ['Packaging %', 'Packaging'], category: 'packaging' },
                { field: 'retail_pct', alt: ['Retail %', 'Retail'], category: 'retail' },
                { field: 'discount_pct', alt: ['Discount %', 'Discounts %', 'Sale Discounts %'], category: 'sale_discounts' }
            ];

            for (const cat of categories) {
                let val = row[cat.field];
                if (val == null) {
                    for (const alt of cat.alt) {
                        if (row[alt] != null) { val = row[alt]; break; }
                    }
                }
                val = Number(val || 0);
                records.push({
                    venue_name: name,
                    venue_key: key,
                    category: cat.category,
                    cogs_pct: val / 100
                });
            }
        }

        return { records, errors };
    },

    // 7. Rent assumptions
    parseRent(wb) {
        const rows = this.sheetToRows(wb);
        const errors = [];
        const records = [];

        for (const row of rows) {
            const name = String(row.venue_name || row['Venue Name'] || row['Venue'] || '').trim();
            if (!name) continue;

            records.push({
                venue_name: name,
                venue_key: this.normaliseVenueName(name),
                base_rent_monthly: Number(row.base_rent_monthly || row['Base Rent Monthly'] || row['Base Rent'] || 0),
                outgoings_monthly: Number(row.outgoings_monthly || row['Outgoings Monthly'] || row['Outgoings'] || 0),
                pct_rent_threshold: Number(row.pct_rent_threshold || row['% Rent Threshold'] || row['Pct Rent Threshold'] || 0),
                pct_rent_rate: Number(row.pct_rent_rate || row['% Rent Rate'] || row['Pct Rent Rate'] || 0) / 100,
                marketing_levy_pct: Number(row.marketing_levy_pct || row['Marketing Levy %'] || row['Marketing Levy'] || 0) / 100
            });
        }

        return { records, errors };
    },

    validateCrossTemplate() {
        const errors = [];
        const venueDetails = this.uploads.venue_details;
        if (!venueDetails) return ['Venue Details template must be uploaded first'];

        const masterKeys = new Set(venueDetails.venues.map(v => v.venue_key));

        const checkTemplate = (name, data, keyField) => {
            if (!data) return;
            const keys = new Set();
            for (const r of data) keys.add(r[keyField || 'venue_key']);
            for (const k of keys) {
                if (!masterKeys.has(k)) {
                    errors.push(`${name}: venue "${k}" not found in Venue Details`);
                }
            }
        };

        if (this.uploads.sales_history) checkTemplate('Sales History', this.uploads.sales_history.records, 'venue_key');
        if (this.uploads.avg_ticket) checkTemplate('Avg Ticket', this.uploads.avg_ticket.records, 'venue_key');
        if (this.uploads.labour) checkTemplate('Labour', this.uploads.labour.records, 'venue_key');
        if (this.uploads.cogs) checkTemplate('COGS', this.uploads.cogs.records, 'venue_key');
        if (this.uploads.rent) checkTemplate('Rent', this.uploads.rent.records, 'venue_key');

        return errors;
    },

    getUploadSummary() {
        const summary = {};
        if (this.uploads.sales_history) {
            const s = this.uploads.sales_history;
            summary.sales_history = { records: s.records.length, dateRange: s.dateRange };
        }
        if (this.uploads.prior_pnl) {
            summary.prior_pnl = { records: this.uploads.prior_pnl.records.length };
        }
        if (this.uploads.venue_details) {
            summary.venue_details = { venues: this.uploads.venue_details.venues.length };
        }
        if (this.uploads.avg_ticket) {
            summary.avg_ticket = { records: this.uploads.avg_ticket.records.length };
        }
        if (this.uploads.labour) {
            summary.labour = { records: this.uploads.labour.records.length };
        }
        if (this.uploads.cogs) {
            summary.cogs = { records: this.uploads.cogs.records.length };
        }
        if (this.uploads.rent) {
            summary.rent = { records: this.uploads.rent.records.length };
        }
        return summary;
    }
};
