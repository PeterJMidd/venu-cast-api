const App = {
    budgetReady: false,

    init() {
        this.setupTabs();
        this.setupFileUploads();
        this.setupButtons();
        this.loadSavedConfig();
    },

    setupTabs() {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');

                if (tab.dataset.tab === 'sales-forecast' && this.budgetReady) {
                    const venueKey = document.getElementById('forecast-venue-filter').value;
                    Charts.renderAllSalesCharts(venueKey);
                }
                if (tab.dataset.tab === 'dashboard' && this.budgetReady) {
                    this.renderDashboard();
                }
                if (tab.dataset.tab === 'pnl' && this.budgetReady) {
                    this.renderPnlTable();
                }
            });
        });
    },

    setupFileUploads() {
        const templates = {
            sales_history: ExcelParser.parseSalesHistory.bind(ExcelParser),
            prior_pnl: ExcelParser.parsePriorPnl.bind(ExcelParser),
            venue_details: ExcelParser.parseVenueDetails.bind(ExcelParser),
            avg_ticket: ExcelParser.parseAvgTicket.bind(ExcelParser),
            labour: ExcelParser.parseLabour.bind(ExcelParser),
            cogs: ExcelParser.parseCogs.bind(ExcelParser),
            rent: ExcelParser.parseRent.bind(ExcelParser)
        };

        for (const [key, parser] of Object.entries(templates)) {
            const dropZone = document.getElementById(`drop-${key}`);
            const input = dropZone.querySelector('input[type="file"]');
            const status = document.getElementById(`status-${key}`);

            const handleFile = async (file) => {
                status.textContent = 'Parsing...';
                status.className = 'upload-status loading';
                try {
                    const wb = await ExcelParser.readFile(file);
                    const result = parser(wb);
                    ExcelParser.uploads[key] = result;

                    const count = result.records?.length || result.venues?.length || 0;
                    status.textContent = `Loaded: ${count} records`;
                    status.className = 'upload-status success';

                    if (result.errors?.length) {
                        status.textContent += ` (${result.errors.length} warnings)`;
                    }

                    this.updateRunButton();
                    this.showPreview(key, result);

                    if (key === 'venue_details') {
                        this.populateVenueFilters(result.venues);
                        this.renderVenueTable(result.venues);
                    }
                } catch (err) {
                    status.textContent = `Error: ${err.message}`;
                    status.className = 'upload-status error';
                }
            };

            input.addEventListener('change', (e) => {
                if (e.target.files[0]) handleFile(e.target.files[0]);
            });

            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
            });
        }
    },

    setupButtons() {
        document.getElementById('btn-run-budget').addEventListener('click', () => this.runBudget());

        document.getElementById('btn-test-connection').addEventListener('click', async () => {
            const status = document.getElementById('connection-status');
            const url = document.getElementById('supabase-url').value.trim();
            const key = document.getElementById('supabase-key').value.trim();
            try {
                SupabaseClient.init(url, key);
                await SupabaseClient.testConnection();
                status.textContent = 'Connected successfully';
                status.style.color = 'var(--success)';
                document.getElementById('btn-push-supabase').disabled = false;
                document.getElementById('btn-run-schema').disabled = false;
                localStorage.setItem('sb_url', url);
                localStorage.setItem('sb_key', key);
            } catch (err) {
                status.textContent = `Failed: ${err.message}`;
                status.style.color = 'var(--danger)';
            }
        });

        document.getElementById('btn-push-supabase').addEventListener('click', async () => {
            const progress = document.getElementById('push-progress');
            const fill = progress.querySelector('.progress-fill');
            const text = progress.querySelector('.progress-text');
            progress.style.display = 'block';
            try {
                const runId = await ExportEngine.pushToSupabase((pct, msg) => {
                    fill.style.width = `${pct}%`;
                    text.textContent = msg;
                });
                text.textContent = `Done! Run ID: ${runId}`;
            } catch (err) {
                text.textContent = `Error: ${err.message}`;
                fill.style.width = '0%';
            }
        });

        document.getElementById('btn-export-daily').addEventListener('click', () => ExportEngine.exportDailyForecast());
        document.getElementById('btn-export-monthly').addEventListener('click', () => ExportEngine.exportMonthlySummary());
        document.getElementById('btn-export-pnl').addEventListener('click', () => ExportEngine.exportPnl());
        document.getElementById('btn-export-variance').addEventListener('click', () => ExportEngine.exportVariance());
        document.getElementById('btn-dl-templates').addEventListener('click', () => ExportEngine.generateTemplates());

        document.getElementById('btn-copy-schema').addEventListener('click', () => {
            navigator.clipboard.writeText(SupabaseClient.getSchemaSQL());
            document.getElementById('btn-copy-schema').textContent = 'Copied!';
            setTimeout(() => document.getElementById('btn-copy-schema').textContent = 'Copy Schema SQL', 2000);
        });

        document.getElementById('btn-run-schema').addEventListener('click', async () => {
            try {
                const { error } = await SupabaseClient.client.rpc('exec_sql', { sql: SupabaseClient.getSchemaSQL() });
                if (error) throw error;
                alert('Schema created successfully');
            } catch (err) {
                alert(`Schema error: ${err.message}\n\nPlease run the SQL manually in the Supabase SQL editor.`);
            }
        });

        document.getElementById('forecast-venue-filter').addEventListener('change', (e) => {
            if (this.budgetReady) Charts.renderAllSalesCharts(e.target.value);
        });

        document.getElementById('pnl-venue-filter').addEventListener('change', () => {
            if (this.budgetReady) this.renderPnlTable();
        });

        document.getElementById('pnl-view').addEventListener('change', () => {
            if (this.budgetReady) this.renderPnlTable();
        });

        document.getElementById('filter-state').addEventListener('change', () => this.filterVenueTable());
        document.getElementById('filter-maturity').addEventListener('change', () => this.filterVenueTable());
    },

    loadSavedConfig() {
        const url = localStorage.getItem('sb_url');
        const key = localStorage.getItem('sb_key');
        const apiUrl = localStorage.getItem('api_url');
        if (url) document.getElementById('supabase-url').value = url;
        if (key) document.getElementById('supabase-key').value = key;
        if (apiUrl) document.getElementById('api-url').value = apiUrl;
    },

    updateRunButton() {
        const required = ['venue_details', 'avg_ticket', 'labour', 'cogs', 'rent'];
        const hasAll = required.every(k => ExcelParser.uploads[k]);
        const hasSalesOrHistory = ExcelParser.uploads.sales_history;
        document.getElementById('btn-run-budget').disabled = !(hasAll && hasSalesOrHistory);

        const uploaded = Object.keys(ExcelParser.uploads).length;
        document.getElementById('venue-count-display').textContent =
            ExcelParser.uploads.venue_details
                ? `${ExcelParser.uploads.venue_details.venues.length} venues`
                : '';
    },

    async runBudget() {
        const btn = document.getElementById('btn-run-budget');
        const progress = document.getElementById('run-progress');
        const fill = progress.querySelector('.progress-fill');
        const text = progress.querySelector('.progress-text');

        btn.disabled = true;
        progress.style.display = 'block';
        fill.style.width = '0%';
        text.textContent = 'Starting...';

        try {
            const forecastStart = document.getElementById('forecast-start').value || CONFIG.BUDGET_YEAR_START;
            const forecastEnd = document.getElementById('forecast-end').value || CONFIG.BUDGET_YEAR_END;
            CONFIG.BUDGET_YEAR_START = forecastStart;
            CONFIG.BUDGET_YEAR_END = forecastEnd;

            const venueData = ExcelParser.uploads.venue_details;
            const salesData = ExcelParser.uploads.sales_history;
            const avgTicketData = ExcelParser.uploads.avg_ticket;
            const labourData = ExcelParser.uploads.labour;
            const cogsData = ExcelParser.uploads.cogs;
            const rentData = ExcelParser.uploads.rent;
            const priorPnl = ExcelParser.uploads.prior_pnl;

            fill.style.width = '10%';
            text.textContent = 'Fetching weather data...';

            if (document.getElementById('fetch-weather').checked && salesData?.dateRange) {
                const states = [...new Set(venueData.venues.map(v => v.state))];
                await WeatherEngine.fetchHistoricalWeather(
                    states, salesData.dateRange.start, salesData.dateRange.end,
                    (done, total) => {
                        fill.style.width = `${10 + (done / total) * 20}%`;
                        text.textContent = `Weather: ${done}/${total} states...`;
                    }
                );
                WeatherEngine.buildTempBandIndices(salesData.records, venueData.venues);
            }

            // Call your existing Render API for Prophet/SARIMA forecasts
            const apiUrl = document.getElementById('api-url').value.trim();
            const useApi = document.getElementById('use-api-forecast').checked && apiUrl;
            if (useApi) {
                CONFIG.FORECAST_API_URL = apiUrl;
                localStorage.setItem('api_url', apiUrl);
                fill.style.width = '35%';
                text.textContent = 'Calling forecast API (Prophet/SARIMA)...';

                const forecastDayCount = Math.ceil((new Date(forecastEnd) - new Date(forecastStart)) / (1000 * 60 * 60 * 24));
                await SalesForecastEngine.fetchApiForecasts(
                    salesData.records, venueData.venues, forecastDayCount,
                    (done, total) => {
                        fill.style.width = `${35 + (done / total) * 25}%`;
                        text.textContent = `API forecast: ${done}/${total} venues...`;
                    }
                );
            }

            fill.style.width = '65%';
            text.textContent = 'Building P&L forecast...';

            if (priorPnl) PnlBuilder.setPriorPnl(priorPnl.records);

            await new Promise(resolve => setTimeout(resolve, 10));

            PnlBuilder.run({
                venueDetails: venueData.venues,
                salesHistory: salesData.records,
                avgTicketData: avgTicketData.records,
                rampUpData: venueData.rampUp || {},
                labourAssumptions: labourData.records,
                cogsAssumptions: cogsData.records,
                rentAssumptions: rentData.records,
                forecastStart,
                forecastEnd
            });

            fill.style.width = '85%';
            text.textContent = 'Rendering...';

            await new Promise(resolve => setTimeout(resolve, 10));

            this.budgetReady = true;
            this.enableExports();
            this.updateKPIs();

            const runName = document.getElementById('run-name').value || 'Budget';
            document.getElementById('run-name-display').textContent = runName;
            document.getElementById('calc-status').textContent =
                `${PnlBuilder.dailyResults.length.toLocaleString()} daily forecasts | ${PnlBuilder.monthlySummary.length.toLocaleString()} monthly rows`;

            fill.style.width = '100%';
            text.textContent = `Done! ${PnlBuilder.dailyResults.length.toLocaleString()} daily rows generated`;

        } catch (err) {
            text.textContent = `Error: ${err.message}`;
            console.error(err);
        }

        btn.disabled = false;
    },

    enableExports() {
        document.getElementById('btn-export-daily').disabled = false;
        document.getElementById('btn-export-monthly').disabled = false;
        document.getElementById('btn-export-pnl').disabled = false;
        document.getElementById('btn-export-variance').disabled = false;
    },

    updateKPIs() {
        const kpis = PnlBuilder.getNetworkKPIs();
        document.getElementById('kpi-sales').textContent =
            new Intl.NumberFormat('en-AU', { style: 'currency', currency: 'AUD', maximumFractionDigits: 0 }).format(kpis.sales);
        document.getElementById('kpi-gp').textContent = kpis.gpPct.toFixed(1) + '%';
        document.getElementById('kpi-labour').textContent = kpis.labourPct.toFixed(1) + '%';
        document.getElementById('kpi-occupancy').textContent = kpis.occupancyPct.toFixed(1) + '%';
        document.getElementById('kpi-contribution').textContent = kpis.contributionPct.toFixed(1) + '%';
    },

    populateVenueFilters(venues) {
        const selects = [
            document.getElementById('forecast-venue-filter'),
            document.getElementById('pnl-venue-filter')
        ];

        for (const select of selects) {
            const current = select.value;
            select.innerHTML = '<option value="__all__">All Venues (Network)</option>';
            for (const v of venues.sort((a, b) => a.venue_name.localeCompare(b.venue_name))) {
                const opt = document.createElement('option');
                opt.value = v.venue_key;
                opt.textContent = `${v.venue_name} (${v.state})`;
                select.appendChild(opt);
            }
            if (current) select.value = current;
        }
    },

    renderVenueTable(venues) {
        const tbody = document.querySelector('#venue-table tbody');
        tbody.innerHTML = '';

        const now = new Date();
        for (const v of venues) {
            const openDate = new Date(v.opening_date);
            const monthsOpen = Math.max(0,
                (now.getFullYear() - openDate.getFullYear()) * 12 + (now.getMonth() - openDate.getMonth())
            );
            const isMature = monthsOpen >= CONFIG.RAMP_UP_MONTHS;

            const rampUp = ExcelParser.uploads.venue_details?.rampUp?.[v.venue_key];
            let currentRamp = isMature ? 1.0 : (rampUp && monthsOpen < rampUp.length ? rampUp[monthsOpen] : null);

            const salesData = ExcelParser.uploads.sales_history?.records?.filter(s => s.venue_key === v.venue_key) || [];
            const last90 = salesData.filter(s => {
                const d = new Date(s.sale_date);
                const diff = (now - d) / (1000 * 60 * 60 * 24);
                return diff <= 90;
            });
            const avgDaily = last90.length > 0
                ? last90.reduce((sum, s) => sum + s.gross_sales, 0) / last90.length
                : null;

            const tr = document.createElement('tr');
            tr.dataset.state = v.state;
            tr.dataset.maturity = isMature ? 'mature' : 'ramping';
            tr.innerHTML = `
                <td>${v.venue_name}</td>
                <td>${v.state}</td>
                <td>${v.opening_date}</td>
                <td class="number">${monthsOpen}</td>
                <td>${isMature ? '<span style="color:var(--success)">Mature</span>' : '<span style="color:var(--warning)">Ramping</span>'}</td>
                <td class="number">${currentRamp != null ? (currentRamp * 100).toFixed(0) + '%' : '—'}</td>
                <td class="number">${avgDaily != null ? '$' + Math.round(avgDaily).toLocaleString() : '—'}</td>
            `;
            tbody.appendChild(tr);
        }
    },

    filterVenueTable() {
        const state = document.getElementById('filter-state').value;
        const maturity = document.getElementById('filter-maturity').value;
        document.querySelectorAll('#venue-table tbody tr').forEach(tr => {
            const matchState = !state || tr.dataset.state === state;
            const matchMat = !maturity || tr.dataset.maturity === maturity;
            tr.style.display = matchState && matchMat ? '' : 'none';
        });
    },

    renderPnlTable() {
        const venueKey = document.getElementById('pnl-venue-filter').value;
        const view = document.getElementById('pnl-view').value;
        const pnl = PnlBuilder.getPnlTable(venueKey, view);

        const thead = document.querySelector('#pnl-table thead');
        const tbody = document.querySelector('#pnl-table tbody');

        let headerHtml = '<tr><th>Line Item</th>';
        for (const m of pnl.months) headerHtml += `<th class="number">${m.label}</th>`;
        headerHtml += '<th class="number">Total</th></tr>';
        thead.innerHTML = headerHtml;

        tbody.innerHTML = '';
        const fmt = (v) => {
            if (Math.abs(v) >= 1000000) return '$' + (v / 1000000).toFixed(1) + 'M';
            if (Math.abs(v) >= 1000) return '$' + (v / 1000).toFixed(0) + 'K';
            return '$' + Math.round(v).toLocaleString();
        };

        for (const row of pnl.rows) {
            const tr = document.createElement('tr');
            if (row.type === 'subtotal' || row.type === 'total') tr.className = 'line-item-subtotal';
            if (row.type === 'revenue') tr.className = 'line-item-header';

            let html = `<td>${row.label}</td>`;
            for (const m of pnl.months) {
                const val = row.values[m.key] || 0;
                html += `<td class="number">${fmt(val)}</td>`;
            }
            html += `<td class="number" style="font-weight:600">${fmt(row.total)}</td>`;
            tr.innerHTML = html;
            tbody.appendChild(tr);
        }

        if (ExcelParser.uploads.prior_pnl) {
            document.querySelector('.pnl-comparison').style.display = 'block';
            this.renderVarianceTable(venueKey);
        }
    },

    renderVarianceTable(venueKey) {
        const variance = PnlBuilder.getVarianceTable(venueKey);
        const thead = document.querySelector('#variance-table thead');
        const tbody = document.querySelector('#variance-table tbody');

        thead.innerHTML = '<tr><th>Line Item</th><th class="number">Budget</th><th class="number">Prior Year</th><th class="number">Variance $</th><th class="number">Variance %</th></tr>';
        tbody.innerHTML = '';

        const fmt = (v) => '$' + Math.round(v).toLocaleString();

        for (const row of variance) {
            const tr = document.createElement('tr');
            if (row.type === 'subtotal' || row.type === 'total') tr.className = 'line-item-subtotal';

            const varClass = row.variance >= 0 ? 'positive' : 'negative';
            tr.innerHTML = `
                <td>${row.label}</td>
                <td class="number">${fmt(row.budget)}</td>
                <td class="number">${fmt(row.prior)}</td>
                <td class="number ${varClass}">${fmt(row.variance)}</td>
                <td class="number ${varClass}">${row.variancePct.toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        }
    },

    renderDashboard() {
        this.updateKPIs();
        Charts.renderAllDashboardCharts();
    },

    showPreview(key, result) {
        const container = document.getElementById('upload-preview');
        const content = document.getElementById('preview-content');
        container.style.display = 'block';

        const records = result.records || result.venues || [];
        const sample = records.slice(0, 10);

        if (sample.length === 0) {
            content.innerHTML = '<p>No records parsed</p>';
            return;
        }

        const keys = Object.keys(sample[0]).filter(k => k !== 'venue_key');
        let html = `<p><strong>${key}</strong>: ${records.length} records (showing first 10)</p>`;
        html += '<table><thead><tr>';
        for (const k of keys) html += `<th>${k}</th>`;
        html += '</tr></thead><tbody>';

        for (const row of sample) {
            html += '<tr>';
            for (const k of keys) {
                const v = row[k];
                html += `<td>${v != null ? v : ''}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        content.innerHTML = html;
    }
};

document.addEventListener('DOMContentLoaded', () => App.init());
