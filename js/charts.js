const Charts = {
    instances: {},

    destroy(id) {
        if (this.instances[id]) {
            this.instances[id].destroy();
            delete this.instances[id];
        }
    },

    formatCurrency(val) {
        return new Intl.NumberFormat('en-AU', { style: 'currency', currency: 'AUD', maximumFractionDigits: 0 }).format(val);
    },

    renderDowChart(venueKey) {
        this.destroy('chart-dow');
        const data = SalesForecastEngine.getSeasonalityData(venueKey);
        const ctx = document.getElementById('chart-dow').getContext('2d');
        this.instances['chart-dow'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: CONFIG.DAYS_OF_WEEK,
                datasets: [{
                    label: 'Index (1.0 = average)',
                    data: data.dow,
                    backgroundColor: data.dow.map(v => v >= 1 ? 'rgba(99,102,241,0.7)' : 'rgba(239,68,68,0.5)'),
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Index' } }
                }
            }
        });
    },

    renderMonthlySeasonChart(venueKey) {
        this.destroy('chart-monthly-season');
        const data = SalesForecastEngine.getSeasonalityData(venueKey);
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const ctx = document.getElementById('chart-monthly-season').getContext('2d');
        this.instances['chart-monthly-season'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: months,
                datasets: [{
                    label: 'Monthly Seasonality Index',
                    data: data.month,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99,102,241,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { title: { display: true, text: 'Index' } }
                }
            }
        });
    },

    renderDailyForecastChart(venueKey) {
        this.destroy('chart-daily-forecast');
        const forecasts = venueKey === '__all__'
            ? SalesForecastEngine.dailyForecasts
            : SalesForecastEngine.dailyForecasts.filter(f => f.venue_key === venueKey);

        const dailyTotals = {};
        for (const f of forecasts) {
            if (!dailyTotals[f.forecast_date]) dailyTotals[f.forecast_date] = 0;
            dailyTotals[f.forecast_date] += f.net_sales;
        }

        const sorted = Object.entries(dailyTotals).sort(([a], [b]) => a.localeCompare(b));
        const step = Math.max(1, Math.floor(sorted.length / 60));
        const sampled = sorted.filter((_, i) => i % step === 0);

        const ctx = document.getElementById('chart-daily-forecast').getContext('2d');
        this.instances['chart-daily-forecast'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: sampled.map(([d]) => d),
                datasets: [{
                    label: 'Daily Net Sales',
                    data: sampled.map(([, v]) => v),
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99,102,241,0.05)',
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { maxTicksLimit: 12, maxRotation: 45 } },
                    y: { title: { display: true, text: 'AUD' }, ticks: { callback: v => this.formatCurrency(v) } }
                }
            }
        });
    },

    renderWeatherCorrelation() {
        this.destroy('chart-weather-corr');
        const indices = WeatherEngine.tempBandIndices;
        const datasets = [];
        const colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
        let i = 0;

        for (const [state, bands] of Object.entries(indices)) {
            const data = CONFIG.TEMP_BANDS.map(b => bands[b.label] || 1.0);
            datasets.push({
                label: state,
                data,
                borderColor: colors[i % colors.length],
                pointRadius: 3,
                borderWidth: 2,
                fill: false
            });
            i++;
        }

        const ctx = document.getElementById('chart-weather-corr').getContext('2d');
        this.instances['chart-weather-corr'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: CONFIG.TEMP_BANDS.map(b => b.label),
                datasets
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } } },
                scales: {
                    y: { title: { display: true, text: 'Sales Index' } },
                    x: { title: { display: true, text: 'Max Temperature Band' } }
                }
            }
        });
    },

    renderForecastMonthlyChart(venueKey) {
        this.destroy('chart-forecast-monthly');
        const data = SalesForecastEngine.getMonthlySalesForecasts(venueKey);
        const ctx = document.getElementById('chart-forecast-monthly').getContext('2d');
        this.instances['chart-forecast-monthly'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.map(d => d.month),
                datasets: [{
                    label: 'Monthly Sales',
                    data: data.map(d => d.sales),
                    backgroundColor: 'rgba(99,102,241,0.7)',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { ticks: { callback: v => this.formatCurrency(v) } }
                }
            }
        });
    },

    renderSalesComparisonChart() {
        this.destroy('chart-sales-comparison');
        const budgetMonthly = SalesForecastEngine.getMonthlySalesForecasts('__all__');
        const priorTotals = PnlBuilder.priorPnlByVenue;

        const priorByMonth = {};
        for (const venueData of Object.values(priorTotals)) {
            for (const [month, items] of Object.entries(venueData)) {
                const sales = items['Net Sales'] || items['net_sales'] || 0;
                if (!priorByMonth[month]) priorByMonth[month] = 0;
                priorByMonth[month] += sales;
            }
        }

        const ctx = document.getElementById('chart-sales-comparison').getContext('2d');
        this.instances['chart-sales-comparison'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: budgetMonthly.map(d => d.month),
                datasets: [
                    {
                        label: 'Budget',
                        data: budgetMonthly.map(d => d.sales),
                        backgroundColor: 'rgba(99,102,241,0.7)',
                        borderRadius: 4
                    },
                    {
                        label: 'Prior Year',
                        data: budgetMonthly.map(d => priorByMonth[d.month] || 0),
                        backgroundColor: 'rgba(148,163,184,0.5)',
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top' } },
                scales: {
                    y: { ticks: { callback: v => this.formatCurrency(v) } }
                }
            }
        });
    },

    renderCostStructureChart() {
        this.destroy('chart-cost-structure');
        const kpis = PnlBuilder.getNetworkKPIs();
        const cogsPct = 100 - kpis.gpPct;
        const ctx = document.getElementById('chart-cost-structure').getContext('2d');
        this.instances['chart-cost-structure'] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['COGS', 'Labour', 'Occupancy', 'Contribution'],
                datasets: [{
                    data: [cogsPct, kpis.labourPct, kpis.occupancyPct, kpis.contributionPct],
                    backgroundColor: ['#ef4444', '#f59e0b', '#8b5cf6', '#10b981']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 12 } },
                    tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.parsed.toFixed(1)}%` } }
                }
            }
        });
    },

    renderStateContributionChart() {
        this.destroy('chart-state-contribution');
        const data = PnlBuilder.getContributionByState();
        const ctx = document.getElementById('chart-state-contribution').getContext('2d');
        this.instances['chart-state-contribution'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.map(d => d.state),
                datasets: [
                    {
                        label: 'Sales',
                        data: data.map(d => d.sales),
                        backgroundColor: 'rgba(99,102,241,0.7)',
                        borderRadius: 4
                    },
                    {
                        label: 'Contribution',
                        data: data.map(d => d.contribution),
                        backgroundColor: 'rgba(16,185,129,0.7)',
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top' } },
                scales: {
                    y: { ticks: { callback: v => this.formatCurrency(v) } }
                }
            }
        });
    },

    renderVenueHeatmap() {
        const container = document.getElementById('venue-heatmap');
        const data = PnlBuilder.monthlySummary;
        if (!data.length) { container.innerHTML = '<p>No data</p>'; return; }

        const venues = [...new Set(data.map(d => d.venue_name))].sort();
        const months = [...new Set(data.map(d => d.budget_month))].sort();

        const lookup = {};
        for (const d of data) {
            lookup[`${d.venue_name}_${d.budget_month}`] = d;
        }

        let html = '<table><thead><tr><th>Venue</th>';
        for (const m of months) {
            html += `<th>${m.substring(0, 7)}</th>`;
        }
        html += '</tr></thead><tbody>';

        for (const venue of venues) {
            html += `<tr><td style="text-align:left;white-space:nowrap">${venue}</td>`;
            for (const month of months) {
                const d = lookup[`${venue}_${month}`];
                if (d && d.net_sales > 0) {
                    const pct = (d.venue_contribution / d.net_sales) * 100;
                    const color = pct > 20 ? '#10b981' : pct > 10 ? '#f59e0b' : pct > 0 ? '#fb923c' : '#ef4444';
                    const opacity = Math.min(1, Math.max(0.2, Math.abs(pct) / 30));
                    html += `<td style="background:${color}${Math.round(opacity * 255).toString(16).padStart(2, '0')};color:#1e293b">${pct.toFixed(0)}%</td>`;
                } else {
                    html += '<td>—</td>';
                }
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        container.innerHTML = html;
    },

    renderAllSalesCharts(venueKey) {
        this.renderDowChart(venueKey);
        this.renderMonthlySeasonChart(venueKey);
        this.renderDailyForecastChart(venueKey);
        this.renderWeatherCorrelation();
        this.renderForecastMonthlyChart(venueKey);
    },

    renderAllDashboardCharts() {
        this.renderSalesComparisonChart();
        this.renderCostStructureChart();
        this.renderStateContributionChart();
        this.renderVenueHeatmap();
    }
};
