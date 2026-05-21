const SalesForecastEngine = {
    seasonalityIndices: {},
    baseDailySales: {},
    dailyForecasts: [],
    apiForecasts: {},
    useApi: false,

    buildSeasonality(salesHistory, venueDetails) {
        const venueMap = {};
        for (const v of venueDetails) venueMap[v.venue_key] = v;

        const grouped = {};
        for (const s of salesHistory) {
            if (!grouped[s.venue_key]) grouped[s.venue_key] = [];
            grouped[s.venue_key].push(s);
        }

        this.seasonalityIndices = {};
        this.baseDailySales = {};

        for (const [venueKey, sales] of Object.entries(grouped)) {
            const venue = venueMap[venueKey];
            if (!venue) continue;

            const totalSales = sales.reduce((sum, s) => sum + s.gross_sales, 0);
            const avgDaily = totalSales / sales.length;
            this.baseDailySales[venueKey] = avgDaily;

            const dowBuckets = Array.from({ length: 7 }, () => ({ total: 0, count: 0 }));
            const monthBuckets = Array.from({ length: 12 }, () => ({ total: 0, count: 0 }));
            const holidayBucket = { total: 0, count: 0 };
            const schoolBucket = { total: 0, count: 0 };
            const normalBucket = { total: 0, count: 0 };

            for (const s of sales) {
                const d = new Date(s.sale_date);
                const dow = (d.getDay() + 6) % 7;
                const month = d.getMonth();

                dowBuckets[dow].total += s.gross_sales;
                dowBuckets[dow].count++;
                monthBuckets[month].total += s.gross_sales;
                monthBuckets[month].count++;

                if (CALENDARS.isPublicHoliday(s.sale_date, venue.state)) {
                    holidayBucket.total += s.gross_sales;
                    holidayBucket.count++;
                } else if (CALENDARS.isSchoolHoliday(s.sale_date, venue.state)) {
                    schoolBucket.total += s.gross_sales;
                    schoolBucket.count++;
                } else {
                    normalBucket.total += s.gross_sales;
                    normalBucket.count++;
                }
            }

            const dowIndices = dowBuckets.map(b =>
                b.count > 0 ? (b.total / b.count) / avgDaily : 1.0
            );

            const monthIndices = monthBuckets.map(b =>
                b.count > 0 ? (b.total / b.count) / avgDaily : 1.0
            );

            const normalAvg = normalBucket.count > 0 ? normalBucket.total / normalBucket.count : avgDaily;
            const holidayIndex = holidayBucket.count > 0 ? (holidayBucket.total / holidayBucket.count) / normalAvg : 1.15;
            const schoolIndex = schoolBucket.count > 0 ? (schoolBucket.total / schoolBucket.count) / normalAvg : 1.10;

            this.seasonalityIndices[venueKey] = {
                dow: dowIndices,
                month: monthIndices,
                publicHoliday: holidayIndex,
                schoolHoliday: schoolIndex
            };
        }
    },

    // Call your existing venu-cast-api /forecast-multi endpoint
    async fetchApiForecasts(salesHistory, venueDetails, forecastDays, onProgress) {
        const apiUrl = CONFIG.FORECAST_API_URL;
        if (!apiUrl) {
            console.warn('No forecast API URL configured — using local seasonality model');
            this.useApi = false;
            return;
        }

        const grouped = {};
        for (const s of salesHistory) {
            if (!grouped[s.venue_key]) grouped[s.venue_key] = [];
            grouped[s.venue_key].push(s);
        }

        const venueNames = Object.keys(grouped);
        const batches = [];
        for (let i = 0; i < venueNames.length; i += CONFIG.FORECAST_BATCH_SIZE) {
            batches.push(venueNames.slice(i, i + CONFIG.FORECAST_BATCH_SIZE));
        }

        this.apiForecasts = {};
        let done = 0;

        for (const batch of batches) {
            const venues = {};
            for (const venueKey of batch) {
                const sales = grouped[venueKey].sort((a, b) => a.sale_date.localeCompare(b.sale_date));
                venues[venueKey] = {
                    dates: sales.map(s => s.sale_date),
                    values: sales.map(s => s.gross_sales),
                    forecast_days: forecastDays
                };
            }

            try {
                const resp = await fetch(`${apiUrl}/forecast-multi`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ venues })
                });

                if (!resp.ok) throw new Error(`API returned ${resp.status}`);
                const result = await resp.json();

                for (const [venueKey, forecast] of Object.entries(result)) {
                    if (forecast.forecast_dates && forecast.forecast_values) {
                        this.apiForecasts[venueKey] = {};
                        for (let i = 0; i < forecast.forecast_dates.length; i++) {
                            this.apiForecasts[venueKey][forecast.forecast_dates[i]] =
                                forecast.forecast_values[i];
                        }
                    }
                }
            } catch (err) {
                console.warn(`API batch forecast failed:`, err);
            }

            done += batch.length;
            if (onProgress) onProgress(done, venueNames.length);
        }

        this.useApi = Object.keys(this.apiForecasts).length > 0;
    },

    getRampUpMultiplier(venue, dateStr, rampUpData) {
        const openDate = new Date(venue.opening_date);
        const forecastDate = new Date(dateStr);

        if (forecastDate < openDate) return 0;

        const monthsOpen = (forecastDate.getFullYear() - openDate.getFullYear()) * 12 +
            (forecastDate.getMonth() - openDate.getMonth());

        if (monthsOpen >= CONFIG.RAMP_UP_MONTHS) return 1.0;

        const venueRamp = rampUpData[venue.venue_key];
        if (venueRamp && monthsOpen < venueRamp.length) {
            return venueRamp[monthsOpen];
        }

        return Math.min(1.0, 0.4 + (monthsOpen * 0.6 / CONFIG.RAMP_UP_MONTHS));
    },

    generateForecasts(venueDetails, avgTicketData, rampUpData, forecastStart, forecastEnd) {
        this.dailyForecasts = [];

        const venueMap = {};
        for (const v of venueDetails) venueMap[v.venue_key] = v;

        const ticketMap = {};
        for (const t of avgTicketData) {
            const key = `${t.venue_key}_${t.budget_month}`;
            ticketMap[key] = t.avg_ticket;
        }

        const start = new Date(forecastStart);
        const end = new Date(forecastEnd);

        for (const venue of venueDetails) {
            if (!venue.is_active) continue;

            const seasonality = this.seasonalityIndices[venue.venue_key];
            const baseDaily = this.baseDailySales[venue.venue_key] || 0;
            const apiData = this.apiForecasts[venue.venue_key];

            if (baseDaily === 0 && new Date(venue.opening_date) > end) continue;

            const current = new Date(start);
            while (current <= end) {
                const dateStr = current.toISOString().substring(0, 10);
                const dow = (current.getDay() + 6) % 7;
                const month = current.getMonth();
                const monthNum = current.getMonth() + 1;
                const monthKey = `${current.getFullYear()}-${String(monthNum).padStart(2, '0')}-01`;

                const rampUp = this.getRampUpMultiplier(venue, dateStr, rampUpData);

                if (rampUp === 0) {
                    current.setDate(current.getDate() + 1);
                    continue;
                }

                let forecastSales;

                // Prefer API forecast (Prophet/SARIMA) if available
                if (this.useApi && apiData && apiData[dateStr] != null) {
                    forecastSales = apiData[dateStr] * rampUp;
                } else {
                    // Fallback: local seasonality model
                    let dowIndex = 1.0;
                    let monthIndex = 1.0;
                    let holidayAdj = 1.0;

                    if (seasonality) {
                        dowIndex = seasonality.dow[dow] || 1.0;
                        monthIndex = seasonality.month[month] || 1.0;

                        if (CALENDARS.isPublicHoliday(dateStr, venue.state)) {
                            holidayAdj = seasonality.publicHoliday;
                        } else if (CALENDARS.isSchoolHoliday(dateStr, venue.state)) {
                            holidayAdj = seasonality.schoolHoliday;
                        }
                    }

                    const weatherIndex = WeatherEngine.getWeatherIndex(venue.state, monthNum);

                    if (baseDaily > 0) {
                        forecastSales = baseDaily * dowIndex * monthIndex * holidayAdj * weatherIndex * rampUp;
                    } else {
                        const networkAvg = this.getNetworkAvgDaily();
                        forecastSales = networkAvg * dowIndex * monthIndex * holidayAdj * weatherIndex * rampUp;
                    }
                }

                forecastSales = Math.max(0, Math.round(forecastSales * 100) / 100);

                const ticketKey = `${venue.venue_key}_${monthKey}`;
                const avgTicket = ticketMap[ticketKey] || this.getDefaultTicket(venue.venue_key, avgTicketData);
                const transactions = avgTicket > 0 ? Math.round(forecastSales / avgTicket) : 0;

                this.dailyForecasts.push({
                    venue_key: venue.venue_key,
                    venue_name: venue.venue_name,
                    state: venue.state,
                    forecast_date: dateStr,
                    gross_sales: forecastSales,
                    net_sales: forecastSales,
                    forecast_transactions: transactions,
                    avg_ticket: avgTicket,
                    ramp_up_multiplier: rampUp,
                    source: (this.useApi && apiData && apiData[dateStr] != null) ? 'api' : 'local'
                });

                current.setDate(current.getDate() + 1);
            }
        }

        return this.dailyForecasts;
    },

    getNetworkAvgDaily() {
        const values = Object.values(this.baseDailySales).filter(v => v > 0);
        if (values.length === 0) return 2000;
        return values.reduce((a, b) => a + b, 0) / values.length;
    },

    getDefaultTicket(venueKey, avgTicketData) {
        const venueTickets = avgTicketData.filter(t => t.venue_key === venueKey);
        if (venueTickets.length > 0) {
            return venueTickets.reduce((sum, t) => sum + t.avg_ticket, 0) / venueTickets.length;
        }
        return 12.50;
    },

    getSeasonalityData(venueKey) {
        if (venueKey === '__all__') {
            const allDow = Array(7).fill(0);
            const allMonth = Array(12).fill(0);
            let count = 0;
            for (const s of Object.values(this.seasonalityIndices)) {
                for (let i = 0; i < 7; i++) allDow[i] += s.dow[i];
                for (let i = 0; i < 12; i++) allMonth[i] += s.month[i];
                count++;
            }
            if (count > 0) {
                return {
                    dow: allDow.map(v => v / count),
                    month: allMonth.map(v => v / count)
                };
            }
        }
        return this.seasonalityIndices[venueKey] || { dow: Array(7).fill(1), month: Array(12).fill(1) };
    },

    getMonthlySalesForecasts(venueKey) {
        const forecasts = venueKey === '__all__'
            ? this.dailyForecasts
            : this.dailyForecasts.filter(f => f.venue_key === venueKey);

        const monthly = {};
        for (const f of forecasts) {
            const monthKey = f.forecast_date.substring(0, 7);
            if (!monthly[monthKey]) monthly[monthKey] = { sales: 0, transactions: 0, days: 0 };
            monthly[monthKey].sales += f.net_sales;
            monthly[monthKey].transactions += f.forecast_transactions;
            monthly[monthKey].days++;
        }

        return Object.entries(monthly)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([month, data]) => ({ month, ...data }));
    }
};
