const WeatherEngine = {
    weatherData: {},
    tempBandIndices: {},

    async fetchHistoricalWeather(states, startDate, endDate, onProgress) {
        const results = {};
        let done = 0;

        for (const state of states) {
            const coords = CONFIG.STATE_COORDS[state];
            if (!coords) continue;

            const url = `${CONFIG.OPEN_METEO_ARCHIVE_URL}?latitude=${coords.lat}&longitude=${coords.lon}` +
                `&start_date=${startDate}&end_date=${endDate}` +
                `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,sunshine_duration` +
                `&timezone=Australia%2FSydney`;

            try {
                const resp = await fetch(url);
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const json = await resp.json();

                const daily = json.daily;
                results[state] = [];
                for (let i = 0; i < daily.time.length; i++) {
                    results[state].push({
                        state,
                        observation_date: daily.time[i],
                        max_temp_c: daily.temperature_2m_max[i],
                        min_temp_c: daily.temperature_2m_min[i],
                        rainfall_mm: daily.precipitation_sum[i],
                        sunshine_hours: daily.sunshine_duration ? daily.sunshine_duration[i] / 3600 : null
                    });
                }
            } catch (err) {
                console.warn(`Weather fetch failed for ${state}:`, err);
                results[state] = [];
            }

            done++;
            if (onProgress) onProgress(done, states.length);
        }

        this.weatherData = results;
        return results;
    },

    getWeatherForDate(state, dateStr) {
        const stateData = this.weatherData[state];
        if (!stateData) return null;
        return stateData.find(w => w.observation_date === dateStr) || null;
    },

    buildTempBandIndices(salesHistory, venueDetails) {
        const venueStateMap = {};
        for (const v of venueDetails) {
            venueStateMap[v.venue_key] = v.state;
        }

        const bandSales = {};
        for (const state of CONFIG.STATES) {
            bandSales[state] = {};
            for (const band of CONFIG.TEMP_BANDS) {
                bandSales[state][band.label] = { totalSales: 0, count: 0 };
            }
        }

        for (const sale of salesHistory) {
            const state = venueStateMap[sale.venue_key];
            if (!state) continue;
            const weather = this.getWeatherForDate(state, sale.sale_date);
            if (!weather || weather.max_temp_c == null) continue;

            const band = this.getTempBand(weather.max_temp_c);
            if (band && bandSales[state][band]) {
                bandSales[state][band].totalSales += sale.gross_sales;
                bandSales[state][band].count++;
            }
        }

        this.tempBandIndices = {};
        for (const state of CONFIG.STATES) {
            let totalSales = 0;
            let totalCount = 0;
            for (const band of CONFIG.TEMP_BANDS) {
                totalSales += bandSales[state][band.label].totalSales;
                totalCount += bandSales[state][band.label].count;
            }
            const avgSales = totalCount > 0 ? totalSales / totalCount : 1;

            this.tempBandIndices[state] = {};
            for (const band of CONFIG.TEMP_BANDS) {
                const bs = bandSales[state][band.label];
                if (bs.count > 0) {
                    this.tempBandIndices[state][band.label] = (bs.totalSales / bs.count) / avgSales;
                } else {
                    this.tempBandIndices[state][band.label] = 1.0;
                }
            }
        }

        return this.tempBandIndices;
    },

    getTempBand(temp) {
        for (const band of CONFIG.TEMP_BANDS) {
            if (temp >= band.min && temp < band.max) return band.label;
        }
        return null;
    },

    getExpectedTempForMonth(state, month) {
        const stateData = this.weatherData[state];
        if (!stateData || stateData.length === 0) return 25;

        const monthTemps = stateData
            .filter(w => {
                const m = parseInt(w.observation_date.substring(5, 7));
                return m === month && w.max_temp_c != null;
            })
            .map(w => w.max_temp_c);

        if (monthTemps.length === 0) return 25;
        return monthTemps.reduce((a, b) => a + b, 0) / monthTemps.length;
    },

    getWeatherIndex(state, month) {
        const expectedTemp = this.getExpectedTempForMonth(state, month);
        const band = this.getTempBand(expectedTemp);
        if (!band || !this.tempBandIndices[state]) return 1.0;
        return this.tempBandIndices[state][band] || 1.0;
    }
};
