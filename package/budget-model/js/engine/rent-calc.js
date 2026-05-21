const RentCalcEngine = {
    calculate(dailyForecasts, rentAssumptions) {
        const rentMap = {};
        for (const r of rentAssumptions) {
            rentMap[r.venue_key] = r;
        }

        const monthlySalesByVenue = {};
        for (const f of dailyForecasts) {
            const monthKey = `${f.venue_key}_${f.forecast_date.substring(0, 7)}`;
            if (!monthlySalesByVenue[monthKey]) monthlySalesByVenue[monthKey] = 0;
            monthlySalesByVenue[monthKey] += f.net_sales;
        }

        for (const forecast of dailyForecasts) {
            const rent = rentMap[forecast.venue_key] ||
                (forecast.similar_venue_key ? rentMap[forecast.similar_venue_key] : null);
            if (!rent) {
                forecast.rent_base = 0;
                forecast.rent_outgoings = 0;
                forecast.rent_percentage = 0;
                forecast.rent_marketing_levy = 0;
                forecast.occupancy_total = 0;
                continue;
            }

            const daysInMonth = new Date(
                parseInt(forecast.forecast_date.substring(0, 4)),
                parseInt(forecast.forecast_date.substring(5, 7)),
                0
            ).getDate();

            const baseRentDaily = rent.base_rent_monthly / daysInMonth;
            const outgoingsDaily = rent.outgoings_monthly / daysInMonth;

            let pctRentDaily = 0;
            if (rent.pct_rent_threshold > 0 && rent.pct_rent_rate > 0) {
                const monthKey = `${forecast.venue_key}_${forecast.forecast_date.substring(0, 7)}`;
                const monthlySales = monthlySalesByVenue[monthKey] || 0;
                if (monthlySales > rent.pct_rent_threshold) {
                    const monthlyPctRent = (monthlySales - rent.pct_rent_threshold) * rent.pct_rent_rate;
                    pctRentDaily = monthlyPctRent / daysInMonth;
                }
            }

            const marketingLevy = forecast.net_sales * (rent.marketing_levy_pct || 0);

            forecast.rent_base = Math.round(baseRentDaily * 100) / 100;
            forecast.rent_outgoings = Math.round(outgoingsDaily * 100) / 100;
            forecast.rent_percentage = Math.round(pctRentDaily * 100) / 100;
            forecast.rent_marketing_levy = Math.round(marketingLevy * 100) / 100;
            forecast.occupancy_total = Math.round(
                (baseRentDaily + outgoingsDaily + pctRentDaily + marketingLevy) * 100
            ) / 100;
        }

        return dailyForecasts;
    }
};
