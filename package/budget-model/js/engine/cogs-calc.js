const CogsCalcEngine = {
    calculate(dailyForecasts, cogsAssumptions) {
        const cogsMap = {};
        for (const c of cogsAssumptions) {
            if (!cogsMap[c.venue_key]) cogsMap[c.venue_key] = {};
            cogsMap[c.venue_key][c.category] = c.cogs_pct;
        }

        for (const forecast of dailyForecasts) {
            const cogs = cogsMap[forecast.venue_key] ||
                (forecast.similar_venue_key ? cogsMap[forecast.similar_venue_key] : null) ||
                {};

            forecast.cogs_food = Math.round(forecast.net_sales * (cogs.food || 0) * 100) / 100;
            forecast.cogs_packaging = Math.round(forecast.net_sales * (cogs.packaging || 0) * 100) / 100;
            forecast.cogs_retail = Math.round(forecast.net_sales * (cogs.retail || 0) * 100) / 100;
            forecast.cogs_discounts = Math.round(forecast.gross_sales * (cogs.sale_discounts || 0) * 100) / 100;
            forecast.cogs_total = Math.round(
                (forecast.cogs_food + forecast.cogs_packaging + forecast.cogs_retail + forecast.cogs_discounts) * 100
            ) / 100;
            forecast.gross_profit = Math.round((forecast.net_sales - forecast.cogs_total) * 100) / 100;
        }

        return dailyForecasts;
    }
};
