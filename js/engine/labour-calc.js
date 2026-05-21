const LabourCalcEngine = {
    calculate(dailyForecasts, labourAssumptions) {
        const labourMap = {};
        for (const l of labourAssumptions) {
            labourMap[l.venue_key] = l;
        }

        for (const forecast of dailyForecasts) {
            const labour = labourMap[forecast.venue_key];
            if (!labour) {
                forecast.crew_labour_hours = 0;
                forecast.crew_labour_cost = 0;
                forecast.crew_oncosts = 0;
                forecast.mgmt_labour_cost = 0;
                forecast.mgmt_oncosts = 0;
                forecast.labour_total = 0;
                continue;
            }

            const daysInMonth = new Date(
                parseInt(forecast.forecast_date.substring(0, 4)),
                parseInt(forecast.forecast_date.substring(5, 7)),
                0
            ).getDate();

            const crewHours = labour.sales_per_labour_hr > 0
                ? forecast.net_sales / labour.sales_per_labour_hr
                : 0;

            const crewCost = crewHours * labour.avg_hourly_rate;
            const crewOncosts = crewCost * labour.oncosts_pct;
            const mgmtDaily = labour.mgmt_salary_monthly / daysInMonth;
            const mgmtOncosts = mgmtDaily * labour.mgmt_oncosts_pct;

            forecast.crew_labour_hours = Math.round(crewHours * 100) / 100;
            forecast.crew_labour_cost = Math.round(crewCost * 100) / 100;
            forecast.crew_oncosts = Math.round(crewOncosts * 100) / 100;
            forecast.mgmt_labour_cost = Math.round(mgmtDaily * 100) / 100;
            forecast.mgmt_oncosts = Math.round(mgmtOncosts * 100) / 100;
            forecast.labour_total = Math.round((crewCost + crewOncosts + mgmtDaily + mgmtOncosts) * 100) / 100;
        }

        return dailyForecasts;
    }
};
