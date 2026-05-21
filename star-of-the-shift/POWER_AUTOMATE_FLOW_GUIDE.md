# Star of the Shift Champion - Power Automate Flow Guide

## Overview
This flow runs daily at 8am, reads the SharePoint "Operations Update" list,
filters to the **current month only** (month-to-date), counts how many times
each person was named "Star of the shift", builds an HTML leaderboard, and
emails it to recipients listed in a CSV on SharePoint.

---

## Prerequisites
1. Upload `email_recipients.csv` to your SharePoint Documents library at:
   `Yochi Team Sharepoint > Documents > StarOfTheShift > email_recipients.csv`
2. Edit the CSV to contain the real recipient names and emails.

---

## Flow Steps (Build in Power Automate)

### Step 1: Trigger - Recurrence
- **Type:** Recurrence
- **Interval:** 1 Day
- **Frequency:** Day
- **At these hours:** 8
- **Time zone:** (UTC+10:00) Canberra, Melbourne, Sydney

---

### Step 2: Get Items from SharePoint (current month only)
- **Action:** SharePoint > Get items
- **Site Address:** `https://embraceyochi.sharepoint.com/sites/YochiTeamSharepoint`
- **List Name:** `Operations Update`
- **Top Count:** 5000
- **Filter Query:** `Date_x0020_today ge '@{formatDateTime(startOfMonth(utcNow()), 'yyyy-MM-ddTHH:mm:ssZ')}'`

This OData filter ensures only rows from the 1st of the current month onwards
are returned, giving you a month-to-date leaderboard.

---

### Step 3: Initialize Variables
Add these three "Initialize variable" actions:

1. **Name:** `varCountsJSON`  |  **Type:** String  |  **Value:** `{}`
2. **Name:** `varVenueJSON`   |  **Type:** String  |  **Value:** `{}`
3. **Name:** `varHTMLRows`    |  **Type:** String  |  **Value:** (empty)

---

### Step 4: Loop Through Items and Count Stars
**Action:** Apply to each (loop over the "value" from Get items)

Inside the loop, add a **Condition**:
- `items('Apply_to_each')?['Star_x0020_of_x0020_the_x0020_shift_x003f_']` **is not equal to** (empty)

> **Note:** The internal column name for "Star of the shift?" may differ.
> To find it: In your browser, go to the list, click the column header >
> Column settings > Edit. The URL will contain `&Field=` with the internal name.
> Common internal name patterns: `Star_x0020_of_x0020_the_x0020_shift_x003f_`

If Yes branch - add a **Compose** action:
- **Inputs:** Use this expression to split comma-separated names:

```
split(items('Apply_to_each')?['Star_x0020_of_x0020_the_x0020_shift_x003f_'], ',')
```

Then add another **Apply to each** over the Compose output (for multi-name entries):

Inside this inner loop, use **Compose** actions and **Set variable** to track counts.

---

### SIMPLER ALTERNATIVE: Use an Office Script

Because aggregation in Power Automate is verbose, the **recommended approach** is:

1. Add a **"Run script"** action (Excel Online > Run script)
2. Use the Office Script below which does ALL the aggregation + HTML generation
3. Power Automate just needs to pass the SharePoint items to the script and email the result

---

## Office Script (paste into Excel Online > Automate > New Script)

```typescript
function main(workbook: ExcelScript.Workbook, items: SharePointItem[]): string {
  const today = new Date();
  const currentMonth = today.getMonth();
  const currentYear = today.getFullYear();

  // Filter to current month only and count stars per person
  const counts: Record<string, { count: number; venue: string }> = {};
  let latestDate: Date | null = null;

  for (const item of items) {
    const starField = item.starOfShift;
    const venue = item.venue;
    const dateStr = item.dateToday;

    // Parse the date and filter to current month
    if (dateStr) {
      const itemDate = new Date(dateStr);
      if (itemDate.getMonth() !== currentMonth || itemDate.getFullYear() !== currentYear) {
        continue;
      }
      if (!latestDate || itemDate > latestDate) {
        latestDate = itemDate;
      }
    }

    if (!starField || starField.trim() === "") continue;

    // Split comma-separated names
    const names = starField.split(",").map(n => n.trim()).filter(n => n.length > 0);

    for (const name of names) {
      const key = name.toLowerCase();
      if (!counts[key]) {
        counts[key] = { count: 0, venue: venue || "Unknown" };
      }
      counts[key].count++;
      if (venue) counts[key].venue = venue;
    }
  }

  // Sort by count descending
  const sorted = Object.entries(counts)
    .map(([name, data]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      venue: data.venue,
      count: data.count
    }))
    .sort((a, b) => b.count - a.count);

  // Build date range string: "1 May - 7 May 2026 (MTD)"
  const monthName = today.toLocaleDateString("en-AU", { month: "long" });
  const lastDataDay = latestDate
    ? latestDate.toLocaleDateString("en-AU", { day: "numeric" })
    : today.toLocaleDateString("en-AU", { day: "numeric" });
  const dateStr = `1 ${monthName} - ${lastDataDay} ${monthName} ${currentYear} (MTD)`;

  const medals = ["&#129351;", "&#129352;", "&#129353;"];

  let rows = "";
  sorted.forEach((entry, i) => {
    const rank = i < 3 ? medals[i] : `${i + 1}`;
    const bgColor = i === 0 ? "#fff9e6" : (i % 2 === 0 ? "#ffffff" : "#f7f7f7");
    const nameWeight = i === 0 ? "700" : (i < 3 ? "600" : "normal");
    const nameColor = i === 0 ? "#1a1a2e" : "#333";
    const countSize = i < 3 ? "16px" : "14px";
    const borderBottom = i < sorted.length - 1
      ? "border-bottom: 1px solid #eee;" : "";

    rows += `<tr style="background-color: ${bgColor};">
      <td style="padding: 12px 16px; ${borderBottom} font-size: 18px;">${rank}</td>
      <td style="padding: 12px 16px; ${borderBottom} font-weight: ${nameWeight}; color: ${nameColor};">${entry.name}</td>
      <td style="padding: 12px 16px; ${borderBottom} color: #555;">${entry.venue}</td>
      <td style="padding: 12px 16px; ${borderBottom} text-align: center; font-weight: 700; color: ${nameColor}; font-size: ${countSize};">${entry.count}</td>
    </tr>`;
  });

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px;">
<div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
  <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 30px 20px; text-align: center;">
    <h1 style="color: #f5c518; margin: 0; font-size: 24px; letter-spacing: 1px;">&#11088; Star of the Shift Champion &#11088;</h1>
    <p style="color: #cccccc; margin: 8px 0 0 0; font-size: 14px;">as at ${dateStr}</p>
  </div>
  <div style="padding: 20px;">
    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
      <thead>
        <tr style="background-color: #1a1a2e;">
          <th style="padding: 12px 16px; text-align: left; color: #f5c518; font-weight: 600; border-bottom: 2px solid #f5c518;">#</th>
          <th style="padding: 12px 16px; text-align: left; color: #f5c518; font-weight: 600; border-bottom: 2px solid #f5c518;">Name</th>
          <th style="padding: 12px 16px; text-align: left; color: #f5c518; font-weight: 600; border-bottom: 2px solid #f5c518;">Venue</th>
          <th style="padding: 12px 16px; text-align: center; color: #f5c518; font-weight: 600; border-bottom: 2px solid #f5c518;"># Star of the Shift</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  </div>
  <div style="background-color: #f9f9f9; padding: 16px 20px; text-align: center; border-top: 1px solid #eee;">
    <p style="margin: 0; font-size: 12px; color: #999;">Generated automatically from Yo-Chi Operations Update | Yochi Team Sharepoint</p>
  </div>
</div>
</body></html>`;
}

interface SharePointItem {
  starOfShift: string;
  venue: string;
  dateToday: string;
}
```

---

## Complete Power Automate Flow (with Office Script approach)

### Step 1: Recurrence Trigger
- Every 1 Day at 8:00 AM AEST

### Step 2: SharePoint > Get items
- Site: `https://embraceyochi.sharepoint.com/sites/YochiTeamSharepoint`
- List: `Operations Update`

### Step 3: Compose - Build Script Input
Transform SharePoint items into the format the script expects:

**Expression:**
```
json(
  concat('[',
    join(
      body('Get_items')?['value'],
      ','
    ),
  ']')
)
```

Or use a **Select** action to map each item:
- **From:** `body('Get_items')?['value']`
- **Map:**
  - `starOfShift`: `item()?['Star_x0020_of_x0020_the_x0020_shift_x003f_']`
  - `venue`: `item()?['Venue']`
  - `dateToday`: `item()?['Date_x0020_today']`

### Step 4: Excel Online > Run script
- **Location:** OneDrive for Business (or SharePoint)
- **Document Library:** Documents
- **File:** Any Excel file (create a blank `StarHelper.xlsx`)
- **Script:** Select the script you created above
- **items:** Output from the Select action (JSON array)

### Step 5: Get CSV Recipients
- **Action:** SharePoint > Get file content
- **Site:** `https://embraceyochi.sharepoint.com/sites/YochiTeamSharepoint`
- **File:** `/Shared Documents/StarOfTheShift/email_recipients.csv`

Then use a **Compose** action to parse CSV lines.

**OR simpler:** Create a SharePoint list called "Star Email Recipients" with
Name and Email columns, and use "Get items" instead of a CSV.

### Step 6: Send Email
- **Action:** Office 365 Outlook > Send an email (V2)
- **To:** Semicolon-joined email addresses from the recipients
- **Subject:** `Star of the Shift Champion - MTD @{formatDateTime(utcNow(), 'MMMM yyyy')}`
- **Body:** Output from the Run script action (the HTML string)
- **Is HTML:** Yes

---

## Simplest Possible Alternative (No Office Script needed)

If you don't have access to Office Scripts, you can use a **pure Power Automate**
approach. I recommend creating the recipients as a SharePoint list instead of CSV,
and using the flow template below. Let me know and I can walk you through it step
by step in Power Automate directly.

---

## Column Name Reference
When building the flow, you'll need the internal SharePoint column names.
From what I observed in the list:

| Display Name           | Likely Internal Name                                    |
|------------------------|----------------------------------------------------------|
| Venue                  | `Venue`                                                  |
| Star of the shift?     | `Star_x0020_of_x0020_the_x0020_shift_x003f_`            |
| What is your name?     | `What_x0020_is_x0020_your_x0020_name_x003f_`            |
| Date today             | `Date_x0020_today`                                       |

To verify: In SharePoint, click column header > Column settings > Edit.
Check the URL for `&Field=InternalName`.
