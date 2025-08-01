// sectionwise.js
// This script expects a global variable `modelResult` containing the backend response.
// It will populate the section-wise prediction table with all relevant fields.

document.addEventListener('DOMContentLoaded', function() {
    // You may need to adjust this selector to match your table's id or class
    const tableBody = document.querySelector('#sectionwise-table tbody');
    if (!tableBody) return;

    // Use a global or window variable, or fetch from API as needed
    // For demo, assume modelResult is available
    const result = window.modelResult || {};
    const sections = Object.keys(result.predicted_price_by_section || {});

    tableBody.innerHTML = '';
    sections.forEach(section => {
        const predictedPrice = result.predicted_price_by_section[section] !== undefined ? `$${result.predicted_price_by_section[section].toFixed(2)}` : '-';
        const buyPriceArr = (result.buy_price_by_section && result.buy_price_by_section[section]) || [];
        const buyPrice = buyPriceArr.length ? `$${buyPriceArr[0].toFixed(2)} - $${buyPriceArr[1].toFixed(2)}` : '-';
        const absOptimal = (result.buy_days_by_section && result.buy_days_by_section[section] !== undefined && result.buy_days_by_section[section] !== null)
            ? `${result.buy_days_by_section[section]} days in advance` : '-';
        const upcomingOptimal = (result.upcoming_optimal_timing_by_section && result.upcoming_optimal_timing_by_section[section]) || '-';
        // Optionally, add upcoming_optimal_days_by_section if you want

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${section}</td>
            <td>${predictedPrice}</td>
            <td>${buyPrice}</td>
            <td>${absOptimal}</td>
            <td>${upcomingOptimal}</td>
        `;
        tableBody.appendChild(row);
    });
});
