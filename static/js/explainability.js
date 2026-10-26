/**
 * Carlytics — Model Explainability (XAI)
 * Handles global feature importance visualizations and local feature ablation diagnostics.
 */

let currentImportanceView = 'base'; // 'base' or 'transformed'
let activeExplainVehicle = {
    brand: 'Toyota',
    year: 2018,
    km_driven: 55000,
    fuel: 'Diesel',
    seller_type: 'Dealer',
    transmission: 'Automatic',
    owner: 'First Owner'
};

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial render of Global Feature Importance Chart
    renderGlobalImportanceChart(currentImportanceView);

    // 2. Initial calculation and render of Local Prediction Feature Sensitivity
    fetchLocalExplanation(activeExplainVehicle);
});

/**
 * Toggle between Base Feature Groups and Transformed Features
 */
function switchImportanceView(view) {
    if (currentImportanceView === view) return;
    currentImportanceView = view;

    document.getElementById('toggleBaseBtn').classList.toggle('active', view === 'base');
    document.getElementById('toggleTransBtn').classList.toggle('active', view === 'transformed');

    renderGlobalImportanceChart(view);
}

/**
 * Render Global Feature Importance horizontal bar chart using genuine MDI values
 */
function renderGlobalImportanceChart(view) {
    let dataList = [];
    if (view === 'base') {
        dataList = [...BASE_IMPORTANCES].reverse(); // reverse for bottom-to-top horizontal bar chart
    } else {
        dataList = [...TRANSFORMED_IMPORTANCES].slice(0, 12).reverse();
    }

    const yNames = dataList.map(d => d.name);
    const xValues = dataList.map(d => d.percentage);
    const mdiValues = dataList.map(d => d.importance);

    // Bar colors gradient based on rank
    const colors = dataList.map((d, i) => {
        if (i >= dataList.length - 3) return '#4f46e5'; // top 3
        if (i >= dataList.length - 6) return '#0284c7';
        return '#64748b';
    });

    const trace = {
        type: 'bar',
        x: xValues,
        y: yNames,
        orientation: 'h',
        marker: {
            color: colors,
            line: { color: '#ffffff', width: 1 }
        },
        text: xValues.map(v => `${v}%`),
        textposition: 'outside',
        textfont: { size: 11, color: '#334155', weight: 'bold' },
        hovertemplate: '<b>%{y}</b><br>Share of Importance: <b>%{x}%</b><br>Exact MDI Score: %{customdata:.4f}<extra></extra>',
        customdata: mdiValues
    };

    const layout = {
        margin: { l: 160, r: 40, t: 15, b: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            title: 'Gini Importance Percentage Share (%)',
            titlefont: { size: 12, color: '#475569' },
            tickfont: { size: 11, color: '#64748b' },
            gridcolor: '#f1f5f9',
            zeroline: false
        },
        yaxis: {
            tickfont: { size: 12, color: '#0f172a' },
            automargin: true
        },
        hovermode: 'closest'
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.react('featureImportanceChart', [trace], layout, config);
}

/**
 * Load a preset vehicle for local prediction explainability
 */
function loadExplainVehicle(brand, year, km, fuel, trans, seller, owner) {
    activeExplainVehicle = {
        brand: brand,
        year: year,
        km_driven: km,
        fuel: fuel,
        transmission: trans,
        seller_type: seller,
        owner: owner
    };

    fetchLocalExplanation(activeExplainVehicle);
}

/**
 * Fetch local feature ablation sensitivity from /api/explain-prediction
 */
async function fetchLocalExplanation(vehicle) {
    try {
        const response = await fetch('/api/explain-prediction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ vehicle: vehicle })
        });

        const data = await response.json();
        if (!data.success) {
            console.error('Explainability error:', data.error);
            return;
        }

        renderLocalExplanation(data);
    } catch (err) {
        console.error('Error fetching prediction explanation:', err);
    }
}

/**
 * Render Diverging Contribution Chart and Attribution Diagnostics Table
 */
function renderLocalExplanation(data) {
    // 1. Update text prices
    document.getElementById('refValuationPill').textContent = 
        `Reference Value: ${data.reference_price_lakh} (${data.reference_price_full})`;
    
    document.getElementById('explainTargetPrice').textContent = 
        `Predicted Value: ${data.vehicle_price_lakh}`;

    // 2. Render Diverging Contribution Chart
    // Sort contributions so largest positive is at top of vertical axis
    const sortedContrib = [...data.contributions].reverse();
    
    const yLabels = sortedContrib.map(c => c.label);
    const xImpacts = sortedContrib.map(c => c.impact_lakhs);
    const barColors = sortedContrib.map(c => c.impact_raw >= 0 ? '#059669' : '#e11d48');

    const trace = {
        type: 'bar',
        x: xImpacts,
        y: yLabels,
        orientation: 'h',
        marker: {
            color: barColors,
            line: { color: '#ffffff', width: 1 }
        },
        text: xImpacts.map(v => (v > 0 ? `+₹ ${v}L` : `-₹ ${Math.abs(v)}L`)),
        textposition: 'outside',
        textfont: { size: 11, color: '#0f172a', weight: 'bold' },
        hovertemplate: '<b>%{y}</b><br>Marginal Impact vs Baseline: <b>%{x:+.2f} Lakhs</b><extra></extra>'
    };

    const layout = {
        margin: { l: 140, r: 60, t: 15, b: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            title: 'Marginal Valuation Impact (₹ Lakhs)',
            titlefont: { size: 12, color: '#475569' },
            tickfont: { size: 11, color: '#64748b' },
            gridcolor: '#f1f5f9',
            zeroline: true,
            zerolinecolor: '#cbd5e1',
            zerolinewidth: 1.5
        },
        yaxis: {
            tickfont: { size: 12, color: '#0f172a' },
            automargin: true
        },
        hovermode: 'closest'
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.react('localAblationChart', [trace], layout, config);

    // 3. Render Diagnostics Table Rows
    const tbody = document.getElementById('ablationTableBody');
    tbody.innerHTML = '';

    data.contributions.forEach(c => {
        const row = document.createElement('tr');
        
        let badgeClass = 'impact-neutral';
        let sign = '';
        if (c.direction === 'positive') {
            badgeClass = 'impact-positive';
            sign = '+';
        } else if (c.direction === 'negative') {
            badgeClass = 'impact-negative';
            sign = '-';
        }

        const absLakh = Math.abs(c.impact_lakhs);
        const impactText = `${sign}₹ ${absLakh.toFixed(2)} Lakhs`;

        row.innerHTML = `
            <td style="font-weight: 600;">${c.label}</td>
            <td><span style="font-weight: 600; color: #4338ca;">${c.vehicle_val}</span></td>
            <td style="color: #64748b;">${c.reference_val}</td>
            <td><span class="impact-badge ${badgeClass}">${impactText}</span></td>
        `;

        tbody.appendChild(row);
    });
}
