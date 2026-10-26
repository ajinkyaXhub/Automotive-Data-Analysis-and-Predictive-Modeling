/**
 * Carlytics — What-If Price Simulator
 * Client-side Controller with debounced Flask ML Pipeline synchronization.
 */

let baselineState = {
    brand: 'Maruti',
    year: 2017,
    km_driven: 45000,
    fuel: 'Petrol',
    seller_type: 'Individual',
    transmission: 'Manual',
    owner: 'First Owner'
};

let modifiedState = { ...baselineState };
let currentCurveType = 'km'; // 'km' or 'year'
let debounceTimer = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // 1. Check for incoming URL parameters (e.g. transferred from Price Estimator)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('brand')) {
        baselineState.brand = urlParams.get('brand');
        baselineState.year = parseInt(urlParams.get('year') || '2017', 10);
        baselineState.km_driven = parseInt(urlParams.get('km_driven') || '45000', 10);
        baselineState.fuel = urlParams.get('fuel') || 'Petrol';
        baselineState.transmission = urlParams.get('transmission') || 'Manual';
        baselineState.seller_type = urlParams.get('seller_type') || 'Individual';
        baselineState.owner = urlParams.get('owner') || 'First Owner';
        modifiedState = { ...baselineState };
    } else {
        // Also check sessionStorage
        const savedEstimator = sessionStorage.getItem('estimator_car_transfer');
        if (savedEstimator) {
            try {
                const parsed = JSON.parse(savedEstimator);
                baselineState = { ...parsed };
                modifiedState = { ...baselineState };
                sessionStorage.removeItem('estimator_car_transfer'); // clear after consume
            } catch (e) {
                console.warn('Could not parse transferred car details:', e);
            }
        }
    }

    // 2. Synchronize UI inputs to initial modifiedState
    syncControlsFromState();

    // 3. Trigger initial simulation
    runSimulationImmediate();
});

/**
 * Synchronize all UI form elements to match modifiedState
 */
function syncControlsFromState() {
    document.getElementById('simBrandSelect').value = modifiedState.brand;
    document.getElementById('simYearSlider').value = modifiedState.year;
    document.getElementById('simYearDisplay').textContent = modifiedState.year;
    
    document.getElementById('simKmSlider').value = modifiedState.km_driven;
    document.getElementById('simKmDisplay').textContent = `${modifiedState.km_driven.toLocaleString('en-IN')} km`;
    
    document.getElementById('simFuelSelect').value = modifiedState.fuel;
    document.getElementById('simTransSelect').value = modifiedState.transmission;
    document.getElementById('simSellerSelect').value = modifiedState.seller_type;
    document.getElementById('simOwnerSelect').value = modifiedState.owner;
}

/**
 * Handle real-time slider updates with 250ms debouncing
 */
function handleSliderChange(field, rawValue) {
    const val = parseInt(rawValue, 10);
    if (field === 'year') {
        modifiedState.year = val;
        document.getElementById('simYearDisplay').textContent = val;
    } else if (field === 'km') {
        modifiedState.km_driven = val;
        document.getElementById('simKmDisplay').textContent = `${val.toLocaleString('en-IN')} km`;
    }

    // Debounce API call to prevent server spam while sliding smoothly
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        runSimulationImmediate();
    }, 250);
}

/**
 * Handle dropdown changes immediately
 */
function handleDropdownChange() {
    modifiedState.brand = document.getElementById('simBrandSelect').value;
    modifiedState.fuel = document.getElementById('simFuelSelect').value;
    modifiedState.transmission = document.getElementById('simTransSelect').value;
    modifiedState.seller_type = document.getElementById('simSellerSelect').value;
    modifiedState.owner = document.getElementById('simOwnerSelect').value;

    runSimulationImmediate();
}

/**
 * Switch sensitivity curve dimension (Mileage vs Age)
 */
function switchCurveType(type) {
    if (currentCurveType === type) return;
    currentCurveType = type;

    document.getElementById('curveKmBtn').classList.toggle('active', type === 'km');
    document.getElementById('curveYearBtn').classList.toggle('active', type === 'year');

    runSimulationImmediate();
}

/**
 * Apply quick preset scenario
 */
function applyPresetScenario(type) {
    if (type === 'low_mileage') {
        modifiedState.km_driven = Math.max(2000, modifiedState.km_driven - 30000);
    } else if (type === 'high_mileage') {
        modifiedState.km_driven = Math.min(200000, modifiedState.km_driven + 50000);
    } else if (type === 'newer_year') {
        modifiedState.year = Math.min(2024, modifiedState.year + 3);
    } else if (type === 'older_year') {
        modifiedState.year = Math.max(1995, modifiedState.year - 4);
    } else if (type === 'auto_upgrade') {
        modifiedState.transmission = 'Automatic';
    } else if (type === 'diesel_swap') {
        modifiedState.fuel = 'Diesel';
    }

    syncControlsFromState();
    runSimulationImmediate();
}

/**
 * Reset modified state back to the baseline configuration
 */
function resetToBaseline() {
    modifiedState = { ...baselineState };
    syncControlsFromState();
    runSimulationImmediate();
}

/**
 * Lock current modified configuration as the new baseline
 */
function setAsBaseline() {
    baselineState = { ...modifiedState };
    runSimulationImmediate();
}

/**
 * Execute API call to Flask /api/simulate
 */
async function runSimulationImmediate() {
    try {
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                baseline: baselineState,
                modified: modifiedState,
                curve_type: currentCurveType
            })
        });

        const data = await response.json();
        if (!data.success) {
            console.error('Simulation error from server:', data.error);
            return;
        }

        renderSimulationResults(data);
    } catch (err) {
        console.error('Network error during simulation:', err);
    }
}

/**
 * Render simulation data into DOM and update Plotly graph
 */
function renderSimulationResults(data) {
    // 1. Update Baseline Card
    document.getElementById('basePriceFull').textContent = data.baseline.price_formatted;
    document.getElementById('basePriceLakh').textContent = `Approx. ${data.baseline.price_lakhs}`;
    document.getElementById('baseSpecsSummary').textContent = 
        `${data.baseline.specs.brand} • ${data.baseline.specs.year} • ${data.baseline.specs.km_driven.toLocaleString('en-IN')} km • ${data.baseline.specs.fuel} • ${data.baseline.specs.transmission}`;

    // 2. Update Modified Card
    const modPriceEl = document.getElementById('modPriceFull');
    modPriceEl.textContent = data.modified.price_formatted;
    document.getElementById('modPriceLakh').textContent = `Approx. ${data.modified.price_lakhs}`;
    document.getElementById('modSpecsSummary').textContent = 
        `${data.modified.specs.brand} • ${data.modified.specs.year} • ${data.modified.specs.km_driven.toLocaleString('en-IN')} km • ${data.modified.specs.fuel} • ${data.modified.specs.transmission}`;

    // Trigger subtle pulse animation on price change
    modPriceEl.classList.remove('price-pulse');
    void modPriceEl.offsetWidth; // force DOM reflow
    modPriceEl.classList.add('price-pulse');

    // 3. Update Dynamic Delta Badge
    const badge = document.getElementById('simDeltaBadge');
    badge.className = `sim-delta-badge ${data.delta.state}`;

    const icon = document.getElementById('deltaIcon');
    if (data.delta.state === 'up') {
        icon.textContent = '▲';
    } else if (data.delta.state === 'down') {
        icon.textContent = '▼';
    } else {
        icon.textContent = '—';
    }

    document.getElementById('deltaFull').textContent = data.delta.formatted;
    
    const pctSign = data.delta.percentage > 0 ? '+' : '';
    document.getElementById('deltaPct').textContent = `${pctSign}${data.delta.percentage}% relative to baseline`;

    // 4. Render or update Sensitivity Plotly Chart
    renderSensitivityChart(data.curve);
}

/**
 * Render the interactive Plotly Sensitivity Curve
 */
function renderSensitivityChart(curve) {
    const isYear = curve.type === 'year';
    const xVals = curve.points.map(p => p.x);
    const yVals = curve.points.map(p => p.y); // in Lakhs

    // Main Curve Trace
    const curveTrace = {
        x: xVals,
        y: yVals,
        mode: 'lines+markers',
        name: 'Model Valuation Curve',
        line: { color: '#4f46e5', width: 3, shape: 'spline' },
        marker: { color: '#4f46e5', size: 6 },
        hovertemplate: isYear 
            ? '<b>Year %{x}</b><br>Predicted: ₹ %{y:.2f} Lakhs<extra></extra>' 
            : '<b>%{x:,.0f} km</b><br>Predicted: ₹ %{y:.2f} Lakhs<extra></extra>'
    };

    // Baseline Reference Point
    const baselineTrace = {
        x: [curve.baseline_x],
        y: [curve.baseline_y],
        mode: 'markers+text',
        name: 'Baseline Car',
        text: ['Baseline'],
        textposition: 'top center',
        textfont: { size: 11, color: '#64748b' },
        marker: { color: '#64748b', size: 12, symbol: 'diamond', line: { color: '#ffffff', width: 2 } },
        hovertemplate: `<b>Baseline Marker</b><br>${isYear ? 'Year' : 'Mileage'}: %{x}<br>Price: ₹ %{y:.2f} Lakhs<extra></extra>`
    };

    // Current Modified Point
    const currentTrace = {
        x: [curve.current_x],
        y: [curve.current_y],
        mode: 'markers+text',
        name: 'Simulated Car',
        text: ['Active Config'],
        textposition: 'bottom center',
        textfont: { size: 11, color: '#4338ca', weight: 'bold' },
        marker: { color: '#059669', size: 14, symbol: 'circle', line: { color: '#ffffff', width: 2.5 } },
        hovertemplate: `<b>Active Simulated Point</b><br>${isYear ? 'Year' : 'Mileage'}: %{x}<br>Price: ₹ %{y:.2f} Lakhs<extra></extra>`
    };

    const layout = {
        margin: { l: 55, r: 25, t: 25, b: 50 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            title: isYear ? 'Manufacturing Year' : 'Odometer Mileage (km)',
            titlefont: { size: 12, color: '#475569' },
            tickfont: { size: 11, color: '#64748b' },
            gridcolor: '#f1f5f9',
            zeroline: false
        },
        yaxis: {
            title: 'Predicted Resale Price (₹ Lakhs)',
            titlefont: { size: 12, color: '#475569' },
            tickfont: { size: 11, color: '#64748b' },
            gridcolor: '#f1f5f9',
            zeroline: false
        },
        legend: {
            orientation: 'h',
            y: 1.12,
            x: 0.5,
            xanchor: 'center',
            font: { size: 11, color: '#475569' }
        },
        hovermode: 'closest'
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.react('simSensitivityChart', [curveTrace, baselineTrace, currentTrace], layout, config);
}
