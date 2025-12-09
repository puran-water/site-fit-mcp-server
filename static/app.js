/**
 * Site-Fit Viewer Application
 *
 * Interactive viewer for site layout solutions using Leaflet with CRS.Simple
 * for planar (meter) coordinates.
 */

// Global state
const state = {
    map: null,
    solutions: [],
    currentSolution: null,
    layers: {
        boundary: null,
        structures: null,
        roads: null,
        entrances: null,
        labels: null,
    },
    layerVisibility: {
        boundary: true,
        structures: true,
        roads: true,
        entrances: true,
        labels: true,
    },
};

// Style configurations
const styles = {
    boundary: {
        color: '#3388ff',
        weight: 3,
        fillColor: '#3388ff',
        fillOpacity: 0.1,
    },
    structure: {
        color: '#ff7800',
        weight: 2,
        fillColor: '#ff7800',
        fillOpacity: 0.5,
    },
    structureSelected: {
        color: '#e94560',
        weight: 3,
        fillColor: '#e94560',
        fillOpacity: 0.6,
    },
    road: {
        color: '#666666',
        weight: 6,
        opacity: 0.8,
        lineCap: 'round',
        lineJoin: 'round',
    },
    entrance: {
        radius: 8,
        color: '#00ff00',
        weight: 2,
        fillColor: '#00ff00',
        fillOpacity: 0.8,
    },
    keepout: {
        color: '#ff0000',
        weight: 2,
        fillColor: '#ff0000',
        fillOpacity: 0.3,
        dashArray: '5,5',
    },
};

/**
 * Initialize the application
 */
function init() {
    initMap();
    initLayerControls();
    // Try to load from backend first, fall back to demo data
    loadJobs();
}

/**
 * Initialize Leaflet map with CRS.Simple for meter coordinates
 */
function initMap() {
    // CRS.Simple for planar coordinates (meters)
    state.map = L.map('map', {
        crs: L.CRS.Simple,
        minZoom: -3,
        maxZoom: 3,
    });

    // Set initial view (will be adjusted when data loads)
    state.map.setView([0, 0], 0);

    // Add scale control
    L.control.scale({
        metric: true,
        imperial: false,
        position: 'bottomleft',
    }).addTo(state.map);
}

/**
 * Initialize layer visibility controls
 */
function initLayerControls() {
    const layerNames = ['boundary', 'structures', 'roads', 'entrances', 'labels'];

    layerNames.forEach(name => {
        const checkbox = document.getElementById(`layer-${name}`);
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                state.layerVisibility[name] = e.target.checked;
                updateLayerVisibility();
            });
        }
    });
}

/**
 * Update layer visibility based on checkbox states
 */
function updateLayerVisibility() {
    Object.keys(state.layers).forEach(name => {
        const layer = state.layers[name];
        if (layer) {
            if (state.layerVisibility[name]) {
                if (!state.map.hasLayer(layer)) {
                    state.map.addLayer(layer);
                }
            } else {
                if (state.map.hasLayer(layer)) {
                    state.map.removeLayer(layer);
                }
            }
        }
    });
}

/**
 * Load a solution GeoJSON and render it
 */
function loadSolution(solution) {
    state.currentSolution = solution;
    clearLayers();

    if (!solution.features_geojson) {
        setStatus('No GeoJSON data in solution');
        return;
    }

    renderGeoJSON(solution.features_geojson);
    updateSolutionDetails(solution);
    fitMapToContent();
    setStatus(`Loaded solution ${solution.id}`);
}

/**
 * Render GeoJSON feature collection
 */
function renderGeoJSON(geojson) {
    // Separate features by kind
    const features = geojson.features || [];

    const boundaryFeatures = features.filter(f => f.properties?.kind === 'boundary');
    const structureFeatures = features.filter(f => f.properties?.kind === 'structure');
    const roadFeatures = features.filter(f => f.properties?.kind === 'road');
    const entranceFeatures = features.filter(f => f.properties?.kind === 'entrance');
    const keepoutFeatures = features.filter(f => f.properties?.kind === 'keepout');

    // Render each layer
    if (boundaryFeatures.length > 0) {
        state.layers.boundary = L.geoJSON(
            { type: 'FeatureCollection', features: boundaryFeatures },
            {
                style: styles.boundary,
                coordsToLatLng: coordsToLatLng,
            }
        ).addTo(state.map);
    }

    if (structureFeatures.length > 0) {
        state.layers.structures = L.geoJSON(
            { type: 'FeatureCollection', features: structureFeatures },
            {
                style: styles.structure,
                coordsToLatLng: coordsToLatLng,
                onEachFeature: onEachStructure,
            }
        ).addTo(state.map);
    }

    if (roadFeatures.length > 0) {
        state.layers.roads = L.geoJSON(
            { type: 'FeatureCollection', features: roadFeatures },
            {
                style: styles.road,
                coordsToLatLng: coordsToLatLng,
                onEachFeature: onEachRoad,
            }
        ).addTo(state.map);
    }

    if (entranceFeatures.length > 0) {
        state.layers.entrances = L.geoJSON(
            { type: 'FeatureCollection', features: entranceFeatures },
            {
                pointToLayer: (feature, latlng) => {
                    return L.circleMarker(latlng, styles.entrance);
                },
                coordsToLatLng: coordsToLatLng,
                onEachFeature: onEachEntrance,
            }
        ).addTo(state.map);
    }

    // Labels layer (structure IDs)
    if (structureFeatures.length > 0) {
        const labelMarkers = structureFeatures.map(f => {
            const coords = getFeatureCentroid(f);
            if (!coords) return null;

            const latlng = coordsToLatLng(coords);
            return L.marker(latlng, {
                icon: L.divIcon({
                    className: 'structure-label',
                    html: `<span>${f.properties?.id || ''}</span>`,
                    iconSize: [100, 20],
                    iconAnchor: [50, 10],
                }),
            });
        }).filter(m => m !== null);

        state.layers.labels = L.layerGroup(labelMarkers).addTo(state.map);
    }
}

/**
 * Convert [x, y] coordinates to Leaflet LatLng
 * For CRS.Simple, y is lat and x is lng
 */
function coordsToLatLng(coords) {
    return L.latLng(coords[1], coords[0]);
}

/**
 * Get centroid of a GeoJSON feature
 */
function getFeatureCentroid(feature) {
    const geom = feature.geometry;
    if (!geom) return null;

    if (geom.type === 'Point') {
        return geom.coordinates;
    }

    if (geom.type === 'Polygon') {
        const ring = geom.coordinates[0];
        if (!ring || ring.length === 0) return null;

        let cx = 0, cy = 0;
        ring.forEach(coord => {
            cx += coord[0];
            cy += coord[1];
        });
        return [cx / ring.length, cy / ring.length];
    }

    if (geom.type === 'LineString') {
        const coords = geom.coordinates;
        if (!coords || coords.length === 0) return null;

        const mid = Math.floor(coords.length / 2);
        return coords[mid];
    }

    return null;
}

/**
 * Handle structure feature setup (popup, events)
 */
function onEachStructure(feature, layer) {
    const props = feature.properties || {};

    layer.bindPopup(`
        <div class="popup-title">${props.id || 'Structure'}</div>
        <div class="popup-detail">Rotation: ${props.rotation || 0}°</div>
    `);

    layer.on('mouseover', () => {
        layer.setStyle(styles.structureSelected);
    });

    layer.on('mouseout', () => {
        layer.setStyle(styles.structure);
    });
}

/**
 * Handle road feature setup
 */
function onEachRoad(feature, layer) {
    const props = feature.properties || {};

    layer.bindPopup(`
        <div class="popup-title">Road Segment</div>
        <div class="popup-detail">ID: ${props.id || 'unknown'}</div>
        <div class="popup-detail">Width: ${props.width || 6}m</div>
    `);
}

/**
 * Handle entrance feature setup
 */
function onEachEntrance(feature, layer) {
    const props = feature.properties || {};

    layer.bindPopup(`
        <div class="popup-title">Entrance</div>
        <div class="popup-detail">ID: ${props.id || 'unknown'}</div>
    `);
}

/**
 * Clear all layers
 */
function clearLayers() {
    Object.keys(state.layers).forEach(key => {
        if (state.layers[key]) {
            state.map.removeLayer(state.layers[key]);
            state.layers[key] = null;
        }
    });
}

/**
 * Fit map to show all content
 */
function fitMapToContent() {
    const bounds = [];

    Object.values(state.layers).forEach(layer => {
        if (layer && layer.getBounds) {
            try {
                const b = layer.getBounds();
                if (b.isValid()) {
                    bounds.push(b);
                }
            } catch (e) {
                // Some layers don't have getBounds
            }
        }
    });

    if (bounds.length > 0) {
        let combinedBounds = bounds[0];
        bounds.slice(1).forEach(b => {
            combinedBounds = combinedBounds.extend(b);
        });
        state.map.fitBounds(combinedBounds, { padding: [50, 50] });
    }
}

/**
 * Update solution details panel
 */
function updateSolutionDetails(solution) {
    const container = document.getElementById('solution-details');
    if (!container) return;

    const metrics = solution.metrics || {};

    container.innerHTML = `
        <div class="detail-row">
            <span class="label">Solution ID</span>
            <span class="value">${solution.id}</span>
        </div>
        <div class="detail-row">
            <span class="label">Rank</span>
            <span class="value">#${solution.rank + 1}</span>
        </div>
        <div class="detail-row">
            <span class="label">Road Length</span>
            <span class="value">${(metrics.road_length || 0).toFixed(1)}m</span>
        </div>
        <div class="detail-row">
            <span class="label">Compactness</span>
            <span class="value">${((metrics.compactness || 0) * 100).toFixed(1)}%</span>
        </div>
        <div class="detail-row">
            <span class="label">Flow Penalty</span>
            <span class="value">${(metrics.topology_penalty || 0).toFixed(1)}</span>
        </div>
        ${solution.diversity_note ? `
            <div class="detail-row" style="flex-direction: column;">
                <span class="label">Note</span>
                <span class="value" style="font-style: italic;">${solution.diversity_note}</span>
            </div>
        ` : ''}
    `;
}

/**
 * Update solution list in sidebar
 */
function updateSolutionList(solutions) {
    const container = document.getElementById('solution-list');
    if (!container) return;

    if (!solutions || solutions.length === 0) {
        container.innerHTML = '<p class="placeholder">No solutions available.</p>';
        return;
    }

    container.innerHTML = solutions.map((sol, idx) => `
        <div class="solution-card ${sol.id === state.currentSolution?.id ? 'selected' : ''}"
             data-solution-index="${idx}">
            <div class="rank">#${sol.rank + 1}</div>
            <div class="metrics">
                Road: ${(sol.metrics?.road_length || 0).toFixed(0)}m |
                Compact: ${((sol.metrics?.compactness || 0) * 100).toFixed(0)}%
            </div>
            ${sol.diversity_note ? `<div class="diversity-note">${sol.diversity_note}</div>` : ''}
        </div>
    `).join('');

    // Add click handlers
    container.querySelectorAll('.solution-card').forEach(card => {
        card.addEventListener('click', () => {
            const idx = parseInt(card.dataset.solutionIndex, 10);
            if (state.solutions[idx]) {
                loadSolution(state.solutions[idx]);
                updateSolutionList(state.solutions);
            }
        });
    });
}

/**
 * Set status message
 */
function setStatus(message) {
    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.textContent = message;
    }
}

/**
 * Load demo data for testing (fallback when no backend available)
 */
function loadDemoData() {
    // Create a simple demo solution
    const demoSolution = {
        id: 'demo-001',
        rank: 0,
        metrics: {
            road_length: 125.5,
            compactness: 0.72,
            topology_penalty: 5.2,
        },
        diversity_note: 'Demo solution (no backend connected)',
        features_geojson: {
            type: 'FeatureCollection',
            features: [
                // Site boundary
                {
                    type: 'Feature',
                    geometry: {
                        type: 'Polygon',
                        coordinates: [[[0, 0], [200, 0], [200, 150], [0, 150], [0, 0]]],
                    },
                    properties: { kind: 'boundary' },
                },
                // Entrance
                {
                    type: 'Feature',
                    geometry: {
                        type: 'Point',
                        coordinates: [100, 0],
                    },
                    properties: { kind: 'entrance', id: 'main-gate' },
                },
                // Structures
                {
                    type: 'Feature',
                    geometry: {
                        type: 'Polygon',
                        coordinates: [[[20, 20], [50, 20], [50, 50], [20, 50], [20, 20]]],
                    },
                    properties: { kind: 'structure', id: 'aeration-1', rotation: 0 },
                },
                {
                    type: 'Feature',
                    geometry: {
                        type: 'Polygon',
                        coordinates: [[[70, 30], [110, 30], [110, 60], [70, 60], [70, 30]]],
                    },
                    properties: { kind: 'structure', id: 'clarifier-1', rotation: 0 },
                },
                {
                    type: 'Feature',
                    geometry: {
                        type: 'Polygon',
                        coordinates: [[[130, 40], [170, 40], [170, 80], [130, 80], [130, 40]]],
                    },
                    properties: { kind: 'structure', id: 'digester-1', rotation: 0 },
                },
                {
                    type: 'Feature',
                    geometry: {
                        type: 'Polygon',
                        coordinates: [[[40, 80], [80, 80], [80, 120], [40, 120], [40, 80]]],
                    },
                    properties: { kind: 'structure', id: 'control-bldg', rotation: 0 },
                },
                // Roads
                {
                    type: 'Feature',
                    geometry: {
                        type: 'LineString',
                        coordinates: [[100, 0], [100, 20], [35, 20]],
                    },
                    properties: { kind: 'road', id: 'road-1', width: 6 },
                },
                {
                    type: 'Feature',
                    geometry: {
                        type: 'LineString',
                        coordinates: [[100, 20], [90, 30]],
                    },
                    properties: { kind: 'road', id: 'road-2', width: 6 },
                },
                {
                    type: 'Feature',
                    geometry: {
                        type: 'LineString',
                        coordinates: [[100, 20], [150, 40]],
                    },
                    properties: { kind: 'road', id: 'road-3', width: 6 },
                },
            ],
        },
    };

    state.solutions = [demoSolution];
    loadSolution(demoSolution);
    updateSolutionList(state.solutions);
}

// ============================================================================
// API Communication
// ============================================================================

const API_BASE = window.location.origin;

/**
 * Fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (e) {
        console.error(`API error (${endpoint}):`, e);
        throw e;
    }
}

/**
 * Load all jobs from backend
 */
async function loadJobs() {
    setStatus('Loading jobs...');
    try {
        const result = await apiFetch('/api/jobs');
        if (result.jobs && result.jobs.length > 0) {
            setStatus(`Found ${result.jobs.length} job(s)`);
            // Load the most recent job
            const latestJob = result.jobs[result.jobs.length - 1];
            await loadSolutionsFromJob(latestJob.job_id);
        } else {
            setStatus('No jobs found. Use API to generate solutions.');
            loadDemoData();
        }
    } catch (e) {
        setStatus(`Backend not available - showing demo data`);
        loadDemoData();
    }
}

/**
 * Load solutions from a job via REST API
 */
async function loadSolutionsFromJob(jobId) {
    setStatus(`Loading job ${jobId}...`);

    try {
        // Get job status
        const job = await apiFetch(`/api/jobs/${jobId}`);

        if (job.status !== 'completed') {
            setStatus(`Job ${jobId}: ${job.status}`);
            return;
        }

        // Get solution list
        const listResult = await apiFetch(`/api/jobs/${jobId}/solutions`);

        if (!listResult.solutions || listResult.solutions.length === 0) {
            setStatus('No solutions found for this job');
            return;
        }

        // Load full details for each solution
        const fullSolutions = [];
        for (const summary of listResult.solutions) {
            try {
                const full = await apiFetch(`/api/solutions/${summary.id}`);
                fullSolutions.push(full);
            } catch (e) {
                console.error(`Failed to load solution ${summary.id}:`, e);
            }
        }

        if (fullSolutions.length === 0) {
            setStatus('Failed to load solution details');
            return;
        }

        state.solutions = fullSolutions;
        loadSolution(fullSolutions[0]);
        updateSolutionList(fullSolutions);
        setStatus(`Loaded ${fullSolutions.length} solutions from job ${jobId}`);

    } catch (e) {
        setStatus(`Error loading job: ${e.message}`);
        console.error('Failed to load job:', e);
    }
}

/**
 * Generate new solutions via REST API
 */
async function generateSolutions(request) {
    setStatus('Generating solutions...');

    try {
        const result = await apiFetch('/api/generate', {
            method: 'POST',
            body: JSON.stringify(request),
        });

        if (result.status === 'completed' && result.num_solutions > 0) {
            setStatus(`Generated ${result.num_solutions} solutions`);
            await loadSolutionsFromJob(result.job_id);
        } else {
            setStatus(`Generation completed with ${result.num_solutions} solutions`);
        }

        return result;
    } catch (e) {
        setStatus(`Generation failed: ${e.message}`);
        throw e;
    }
}

/**
 * Export solution to format
 */
async function exportSolution(solutionId, format) {
    setStatus(`Exporting solution as ${format}...`);

    try {
        const result = await apiFetch(`/api/solutions/${solutionId}/export/${format}`);
        setStatus(`Exported solution as ${format}`);
        return result;
    } catch (e) {
        setStatus(`Export failed: ${e.message}`);
        throw e;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);

// Export functions for external use
window.SiteFitViewer = {
    loadSolution,
    loadSolutionsFromJob,
    setStatus,
};
