/* ========================================================
   Threat Detection Console - Minimalist Client Logic
   ======================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const digitalClock = document.getElementById('digitalClock');
    const fpsDisplay = document.getElementById('fpsDisplay');
    const globalStatusBadge = document.getElementById('globalStatusBadge');
    const systemStatusDot = document.getElementById('systemStatusDot');
    const detectedCountOverlay = document.getElementById('detectedCountOverlay');
    const alarmBanner = document.getElementById('alarmBanner');
    const cameraSelect = document.getElementById('cameraSelect');
    const engineSelect = document.getElementById('engineSelect');
    const confSlider = document.getElementById('confSlider');
    const confValue = document.getElementById('confValue');
    const inferFpsDisplay = document.getElementById('inferFpsDisplay');
    const classTagsContainer = document.getElementById('classTagsContainer');
    const toggleAudioBtn = document.getElementById('toggleAudioBtn');
    const snapshotBtn = document.getElementById('snapshotBtn');
    const restartFeedBtn = document.getElementById('restartFeedBtn');
    const logBody = document.getElementById('logBody');
    const clearLogBtn = document.getElementById('clearLogBtn');
    const evidenceGrid = document.getElementById('evidenceGrid');
    const evidenceCount = document.getElementById('evidenceCount');
    const alertAudio = document.getElementById('alertAudio');
    const videoStream = document.getElementById('videoStream');

    let audioEnabled = false;
    let loggedDetections = new Set();
    let evidenceSnapshots = [];
    let lastAudioTime = 0;

    // 1. Digital Clock
    function updateClock() {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        digitalClock.textContent = `${hours}:${minutes}:${seconds} UTC+05:30`;
    }
    setInterval(updateClock, 1000);
    updateClock();

    // 2. Audio Alert Toggle
    toggleAudioBtn.addEventListener('click', () => {
        audioEnabled = !audioEnabled;
        if (audioEnabled) {
            toggleAudioBtn.innerHTML = '<span class="btn-icon">🔊</span> ALARM: ON';
            toggleAudioBtn.classList.add('btn-primary');
            toggleAudioBtn.classList.remove('btn-secondary');
            alertAudio.play().then(() => alertAudio.pause()).catch(() => {});
        } else {
            toggleAudioBtn.innerHTML = '<span class="btn-icon">🔇</span> ALARM: OFF';
            toggleAudioBtn.classList.remove('btn-primary');
            toggleAudioBtn.classList.add('btn-secondary');
        }
    });

    // 3. Confidence Slider
    confSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value).toFixed(2);
        confValue.textContent = val;
        updateConfig({ confidence_threshold: parseFloat(val) });
    });

    // 4. Camera & Engine Selection
    cameraSelect.addEventListener('change', (e) => {
        updateConfig({ camera_index: parseInt(e.target.value) });
        restartFeed();
    });

    if (engineSelect) {
        engineSelect.addEventListener('change', (e) => {
            updateConfig({ engine: e.target.value });
            updateClassTags(e.target.value);
        });
    }

    function updateClassTags(engine) {
        if (!classTagsContainer) return;
        if (engine === 'sota_threat' || engine === 'specialized') {
            classTagsContainer.innerHTML = `
                <span class="tag">Gun / Pistol</span>
                <span class="tag">Grenade / Bomb</span>
                <span class="tag">Knife / Blade</span>
                <span class="tag">Explosive</span>
                <span class="tag" style="opacity:0.6">Immune: Scissors, Bare Arm, Tools</span>
            `;
        } else if (engine === 'yolo11_tactical') {
            classTagsContainer.innerHTML = `
                <span class="tag">Gun / Firearm</span>
                <span class="tag">Knife / Blade</span>
                <span class="tag" style="opacity:0.6">YOLOv11 Tactical Medium</span>
            `;
        } else if (engine === 'extended') {
            classTagsContainer.innerHTML = `
                <span class="tag">Gun / Pistol</span>
                <span class="tag">Rifle / Firearm</span>
                <span class="tag">Heavy Weapon</span>
                <span class="tag">Knife / Blade</span>
            `;
        } else if (engine === 'legacy') {
            classTagsContainer.innerHTML = `
                <span class="tag">Pistol / Handgun</span>
                <span class="tag">Combat Knife</span>
                <span class="tag" style="opacity:0.6">Legacy Dual-Class Model</span>
            `;
        } else {
            classTagsContainer.innerHTML = `
                <span class="tag">Handgun / Pistol</span>
                <span class="tag">Rifle / Firearm</span>
                <span class="tag">Knife / Blade</span>
                <span class="tag">Bomb / Explosive</span>
                <span class="tag">Grenade</span>
                <span class="tag" style="opacity:0.6">Grounded: Scissors, Tools, Arm</span>
            `;
        }
    }

    // 5. Restart Feed Button
    restartFeedBtn.addEventListener('click', () => {
        restartFeed();
    });

    function restartFeed() {
        videoStream.src = '';
        setTimeout(() => {
            videoStream.src = `/video_feed?t=${Date.now()}`;
        }, 300);
    }

    videoStream.addEventListener('error', () => {
        setTimeout(restartFeed, 1500);
    });

    // 6. Manual Snapshot
    snapshotBtn.addEventListener('click', async () => {
        try {
            const resp = await fetch('/api/snapshot', { method: 'POST' });
            const data = await resp.json();
            if (data.status === 'success') {
                addEvidenceSnapshot(data.snapshot);
            }
        } catch (err) {
            console.error('Snapshot failed:', err);
        }
    });

    // 7. Clear Log
    clearLogBtn.addEventListener('click', () => {
        logBody.innerHTML = '<tr class="empty-row"><td colspan="4">No security incidents detected.</td></tr>';
        loggedDetections.clear();
    });

    // Helper: Update Backend Config
    async function updateConfig(config) {
        try {
            await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
        } catch (err) {
            console.error('Config update error:', err);
        }
    }

    // 8. Real-time Status Polling
    async function pollStatus() {
        try {
            const resp = await fetch('/api/status');
            if (!resp.ok) return;
            const data = await resp.json();

            // FPS Displays
            fpsDisplay.textContent = (data.fps || '--') + ' FPS';
            if (inferFpsDisplay) {
                inferFpsDisplay.textContent = (data.inference_fps || '--') + ' FPS';
            }

            // Active Threat Status
            const hasThreat = data.active_threats && data.active_threats.length > 0;
            if (hasThreat) {
                globalStatusBadge.textContent = 'THREAT DETECTED';
                globalStatusBadge.className = 'status-indicator-badge status-alert';
                alarmBanner.classList.add('active');
                systemStatusDot.style.backgroundColor = 'var(--status-threat)';
                detectedCountOverlay.textContent = `THREATS: ${data.active_threats.length}`;

                // Play alarm sound if enabled (throttle every 3s)
                const now = Date.now();
                if (audioEnabled && now - lastAudioTime > 3000) {
                    lastAudioTime = now;
                    alertAudio.currentTime = 0;
                    alertAudio.play().catch(() => {});
                }

                // Add to incident log
                data.active_threats.forEach(t => {
                    const dedupeKey = `${t.label}_${Math.floor(Date.now() / 2000)}`;
                    if (!loggedDetections.has(dedupeKey)) {
                        loggedDetections.add(dedupeKey);
                        addLogEntry(t);
                    }
                });
            } else {
                globalStatusBadge.textContent = 'MONITORING';
                globalStatusBadge.className = 'status-indicator-badge status-idle';
                alarmBanner.classList.remove('active');
                systemStatusDot.style.backgroundColor = '#ffffff';
                detectedCountOverlay.textContent = 'THREATS: 0';
            }

            // Sync evidence snapshots if new
            if (data.latest_snapshot && (!evidenceSnapshots.length || evidenceSnapshots[0].url !== data.latest_snapshot.url)) {
                addEvidenceSnapshot(data.latest_snapshot);
            }
        } catch (err) {
            // Keep silent on connection retry
        } finally {
            setTimeout(pollStatus, 500);
        }
    }

    // Helper: Add Incident Log Entry
    function addLogEntry(threat) {
        const emptyRow = logBody.querySelector('.empty-row');
        if (emptyRow) {
            emptyRow.remove();
        }

        const now = new Date();
        const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

        const isCritical = ['bomb', 'grenade', 'explosive', 'rifle', 'firearm', 'gun', 'pistol'].some(c => 
            threat.label.toLowerCase().includes(c)
        );

        const badgeClass = isCritical ? 'threat-high' : 'threat-elevated';
        const levelText = isCritical ? 'CRITICAL' : 'ELEVATED';

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${timeStr}</td>
            <td><strong>${threat.label.toUpperCase()}</strong></td>
            <td>${(threat.confidence * 100).toFixed(0)}%</td>
            <td><span class="threat-badge ${badgeClass}">${levelText}</span></td>
        `;

        logBody.insertBefore(tr, logBody.firstChild);

        // Limit log entries to 30
        while (logBody.children.length > 30) {
            logBody.removeChild(logBody.lastChild);
        }
    }

    // Helper: Add Evidence Snapshot
    function addEvidenceSnapshot(snapshot) {
        if (!snapshot || !snapshot.url) return;
        evidenceSnapshots.unshift(snapshot);
        evidenceCount.textContent = `${evidenceSnapshots.length} ITEMS`;

        const empty = evidenceGrid.querySelector('.evidence-empty');
        if (empty) empty.remove();

        const item = document.createElement('div');
        item.className = 'evidence-item';
        item.innerHTML = `
            <img src="${snapshot.url}" alt="Threat evidence">
            <div class="evidence-caption">${snapshot.timestamp} // ${snapshot.label || 'THREAT'}</div>
        `;
        item.addEventListener('click', () => {
            window.open(snapshot.url, '_blank');
        });

        evidenceGrid.insertBefore(item, evidenceGrid.firstChild);

        // Limit to 10 snapshots in UI
        while (evidenceGrid.children.length > 10) {
            evidenceGrid.removeChild(evidenceGrid.lastChild);
        }
    }

    // Start Polling
    pollStatus();
});
