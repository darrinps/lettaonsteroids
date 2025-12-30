document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const resetBtn = document.getElementById('reset-btn');
    const progressLog = document.getElementById('progress-log');
    const resultsLog = document.getElementById('results-log');
    const summaryEl = document.getElementById('summary');
    const sceneContainer = document.getElementById('scene-container');
    const modal = document.getElementById('modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const overviewBtn = document.getElementById('overview-btn');
    const starWarsContainer = document.getElementById('star-wars-container');
    const crawlContent = document.getElementById('crawl-content');
    const loadingMessage = document.getElementById('loading-message');
    const starWarsCrawl = document.getElementById('star-wars-crawl');

    const API_BASE = "http://localhost:8000";
    let eventSource;

    // Three.js setup
    let scene, camera, renderer, crystal, resultsGraphic;
    let isGraphSpinning = true;
    const beams = [];
    const originalColor = 0x00f0ff;
    const connectedColor = 0x00ff7f;

    let isPulsing = false;
    const baseIntensity = 1;
    let textAnimator;

    // Crawl timer variables
    let crawlTimer;
    let crawlStartTime;
    let crawlRemainingTime = 20000;

    function startTextAnimation(baseText) {
        let dotCount = 0;
        if (textAnimator) clearInterval(textAnimator);
        textAnimator = setInterval(() => {
            dotCount = (dotCount + 1) % 4;
            resultsLog.innerHTML = `<p>${baseText}${'.'.repeat(dotCount)}</p>`;
        }, 500);
    }

    function stopTextAnimation() {
        if (textAnimator) clearInterval(textAnimator);
    }

    function setCrystalColor(color) {
        if (crystal) {
            crystal.material.color.setHex(color);
            crystal.material.emissive.setHex(color);
        }
    }

    async function showStarWarsCrawl() {
        const intro = "In an LLM far, far away...<br>no, that's not right....ah...here it is...";

        // Change button to LOADING
        overviewBtn.textContent = 'LOADING';
        overviewBtn.disabled = true;

        // Show container with loading message
        starWarsContainer.style.display = 'block';
        loadingMessage.style.display = 'block';
        starWarsCrawl.style.display = 'none';

        // Fetch content
        try {
            const [headerData, evalData] = await Promise.all([
                fetch(`${API_BASE}/api/readme/header`).then(res => res.json()),
                fetch(`${API_BASE}/api/readme/eval-query-tests`).then(res => res.json())
            ]);

            if (!headerData.header || !evalData.section) {
                throw new Error("Invalid data from server");
            }
            const readmeLines = headerData.header.join('<br>');
            const evalLines = evalData.section.replace(/\n/g, '<br>');

            // Set all content at once
            crawlContent.innerHTML = `<p>${intro}</p><br>${readmeLines}<br><br>${evalLines}`;

            // Hide loading, show crawl, start animation
            loadingMessage.style.display = 'none';
            starWarsCrawl.style.display = 'block';
            starWarsCrawl.classList.remove('paused');
            starWarsCrawl.style.animation = 'none';
            starWarsCrawl.offsetHeight; /* trigger reflow */
            starWarsCrawl.style.animation = null;
            starWarsCrawl.style.animation = 'crawl 20s linear forwards';

            // Change button to PAUSE
            overviewBtn.textContent = 'PAUSE';
            overviewBtn.disabled = false;

            const onCrawlAnimationEnd = () => {
                // Change button to CLEAR when animation ends
                overviewBtn.textContent = 'CLEAR';
                starWarsCrawl.removeEventListener('animationend', onCrawlAnimationEnd);
            };
            starWarsCrawl.addEventListener('animationend', onCrawlAnimationEnd);

        } catch (err) {
            console.error('Error fetching README content:', err);
            starWarsContainer.style.display = 'none';
            overviewBtn.textContent = 'Eval Overview';
            overviewBtn.disabled = false;
            alert("Could not fetch the evaluation overview. Is the server running?");
        }
    }

    function handleOverviewButtonClick() {
        const buttonText = overviewBtn.textContent;

        if (buttonText === 'Eval Overview' || buttonText === 'LOADING') {
            // Start showing the crawl
            showStarWarsCrawl();
        } else if (buttonText === 'PAUSE') {
            // Pause the animation
            starWarsCrawl.style.animationPlayState = 'paused';
            overviewBtn.textContent = 'RESUME';
        } else if (buttonText === 'RESUME') {
            // Resume the animation
            starWarsCrawl.style.animationPlayState = 'running';
            overviewBtn.textContent = 'PAUSE';
        } else if (buttonText === 'CLEAR') {
            // Clear and reset
            starWarsContainer.style.display = 'none';
            crawlContent.innerHTML = '';
            overviewBtn.textContent = 'Eval Overview';
        }
    }

    function initThree() {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, sceneContainer.clientWidth / sceneContainer.clientHeight, 0.1, 1000);
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(sceneContainer.clientWidth, sceneContainer.clientHeight);
        renderer.domElement.style.position = 'absolute';
        renderer.domElement.style.top = '0';
        renderer.domElement.style.left = '0';
        renderer.domElement.style.zIndex = '1000';
        sceneContainer.appendChild(renderer.domElement);

        const geometry = new THREE.IcosahedronGeometry(2, 0);
        const material = new THREE.MeshStandardMaterial({
            color: originalColor,
            emissive: originalColor,
            emissiveIntensity: baseIntensity,
            metalness: 1,
            roughness: 0.2,
            wireframe: true,
        });
        crystal = new THREE.Mesh(geometry, material);
        scene.add(crystal);

        resultsGraphic = new THREE.Group();
        scene.add(resultsGraphic);

        const ambientLight = new THREE.AmbientLight(0x404040, 2);
        scene.add(ambientLight);
        const pointLight = new THREE.PointLight(0xffffff, 1, 100);
        pointLight.position.set(5, 5, 5);
        scene.add(pointLight);

        camera.position.z = 10;

        animate();
    }

    function animate() {
        requestAnimationFrame(animate);
        
        if (crystal && crystal.visible) {
            if (isPulsing) {
                const time = Date.now() * 0.005;
                crystal.material.emissiveIntensity = baseIntensity + Math.sin(time) * 0.5;
            } else {
                crystal.material.emissiveIntensity = baseIntensity;
            }
            crystal.rotation.x += 0.001;
            crystal.rotation.y += 0.002;
        }

        if (resultsGraphic && resultsGraphic.visible && isGraphSpinning) {
            resultsGraphic.rotation.y += 0.005;
        }

        beams.forEach((beam, index) => {
            beam.position.z += beam.velocity;
            if (beam.position.z > 5) { // Stop beam at the crystal
                scene.remove(beam);
                beams.splice(index, 1);
            }
        });

        renderer.render(scene, camera);
    }

    function createBeam(color, latency) {
        const geometry = new THREE.CylinderGeometry(0.05, 0.05, 4, 16);
        const material = new THREE.MeshBasicMaterial({ color });
        const beam = new THREE.Mesh(geometry, material);
        beam.rotation.x = Math.PI / 2;
        
        const angle = Math.random() * Math.PI * 2;
        const radius = 3;
        beam.position.set(Math.cos(angle) * radius, Math.sin(angle) * radius, -20);
        
        beam.velocity = Math.max(0.1, 100 / latency);
        
        beams.push(beam);
        scene.add(beam);
    }

    function createResultsGraphic(metrics) {
        // Clear previous results and hide crystal
        while(resultsGraphic.children.length > 0){
            resultsGraphic.remove(resultsGraphic.children[0]);
        }
        crystal.visible = false;
        resultsGraphic.visible = true;

        const baseline = metrics.baseline;
        const letta = metrics.letta;
        const mem0 = metrics.mem0;
        const mem0_enhanced = metrics.mem0_enhanced;

        // Calculate max values, handling optional mem0 systems
        let latencyValues = [baseline.avg_latency_ms, letta.avg_latency_ms];
        let recallValues = [baseline.avg_recall_at_k, letta.avg_recall_at_k];

        if (mem0) {
            latencyValues.push(mem0.avg_latency_ms);
            recallValues.push(mem0.avg_recall_at_k);
        }
        if (mem0_enhanced) {
            latencyValues.push(mem0_enhanced.avg_latency_ms);
            recallValues.push(mem0_enhanced.avg_recall_at_k);
        }

        const maxLatency = Math.max(...latencyValues);
        const maxRecall = Math.max(...recallValues);

        const barWidth = 0.5;
        const barDepth = 0.5;

        // Baseline Bars
        const baselineLatencyHeight = (baseline.avg_latency_ms / maxLatency) * 5;
        const baselineLatencyGeom = new THREE.BoxGeometry(barWidth, baselineLatencyHeight, barDepth);
        const baselineLatencyMat = new THREE.MeshStandardMaterial({ color: 0x0099ff, emissive: 0x0099ff, emissiveIntensity: 0.5 });
        const baselineLatencyBar = new THREE.Mesh(baselineLatencyGeom, baselineLatencyMat);
        baselineLatencyBar.position.set(-3, baselineLatencyHeight / 2 - 2.5, 0);
        resultsGraphic.add(baselineLatencyBar);

        const baselineRecallHeight = (baseline.avg_recall_at_k / maxRecall) * 4;
        const baselineRecallGeom = new THREE.BoxGeometry(barWidth * 0.5, baselineRecallHeight, barDepth * 0.5);
        const baselineRecallMat = new THREE.MeshStandardMaterial({ color: 0x0099ff, metalness: 0.8, roughness: 0.1 });
        const baselineRecallBar = new THREE.Mesh(baselineRecallGeom, baselineRecallMat);
        baselineRecallBar.position.set(-3, baselineRecallHeight / 2 - 2.5, 1);
        resultsGraphic.add(baselineRecallBar);

        // Letta Bars
        const lettaLatencyHeight = (letta.avg_latency_ms / maxLatency) * 5;
        const lettaLatencyGeom = new THREE.BoxGeometry(barWidth, lettaLatencyHeight, barDepth);
        const lettaLatencyMat = new THREE.MeshStandardMaterial({ color: 0x00ff99, emissive: 0x00ff99, emissiveIntensity: 0.5 });
        const lettaLatencyBar = new THREE.Mesh(lettaLatencyGeom, lettaLatencyMat);
        lettaLatencyBar.position.set(-1, lettaLatencyHeight / 2 - 2.5, 0);
        resultsGraphic.add(lettaLatencyBar);

        const lettaRecallHeight = (letta.avg_recall_at_k / maxRecall) * 4;
        const lettaRecallGeom = new THREE.BoxGeometry(barWidth * 0.5, lettaRecallHeight, barDepth * 0.5);
        const lettaRecallMat = new THREE.MeshStandardMaterial({ color: 0x00ff99, metalness: 0.8, roughness: 0.1 });
        const lettaRecallBar = new THREE.Mesh(lettaRecallGeom, lettaRecallMat);
        lettaRecallBar.position.set(-1, lettaRecallHeight / 2 - 2.5, 1);
        resultsGraphic.add(lettaRecallBar);

        // Mem0 Basic Bars (only if available)
        if (mem0) {
            const mem0LatencyHeight = (mem0.avg_latency_ms / maxLatency) * 5;
            const mem0LatencyGeom = new THREE.BoxGeometry(barWidth, mem0LatencyHeight, barDepth);
            const mem0LatencyMat = new THREE.MeshStandardMaterial({ color: 0xff9900, emissive: 0xff9900, emissiveIntensity: 0.5 });
            const mem0LatencyBar = new THREE.Mesh(mem0LatencyGeom, mem0LatencyMat);
            mem0LatencyBar.position.set(1, mem0LatencyHeight / 2 - 2.5, 0);
            resultsGraphic.add(mem0LatencyBar);

            const mem0RecallHeight = (mem0.avg_recall_at_k / maxRecall) * 4;
            const mem0RecallGeom = new THREE.BoxGeometry(barWidth * 0.5, mem0RecallHeight, barDepth * 0.5);
            const mem0RecallMat = new THREE.MeshStandardMaterial({ color: 0xff9900, metalness: 0.8, roughness: 0.1 });
            const mem0RecallBar = new THREE.Mesh(mem0RecallGeom, mem0RecallMat);
            mem0RecallBar.position.set(1, mem0RecallHeight / 2 - 2.5, 1);
            resultsGraphic.add(mem0RecallBar);
        }

        // Mem0 Enhanced Bars (only if available)
        if (mem0_enhanced) {
            const mem0EnhancedLatencyHeight = (mem0_enhanced.avg_latency_ms / maxLatency) * 5;
            const mem0EnhancedLatencyGeom = new THREE.BoxGeometry(barWidth, mem0EnhancedLatencyHeight, barDepth);
            const mem0EnhancedLatencyMat = new THREE.MeshStandardMaterial({ color: 0xff0099, emissive: 0xff0099, emissiveIntensity: 0.5 });
            const mem0EnhancedLatencyBar = new THREE.Mesh(mem0EnhancedLatencyGeom, mem0EnhancedLatencyMat);
            mem0EnhancedLatencyBar.position.set(3, mem0EnhancedLatencyHeight / 2 - 2.5, 0);
            resultsGraphic.add(mem0EnhancedLatencyBar);

            const mem0EnhancedRecallHeight = (mem0_enhanced.avg_recall_at_k / maxRecall) * 4;
            const mem0EnhancedRecallGeom = new THREE.BoxGeometry(barWidth * 0.5, mem0EnhancedRecallHeight, barDepth * 0.5);
            const mem0EnhancedRecallMat = new THREE.MeshStandardMaterial({ color: 0xff0099, metalness: 0.8, roughness: 0.1 });
            const mem0EnhancedRecallBar = new THREE.Mesh(mem0EnhancedRecallGeom, mem0EnhancedRecallMat);
            mem0EnhancedRecallBar.position.set(3, mem0EnhancedRecallHeight / 2 - 2.5, 1);
            resultsGraphic.add(mem0EnhancedRecallBar);
        }
        
        // Add text labels
        const fontLoader = new THREE.FontLoader();
        fontLoader.load('https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/fonts/helvetiker_regular.typeface.json', function (font) {
            const textMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
            
            const baselineTextGeo = new THREE.TextGeometry('Baseline', { font: font, size: 0.2, height: 0.1 });
            const baselineTextMesh = new THREE.Mesh(baselineTextGeo, textMaterial);
            baselineTextMesh.position.set(-3.5, -3.5, 0);
            resultsGraphic.add(baselineTextMesh);

            const lettaTextGeo = new THREE.TextGeometry('Letta', { font: font, size: 0.2, height: 0.1 });
            const lettaTextMesh = new THREE.Mesh(lettaTextGeo, textMaterial);
            lettaTextMesh.position.set(-1.5, -3.5, 0);
            resultsGraphic.add(lettaTextMesh);

            if (mem0) {
                const mem0BasicTextGeo = new THREE.TextGeometry('Mem0 Basic', { font: font, size: 0.2, height: 0.1 });
                const mem0BasicTextMesh = new THREE.Mesh(mem0BasicTextGeo, textMaterial);
                mem0BasicTextMesh.position.set(0.5, -3.5, 0);
                resultsGraphic.add(mem0BasicTextMesh);
            }

            if (mem0_enhanced) {
                const mem0EnhancedTextGeo = new THREE.TextGeometry('Mem0 Enhanced', { font: font, size: 0.2, height: 0.1 });
                const mem0EnhancedTextMesh = new THREE.Mesh(mem0EnhancedTextGeo, textMaterial);
                mem0EnhancedTextMesh.position.set(2.5, -3.5, 0);
                resultsGraphic.add(mem0EnhancedTextMesh);
            }

            const latencyTextGeo = new THREE.TextGeometry('Latency', { font: font, size: 0.4, height: 0.1 });
            const latencyTextMesh = new THREE.Mesh(latencyTextGeo, textMaterial);
            latencyTextMesh.position.set(-4, 3, 0);
            resultsGraphic.add(latencyTextMesh);

            const recallTextGeo = new THREE.TextGeometry('Recall', { font: font, size: 0.4, height: 0.1 });
            const recallTextMesh = new THREE.Mesh(recallTextGeo, textMaterial);
            recallTextMesh.position.set(2, 3, 1);
            resultsGraphic.add(recallTextMesh);

            // Add recall scale on the right (Z=1, same as recall bars)
            // Recall bars have max height of 4, starting from Y=-2.5
            const recallScaleX = 4.5;
            const recallScaleValues = [0.25, 0.50, 0.75, 1.0];
            recallScaleValues.forEach(value => {
                const y = -2.5 + (value * 4);
                const scaleTextGeo = new THREE.TextGeometry(value.toFixed(2), { font: font, size: 0.15, height: 0.05 });
                const scaleTextMesh = new THREE.Mesh(scaleTextGeo, textMaterial);
                scaleTextMesh.position.set(recallScaleX, y, 1);
                resultsGraphic.add(scaleTextMesh);
            });

            // Add latency scale on the left (Z=0, same as latency bars)
            // Latency bars have max height of 5, starting from Y=-2.5
            const latencyScaleX = -5.5;
            const latencyScaleValues = [1000, 2000, 3000, 4000, 5000];
            latencyScaleValues.forEach(value => {
                const y = -2.5 + (value / maxLatency * 5);
                const scaleTextGeo = new THREE.TextGeometry(value.toString(), { font: font, size: 0.15, height: 0.05 });
                const scaleTextMesh = new THREE.Mesh(scaleTextGeo, textMaterial);
                scaleTextMesh.position.set(latencyScaleX, y, 0);
                resultsGraphic.add(scaleTextMesh);
            });
        });

        resultsGraphic.rotation.x = -0.2;
    }

    function resetScene() {
        crystal.visible = true;
        resultsGraphic.visible = false;
        resultsLog.innerHTML = '';
        summaryEl.innerHTML = '';
        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = true;
        setCrystalColor(originalColor);
    }

    async function checkServerStatus() {
        try {
            const response = await fetch(`${API_BASE}/api/eval/status`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    async function startEvaluation() {
        const isServerRunning = await checkServerStatus();
        if (!isServerRunning) {
            modal.style.display = 'flex';
            return;
        }

        resetScene();
        isPulsing = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        progressLog.innerHTML = ''; // Clear progress log
        resultsLog.innerHTML = '<p>Connecting to evaluation server...</p>';
        summaryEl.innerHTML = '';
        setCrystalColor(originalColor);

        fetch(`${API_BASE}/api/eval/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: 'openai',
                openai_model: 'gpt-4o-mini',
                top_k: 5
            })
        })
            .then(response => response.json())
            .then(data => {
                console.log('Evaluation started:', data);
                streamResults();
            })
            .catch(err => {
                console.error('Error starting evaluation:', err);
                resultsLog.innerHTML = `<p style="color: red;">Error starting evaluation.</p>`;
                isPulsing = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
            });
    }

    function stopEvaluation() {
        fetch(`${API_BASE}/api/eval/stop`, { method: 'POST' });
        if (eventSource) {
            eventSource.close();
        }
        isPulsing = false;
        stopTextAnimation();
        setCrystalColor(originalColor);
        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = crystal.visible;
    }

    function streamResults() {
        eventSource = new EventSource(`${API_BASE}/api/eval/stream`);

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleEvent(data);
        };

        eventSource.onerror = () => {
            resultsLog.innerHTML += `<p style="color: red;">Connection to server lost.</p>`;
            eventSource.close();
            isPulsing = false;
            stopTextAnimation();
            setCrystalColor(originalColor);
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };
    }

    function handleEvent(data) {
        switch (data.type) {
            case 'connected':
                setCrystalColor(connectedColor);
                startTextAnimation('Connected. Initializing system');
                break;
            case 'system_loading':
                stopTextAnimation();
                isPulsing = true;
                resultsLog.innerHTML = '<p>System loading... Please wait.</p>';
                break;
            case 'system_loaded':
                isPulsing = false;
                stopTextAnimation();
                setCrystalColor(originalColor);
                resultsLog.innerHTML = '<p>System ready. Starting queries...</p>';
                break;
            case 'evaluation_started':
                stopTextAnimation();
                resultsLog.innerHTML = `<p>Evaluation running for ${data.total_queries} queries.</p>`;
                break;
            case 'query_completed':
                isPulsing = false;
                stopTextAnimation();
                setCrystalColor(originalColor);
                const { query, baseline, letta, mem0, mem0_enhanced } = data;
                createBeam(0x0099ff, baseline.latency_ms); // Blue for baseline
                createBeam(0x00ff99, letta.latency_ms); // Green for letta

                // Build result HTML
                let resultHTML = `<p><strong>Query:</strong> ${query}</p>`;
                resultHTML += `<p><span style="color: #0099ff;">Baseline</span>: ${baseline.latency_ms.toFixed(0)}ms (Recall: ${baseline.recall_at_k.toFixed(2)})</p>`;

                if (mem0) {
                    resultHTML += `<p><span style="color: #ff9900;">Mem0 (Basic)</span>: ${mem0.latency_ms.toFixed(0)}ms (Recall: ${mem0.recall_at_k.toFixed(2)})</p>`;
                }

                if (mem0_enhanced) {
                    resultHTML += `<p><span style="color: #ff0099;">Mem0 (Enhanced)</span>: ${mem0_enhanced.latency_ms.toFixed(0)}ms (Recall: ${mem0_enhanced.recall_at_k.toFixed(2)})</p>`;
                }

                resultHTML += `<p><span style="color: #00ff99;">Letta</span>: ${letta.latency_ms.toFixed(0)}ms (Recall: ${letta.recall_at_k.toFixed(2)})</p>`;

                const newItem = document.createElement('div');
                newItem.classList.add('result-item');
                newItem.innerHTML = resultHTML;
                resultsLog.prepend(newItem);
                break;
            case 'evaluation_completed':
                const { aggregate_metrics, duration_seconds } = data;

                // Build summary HTML
                let summaryHTML = `
                    <h3>Evaluation Complete</h3>
                    <p>Total Duration: ${duration_seconds.toFixed(1)}s</p>
                    <p><strong><span style="color: #0099ff;">Baseline</span> Avg:</strong> ${aggregate_metrics.baseline.avg_latency_ms.toFixed(0)}ms (Recall: ${aggregate_metrics.baseline.avg_recall_at_k.toFixed(2)})</p>
                `;

                if (aggregate_metrics.mem0) {
                    summaryHTML += `<p><strong><span style="color: #ff9900;">Mem0 (Basic)</span> Avg:</strong> ${aggregate_metrics.mem0.avg_latency_ms.toFixed(0)}ms (Recall: ${aggregate_metrics.mem0.avg_recall_at_k.toFixed(2)})</p>`;
                }

                if (aggregate_metrics.mem0_enhanced) {
                    summaryHTML += `<p><strong><span style="color: #ff0099;">Mem0 (Enhanced)</span> Avg:</strong> ${aggregate_metrics.mem0_enhanced.avg_latency_ms.toFixed(0)}ms (Recall: ${aggregate_metrics.mem0_enhanced.avg_recall_at_k.toFixed(2)})</p>`;
                }

                summaryHTML += `<p><strong><span style="color: #00ff99;">Letta</span> Avg:</strong> ${aggregate_metrics.letta.avg_latency_ms.toFixed(0)}ms (Recall: ${aggregate_metrics.letta.avg_recall_at_k.toFixed(2)})</p>`;

                summaryEl.innerHTML = summaryHTML;
                createResultsGraphic(aggregate_metrics);
                stopEvaluation();
                resetBtn.disabled = false;
                break;
            case 'evaluation_stopped':
                 resultsLog.innerHTML += `<p>Evaluation stopped by user.</p>`;
                 stopEvaluation();
                 break;
            case 'log':
                const logEntry = document.createElement('div');
                logEntry.classList.add('log-entry');
                logEntry.textContent = data.message;
                progressLog.appendChild(logEntry);
                // Auto-scroll to bottom
                progressLog.scrollTop = progressLog.scrollHeight;
                break;
        }
    }
    
    closeModalBtn.addEventListener('click', () => {
        modal.style.display = 'none';
        startBtn.disabled = false;
        stopBtn.disabled = true;
    });

    overviewBtn.addEventListener('click', async () => {
        const isServerRunning = await checkServerStatus();
        if (!isServerRunning) {
            modal.style.display = 'flex';
            return;
        }
        handleOverviewButtonClick();
    });

    startBtn.addEventListener('click', startEvaluation);
    stopBtn.addEventListener('click', stopEvaluation);
    resetBtn.addEventListener('click', resetScene);

    // Add mouse interaction for 3D graph
    sceneContainer.addEventListener('mousedown', (e) => {
        if (e.button === 0) { // Left click
            isGraphSpinning = !isGraphSpinning; // Toggle spinning
        }
    });

    initThree();
});