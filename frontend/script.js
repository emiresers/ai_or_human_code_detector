document.getElementById("analyzeBtn").addEventListener("click", async () => {
    const code = document.getElementById("codeInput").value.trim();

    if (!code) {
        alert("Lütfen önce kod girin!");
        return;
    }

    // Butonu devre dışı bırak (çoklu tıklamayı önle)
    const btn = document.getElementById("analyzeBtn");
    btn.disabled = true;
    btn.textContent = "Analiz ediliyor...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code: code })
        });

    const data = await response.json();

    document.getElementById("results").classList.remove("hidden");

        // Progress bar güncelleme fonksiyonu
        function updateProgress(aiPercent, humanPercent, prefix) {
            document.getElementById(`${prefix}AiPercent`).textContent = `${aiPercent.toFixed(1)}%`;
            document.getElementById(`${prefix}HumanPercent`).textContent = `${humanPercent.toFixed(1)}%`;
            document.getElementById(`${prefix}AiBar`).style.width = `${aiPercent}%`;
            document.getElementById(`${prefix}HumanBar`).style.width = `${humanPercent}%`;
        }

        // Logistic Regression
        const lrAi = data.logistic_regression.prob_ai * 100;
        const lrHuman = data.logistic_regression.prob_human * 100;
        updateProgress(lrAi, lrHuman, "lr");

        // Naive Bayes
        const nbAi = data.naive_bayes.prob_ai * 100;
        const nbHuman = data.naive_bayes.prob_human * 100;
        updateProgress(nbAi, nbHuman, "nb");

        // Random Forest
        const rfAi = data.random_forest.prob_ai * 100;
        const rfHuman = data.random_forest.prob_human * 100;
        updateProgress(rfAi, rfHuman, "rf");

        // Ortalama (Genel Özet)
        const avgAi = data.average.prob_ai * 100;
        const avgHuman = data.average.prob_human * 100;
        updateProgress(avgAi, avgHuman, "avg");

        // Final Decision
        const decision = data.final_decision.toUpperCase();
        const emoji = decision === "AI" ? "🤖" : "👤";
        document.getElementById("finalDecision").innerHTML = 
            `${emoji} Final Karar: <span class="decision-${decision.toLowerCase()}">${decision}</span> tarafından yazılmış`;

    } catch (error) {
        alert("Backend'e bağlanılamadı.\nLütfen FastAPI'nin port 8000'de çalıştığından emin olun.");
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = "Kodu Analiz Et";
    }
});
