const API_URL = "http://127.0.0.1:8000/predict_batch";

const analyzeBtn = document.getElementById("analyzeBtn");
const statusElement = document.getElementById("status");
const resultsElement = document.getElementById("results");
const commentsResults = document.getElementById("commentsResults");
const themeBtn = document.getElementById("themeBtn");
const copyBtn = document.getElementById("copyBtn");

let allResults = [];
let currentFilter = "Tous";


analyzeBtn.addEventListener("click", async () => {

    analyzeBtn.disabled = true;
    statusElement.textContent = "Récupération des commentaires...";
    resultsElement.classList.add("hidden");

    try {
        
        const [tab] = await chrome.tabs.query({
            active: true,
            currentWindow: true
        });

        if (!tab || !tab.id || !tab.url ||
            !tab.url.startsWith("https://www.youtube.com/")) {
            throw new Error("Veuillez ouvrir une vidéo YouTube.");
        }

        
        const response = await chrome.tabs.sendMessage(tab.id, {
            action: "GET_COMMENTS"
        });

        const comments = response?.comments || [];

        if (comments.length === 0) {
            throw new Error(
                "Aucun commentaire trouvé. Faites défiler la page YouTube."
            );
        }

        statusElement.textContent =
            `${comments.length} commentaires récupérés. Analyse en cours...`;

        
        const apiResponse = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ comments })
        });

        if (!apiResponse.ok) {
            throw new Error("Erreur lors de l'analyse par l'API.");
        }

        const data = await apiResponse.json();

        if (!Array.isArray(data.results) || !data.statistics) {
            throw new Error("Format de réponse API incorrect.");
        }

        allResults = data.results;
        currentFilter = "Tous";

        updateFilterButtons();
        displayStatistics(data);
        displayComments();

        resultsElement.classList.remove("hidden");

        statusElement.textContent =
            `Analyse terminée en ${data.processing_time_ms} ms.`;

    } catch (error) {
        statusElement.textContent = error.message;
        console.error("Erreur :", error);

    } finally {
        analyzeBtn.disabled = false;
    }
});



function displayStatistics(data) {

    const stats = data.statistics;

    document.getElementById("totalComments").textContent =
        `Total : ${data.total_comments} commentaires`;

    document.getElementById("positiveCount").textContent =
        stats["Positif"]?.count ?? 0;

    document.getElementById("neutralCount").textContent =
        stats["Neutre"]?.count ?? 0;

    document.getElementById("negativeCount").textContent =
        stats["Négatif"]?.count ?? 0;

    document.getElementById("positivePercentage").textContent =
        `${stats["Positif"]?.percentage ?? 0}%`;

    document.getElementById("neutralPercentage").textContent =
        `${stats["Neutre"]?.percentage ?? 0}%`;

    document.getElementById("negativePercentage").textContent =
        `${stats["Négatif"]?.percentage ?? 0}%`;
}



function displayComments() {

    commentsResults.innerHTML = "";

    const filteredResults = currentFilter === "Tous"
        ? allResults
        : allResults.filter(
            result => result.sentiment === currentFilter
        );

    if (filteredResults.length === 0) {
        commentsResults.textContent =
            "Aucun commentaire pour ce sentiment.";
        return;
    }

    filteredResults.forEach(result => {

        const div = document.createElement("div");
        div.className = "comment-result";

        const comment = document.createElement("p");
        comment.className = "comment-text";
        comment.textContent = result.comment;

        const info = document.createElement("div");
        info.className = "comment-info";

        const sentiment = document.createElement("span");
        sentiment.textContent = result.sentiment;

        const confidence = document.createElement("span");
        confidence.textContent =
            `${result.confidence}% de confiance`;

        info.appendChild(sentiment);
        info.appendChild(confidence);

        div.appendChild(comment);
        div.appendChild(info);

        commentsResults.appendChild(div);
    });
}



document.querySelectorAll(".filter-btn").forEach(button => {

    button.addEventListener("click", () => {

        currentFilter = button.dataset.filter;

        updateFilterButtons();
        displayComments();
    });
});

function updateFilterButtons() {

    document.querySelectorAll(".filter-btn").forEach(button => {

        button.classList.toggle(
            "active",
            button.dataset.filter === currentFilter
        );
    });
}



themeBtn.addEventListener("click", () => {

    document.body.classList.toggle("dark");

    const darkMode = document.body.classList.contains("dark");

    themeBtn.textContent = darkMode ? "☀️" : "🌙";
    themeBtn.title = darkMode
        ? "Activer le mode clair"
        : "Activer le mode sombre";
});



copyBtn.addEventListener("click", async () => {

    if (allResults.length === 0) {
        statusElement.textContent = "Aucun résultat à copier.";
        return;
    }

    const textToCopy = [
        "ANALYSE DE SENTIMENT YOUTUBE",
        "-----------------------------",
        "",
        ...allResults.map(result =>
            `Commentaire : ${result.comment}\n` +
            `Sentiment : ${result.sentiment}\n` +
            `Confiance : ${result.confidence}%\n`
        )
    ].join("\n");

    try {
        await navigator.clipboard.writeText(textToCopy);
        statusElement.textContent = "Résultats copiés avec succès !";

    } catch (error) {
        statusElement.textContent = "Impossible de copier les résultats.";
        console.error(error);
    }
});