// ==========================================================
// AI Health Predictor - Frontend Interaction Script
// Author: Kareem Mostafa
// ==========================================================

// ----------------------------------------------------------
// DOM Elements
// ----------------------------------------------------------
const form = document.getElementById("healthForm");
const resultBox = document.getElementById("resultBox");
const resultTitle = document.getElementById("resultTitle");
const resultValue = document.getElementById("resultValue");
const chartContainer = document.getElementById("chartContainer");

// ----------------------------------------------------------
// Event Listener - Handle Form Submission
// ----------------------------------------------------------
if (form) {
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        resultBox.style.display = "none";

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                throw new Error("Failed to fetch prediction result.");
            }

            const result = await response.json();
            displayResult(result);
        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred while processing your request.");
        }
    });
}

// ----------------------------------------------------------
// Display Prediction Result
// ----------------------------------------------------------
function displayResult(result) {
    const { probability, risk_category, explanation } = result;

    resultTitle.textContent = "Prediction Result";
    resultValue.innerHTML = `
        <strong>Health Risk Probability:</strong> ${(probability * 100).toFixed(2)}%<br>
        <strong>Risk Category:</strong> ${risk_category}
    `;

    resultBox.style.display = "block";
    renderChart(explanation);
}

// ----------------------------------------------------------
// Render Simple Feature Importance Chart
// ----------------------------------------------------------
function renderChart(explanation) {
    if (!explanation || Object.keys(explanation).length === 0) {
        chartContainer.innerHTML = "<p>No explanation data available.</p>";
        return;
    }

    const labels = Object.keys(explanation);
    const values = Object.values(explanation);

    const canvas = document.createElement("canvas");
    chartContainer.innerHTML = "";
    chartContainer.appendChild(canvas);

    new Chart(canvas, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Feature Impact",
                    data: values,
                },
            ],
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });
}
