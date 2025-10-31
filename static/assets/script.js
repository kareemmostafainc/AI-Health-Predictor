// ==========================================================
// AI Health Predictor - Frontend Script
// Author: Kareem Mostafa
// ==========================================================

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictionForm");
    const submitBtn = document.getElementById("submitBtn");
    const loader = document.getElementById("loadingSpinner");

    if (!form) return;

    // Hide loader by default
    if (loader) loader.style.display = "none";

    // Handle form submission
    form.addEventListener("submit", async function (e) {
        e.preventDefault();

        if (submitBtn) submitBtn.disabled = true;
        if (loader) loader.style.display = "block";

        try {
            const formData = new FormData(form);
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Prediction request failed.");
            }

            // Redirect to result page (Flask handles rendering)
            window.location.href = "/result";
        } catch (error) {
            console.error(error);
            alert("An error occurred during prediction. Please try again.");
        } finally {
            if (loader) loader.style.display = "none";
            if (submitBtn) submitBtn.disabled = false;
        }
    });
});
