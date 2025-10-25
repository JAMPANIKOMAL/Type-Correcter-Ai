/*
 * Client-side JavaScript for the Type Correcter Ai GUI.
 * This script handles button clicks and API calls to the Flask backend.
 */

// Wait for the entire HTML document to be loaded and parsed.
document.addEventListener('DOMContentLoaded', () => {

    // Get references to the DOM elements we need to interact with
    const correctButton = document.getElementById('correct-button');
    const inputText = document.getElementById('input-text');
    const outputText = document.getElementById('output-text');
    const loadingSpinner = document.getElementById('loading-spinner');

    // This is the main function that talks to our Python backend
    const getCorrection = async () => {
        const text = inputText.value;

        // Don't do anything if the input is empty
        if (!text.trim()) {
            outputText.value = '';
            return;
        }

        // --- 1. Show loading state (Good UX) ---
        loadingSpinner.style.display = 'block';
        correctButton.disabled = true;
        outputText.value = 'Correcting...';

        try {
            // --- 2. Send the text to the Flask backend ---
            // We use 'fetch' to send a POST request to our '/predict' endpoint
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Convert the JavaScript object to a JSON string
                body: JSON.stringify({ text: text }),
            });

            if (!response.ok) {
                // Handle server errors (e.g., 500 internal server error)
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // --- 3. Get the JSON response ---
            const data = await response.json();

            // --- 4. Display the result ---
            if (data.correction) {
                outputText.value = data.correction;
            } else if (data.error) {
                // Display any errors sent from the server
                outputText.value = `Error: ${data.error}`;
            }

        } catch (error) {
            // Handle network errors (e.g., server is down)
            console.error('Fetch error:', error);
            outputText.value = 'An error occurred. Check the server console.';
        } finally {
            // --- 5. Hide loading state (always runs) ---
            loadingSpinner.style.display = 'none';
            correctButton.disabled = false;
        }
    };

    // Attach the event listener to the button
    correctButton.addEventListener('click', getCorrection);
    
    // Bonus: Add a Ctrl+Enter shortcut for convenience
    inputText.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            getCorrection();
        }
    });
});