// Handling input text operations
document.getElementById('clear-btn').addEventListener('click', function() {
    document.getElementById('input-text').value = '';
});

document.getElementById('copy-btn').addEventListener('click', function() {
    const inputText = document.getElementById('input-text');
    inputText.select();
    document.execCommand('copy');
});

document.getElementById('paste-btn').addEventListener('click', function() {
    navigator.clipboard.readText().then(text => {
        document.getElementById('input-text').value = text;
    });
});

// Copy output to clipboard
document.getElementById('copy-output-btn').addEventListener('click', function() {
    const outputText = document.getElementById('output-text');
    outputText.select();
    document.execCommand('copy');
});

// Translation direction toggle
let isMarathiToEnglish = true;
document.getElementById('toggle-button').addEventListener('click', function() {
    isMarathiToEnglish = !isMarathiToEnglish;
    document.getElementById('direction-label').textContent = isMarathiToEnglish ? 'Marathi to English' : 'English to Marathi';
    document.querySelector('.input-section h2').textContent = isMarathiToEnglish ? 'Marathi' : 'English';
    document.querySelector('.output-section h2').textContent = isMarathiToEnglish ? 'English' : 'Marathi';
});

// Handle translation request
document.getElementById('translate-btn').addEventListener('click', function() {
    const inputText = document.getElementById('input-text').value;

    if (!inputText.trim()) {
        alert('Please enter some text to translate.');
        return;
    }

    // Determine the direction: '1' for Marathi to English, '2' for English to Marathi
    const direction = isMarathiToEnglish ? '1' : '2';

    // Show loading spinner
    const loadingIcon = document.getElementById('loading-icon');
    loadingIcon.style.display = 'block';

    // Call the Flask backend to perform the translation
    fetch('/translate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: inputText,
            direction: direction
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('output-text').value = data.translated_text;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('There was an error with the translation.');
    })
    .finally(() => {
        // Hide the loading spinner once the translation is done
        loadingIcon.style.display = 'none';
    });
});
