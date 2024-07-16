document.getElementById('summarizationForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const inputText = document.getElementById('inputText').value;
    const summaryTextElement = document.getElementById('summaryText');
    const errorElement = document.getElementById('error');
    const resultElement = document.getElementById('result');

    summaryTextElement.textContent = '';
    errorElement.textContent = '';
    errorElement.classList.add('hidden');

    if (!inputText.trim()) {
        errorElement.textContent = 'Please enter text to summarize.';
        errorElement.classList.remove('hidden');
        return;
    }

    try {
        const response = await fetch('http://localhost:5000/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
            throw new Error('Failed to summarize text. Please try again later.');
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        summaryTextElement.textContent = data.summary;
        resultElement.classList.remove('hidden');
    } catch (error) {
        errorElement.textContent = error.message;
        errorElement.classList.remove('hidden');
    }
});

document.getElementById('clearButton').addEventListener('click', function() {
    document.getElementById('inputText').value = '';
    document.getElementById('summaryText').textContent = '';
    document.getElementById('error').textContent = '';
    document.getElementById('error').classList.add('hidden');
    document.getElementById('result').classList.add('hidden');
});
