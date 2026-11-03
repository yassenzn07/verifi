// ربط البحث بسيرفر بايثون
async function handleSearch() {
    const query = document.getElementById('searchInput').value.trim();
    const resultDiv = document.getElementById('searchResult');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = "⏳ جاري البحث...";

    const response = await fetch('/search-claim', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
    });
    const data = await response.json();
    resultDiv.innerHTML = "<strong>Result:</strong> " + data.answer;
}

// ربط كشف الذكاء الاصطناعي بنموذج Hugging Face عبر بايثون
async function handleAIDetection() {
    const text = document.getElementById('aiText').value.trim();
    const resultDiv = document.getElementById('aiResult');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = "⏳ جاري فحص النص بواسطة نموذج الذكاء الاصطناعي...";

    const response = await fetch('/detect-ai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    });
    const data = await response.json();
    
    if(data.status === 'success') {
        resultDiv.innerHTML = `<strong>🤖 AI Result:</strong><br> Verdict: ${data.verdict}<br> Confidence: ${data.confidence}`;
    } else {
        resultDiv.innerHTML = data.message;
    }
}