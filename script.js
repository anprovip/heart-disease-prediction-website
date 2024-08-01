document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const height = document.getElementById('height').value;
    const weight = document.getElementById('weight').value;
    const ap_hi = document.getElementById('ap_hi').value;
    const ap_lo = document.getElementById('ap_lo').value;
    const cholesterol = document.getElementById('cholesterol').value;
    const gluc = document.getElementById('gluc').value;
    const smoke = document.getElementById('smoke').value;
    const alco = document.getElementById('alco').value;
    const active = document.getElementById('active').value;

    const data = {
        age: parseInt(age),
        gender: parseInt(gender),
        height: parseInt(height),
        weight: parseFloat(weight),
        ap_hi: parseInt(ap_hi),
        ap_lo: parseInt(ap_lo),
        cholesterol: parseInt(cholesterol),
        gluc: parseInt(gluc),
        smoke: parseInt(smoke),
        alco: parseInt(alco),
        active: parseInt(active)
    };

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        const resultDiv = document.getElementById('result');
        if (result.prediction) {
            resultDiv.textContent = "You may have a risk of heart disease";
            resultDiv.style.color = "#e74c3c";
        } else {
            resultDiv.textContent = "You are less likely to have a heart disease";
            resultDiv.style.color = "#2ecc71";
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
