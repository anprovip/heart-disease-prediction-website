document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    // Collecting form data
    const weight = document.getElementById('weight').value;
    const height = document.getElementById('height').value;
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const bloodPressure = document.getElementById('blood-pressure').value;
    const smokingStatus = document.getElementById('smoking-status').value;

    // Creating a data object
    const data = {
        weight: weight,
        height: height,
        age: age,
        gender: gender,
        bloodPressure: bloodPressure,
        smokingStatus: smokingStatus
    };

    // Sending the data to the backend (assuming an endpoint exists)
    fetch('YOUR_BACKEND_ENDPOINT_URL', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        // Displaying the result
        const resultDiv = document.getElementById('result');
        resultDiv.textContent = result.prediction ? "You may have a heart disease" : "You are less likely to have a heart disease";
        resultDiv.style.color = result.prediction ? "#e74c3c" : "#2ecc71";
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
