document.getElementById('requirementForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const requirementInput = document.getElementById('requirementInput').value;
    
    if (requirementInput.trim() === "") {
        alert("Please enter a software requirement.");
        return;
    }

    const response = await fetch('/generate_test_case', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ requirement: requirementInput })
    });

    const data = await response.json();
    document.getElementById('testCase').textContent = JSON.stringify(data);
    
    document.getElementById('output').classList.add('fadeIn');
    setTimeout(() => {
        document.getElementById('output').classList.remove('fadeIn');
    }, 1000);
});
