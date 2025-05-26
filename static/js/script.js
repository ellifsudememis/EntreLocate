let map;
let chatSessionId = null;

const citySelect = document.getElementById("city");
const districtSelect = document.getElementById("district");
const businessTypeInput = document.getElementById("business_type");

document.addEventListener('DOMContentLoaded', function() {
    map = L.map("map").setView([20, 0], 2);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    setTimeout(() => {
        map.invalidateSize();
    }, 300);

    loadCities('/static/data/cities_with_districts.json');

    const categoryHelpBtn = document.getElementById('categoryHelp');
    const categorySuggestions = document.getElementById('categorySuggestions');

    if (categoryHelpBtn) {
        categoryHelpBtn.addEventListener('click', function() {
            const chatBox = document.getElementById("chatResponse");
            chatBox.innerHTML += `<div class="message bot-message">
                <strong>Need help choosing a category?</strong><br>
                👉 Just type a specific keyword (like cafe or bakery) in the category input field and click Search!
            </div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        });
    }
});

async function loadCities(filePath) {
    citySelect.innerHTML = '<option value="">Loading cities...</option>';
    try {
        const response = await fetch(filePath);
        const citiesData = await response.json();
        citySelect.innerHTML = '<option value="">Select a city</option>';
        if (typeof citiesData === 'object' && !Array.isArray(citiesData)) {
            Object.keys(citiesData).forEach(cityName => {
                const option = document.createElement('option');
                option.value = cityName;
                option.textContent = cityName;
                citySelect.appendChild(option);
            });
        } else {
            console.error('Invalid JSON format for cities.');
            citySelect.innerHTML = '<option value="">Error loading cities</option>';
        }
    } catch (error) {
        console.error('Error fetching cities:', error);
        citySelect.innerHTML = '<option value="">Error loading cities</option>';
    }
}

async function loadDistricts(selectedCity) {
    const queryUrl = `/get_districts?city=${encodeURIComponent(selectedCity)}`;
    districtSelect.innerHTML = '<option value="">Loading districts...</option>';
    districtSelect.disabled = true;

    if (!selectedCity) {
        districtSelect.innerHTML = '<option value="">Select a city first</option>';
        return;
    }

    try {
        const response = await fetch(queryUrl);
        const districts = await response.json();

        districtSelect.innerHTML = '<option value="">Select a district</option>';
        if (Array.isArray(districts)) {
            districts.forEach(districtName => {
                const option = document.createElement("option");
                option.value = districtName;
                option.textContent = districtName;
                districtSelect.appendChild(option);
            });
            districtSelect.disabled = false;
        } else if (districts && districts.error) {
            console.error('Error from server:', districts.error);
            districtSelect.innerHTML = `<option value="">Error: ${districts.error}</option>`;
        } else {
            console.error('Invalid response format for districts.');
            districtSelect.innerHTML = '<option value="">Error loading districts</option>';
        }
    } catch (error) {
        console.error('Error fetching districts:', error);
        districtSelect.innerHTML = '<option value="">Error loading districts</option>';
    }
}

citySelect.addEventListener("change", () => {
    const selectedCity = citySelect.value;
    if (selectedCity) {
        loadDistricts(selectedCity);
    } else {
        districtSelect.innerHTML = '<option value="">Select a city first</option>';
        districtSelect.disabled = true;
    }
});

document.getElementById("searchForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const city = citySelect.value;
    const district = districtSelect.value;
    const business_type = businessTypeInput.value.trim();

    if (!city || !district || !business_type) {
        alert("Please fill all fields");
        return;
    }

    const submitButton = this.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    submitButton.textContent = "Searching...";
    submitButton.disabled = true;

    try {
        const headers = { "Content-Type": "application/json" };
        if (chatSessionId) headers['X-Session-ID'] = chatSessionId;

        const response = await fetch("/get_businesses", {
            method: "POST",
            headers: headers,
            body: JSON.stringify({ city, district, business_type })
        });

        const data = await response.json();
        submitButton.textContent = originalText;
        submitButton.disabled = false;

        if (data.error) return alert(data.error);

        const lat = data.lat, lon = data.lon;
        if (map) {
            map.eachLayer(layer => { if (layer instanceof L.Marker) map.removeLayer(layer); });
            map.setView([lat, lon], 14);
        }

        if (Array.isArray(data.data)) {
            if (data.data.length === 0) return alert("No businesses found matching your criteria.");

            const chatBox = document.getElementById("chatResponse");
            chatBox.innerHTML += `<div class="message bot-message">I found ${data.data.length} businesses matching your search for ${business_type} in ${district}, ${city}. Check the map!</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            data.data.forEach(item => {
                if (item.lat && item.lon) {
                    const popup = item.tags?.name ? `<strong>${item.tags.name}</strong>` : "Unnamed Business";
                    L.marker([item.lat, item.lon]).addTo(map).bindPopup(popup);
                }
            });

            setBusinessContext(business_type, city, district);
        }
    } catch (err) {
        submitButton.textContent = originalText;
        submitButton.disabled = false;
        alert("Failed to fetch data from server. Please try again.");
        console.error(err);
    }
});

async function sendChat() {
    const input = document.getElementById("chatInput");
    const message = input.value.trim();
    if (!message) return;

    const chatBox = document.getElementById("chatResponse");
    chatBox.innerHTML += `<div class="message user-message">${escapeHtml(message)}</div>`;
    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'message bot-message';
    typingIndicator.id = 'typing-indicator';
    typingIndicator.textContent = 'Typing...';
    chatBox.appendChild(typingIndicator);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const headers = { "Content-Type": "application/json" };
        if (chatSessionId) headers['X-Session-ID'] = chatSessionId;

        const response = await fetch('/chat', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                message: message,
                city: citySelect.value,
                district: districtSelect.value,
                business_type: businessTypeInput.value.trim()
            })
        });

        const data = await response.json();
        const indicator = document.getElementById('typing-indicator');
        if (indicator) chatBox.removeChild(indicator);

        const formattedResponse = formatChatResponse(data.response);
        chatBox.innerHTML += `<div class="message bot-message">${formattedResponse}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;

        if (data.session_id) chatSessionId = data.session_id;
    } catch (err) {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) chatBox.removeChild(indicator);
        chatBox.innerHTML += `<div class="message bot-message">Sorry, I couldn't process your request right now. Please try again later.</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
        console.error(err);
    }
}

document.getElementById("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChat();
    }
});

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatChatResponse(text) {
    return escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/^## (.*?)$/gm, "<h3>$1</h3>")
        .replace(/^### (.*?)$/gm, "<h4>$1</h4>")
        .replace(/^\* (.*?)$/gm, "<li>$1</li>")
        .replace(/\n{2,}/g, "<br><br>")
        .replace(/\n/g, "<br>");
}

const sidebar = document.getElementById("sidebar");
const content = document.getElementById("content");
const toggleBtn = document.getElementById("sidebarToggle");

toggleBtn.addEventListener("click", () => {
    sidebar.classList.toggle("closed");
    content.classList.toggle("expanded");
    setTimeout(() => { if (map) map.invalidateSize(); }, 350);
});

window.addEventListener('resize', () => { if (map) map.invalidateSize(); });

document.getElementById("downloadBtn").addEventListener("click", () => {
    window.location.href = "/download";
});