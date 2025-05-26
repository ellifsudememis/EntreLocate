from asyncio.log import logger
import json
import asyncio
import os
import requests
from flask import Flask, render_template, request, jsonify, send_file
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from app2 import Manager
from app2 import format
import google.generativeai  as genai

load_dotenv()
app = Flask(__name__)

with open('static/data/cities_with_districts.json', 'r', encoding='utf-8') as f:
    districts_data = json.load(f)

with open('static/data/business_categories.json', 'r', encoding='utf-8') as f:
    categories_data = json.load(f)


session_managers = {}
business_context = {}


TAG_MAPPING = {
    "cafe": {"amenity": "cafe"},
    "books": {"shop": "books"},
    "lawyer": {"amenity": "lawyer"},
    "dessert": {"shop": "confectionery"},
    "restaurant": {"amenity": "restaurant"},
    "hospital": {"amenity": "hospital"},
    "school": {"amenity": "school"},
    "shopping_mall": {"shop": "mall"},
    "pharmacy": {"amenity": "pharmacy"},
    "optician": {"shop": "optician"},
    "beauty": {"shop": "beauty"},
    "hairdresser": {"amenity": "hairdresser"},
    "childcare": {"amenity": "childcare"},
    "college": {"amenity": "college"},
    "university": {"amenity": "university"},
    "training": {"amenity": "training"},
    "library": {"amenity": "library"},
    "museum": {"tourism": "museum"},
    "cinema": {"amenity": "cinema"},
    "theatre": {"amenity": "theatre"},
    "music": {"shop": "musical_instrument"},
    "games": {"shop": "games"},
    "sports": {"sport": "sports"},
    "pet": {"shop": "pet"},
    "second_hand": {"shop": "second_hand"},
    "art": {"shop": "art"},
    "florist": {"shop": "florist"},
}

def get_osm_type(category_name):
    for item in categories_data:
        if item["category"].lower() == category_name.lower():
            return item["type"]
    return None

def get_top_business_needs_and_prices(category, city, district):
    return {
        "items": ["oven", "flour", "fridge", "staff", "rent"],
        "prices": [
            {"item": "oven", "price": "$1,500", "title": "Industrial Oven"},
            {"item": "flour", "price": "$200", "title": "50kg Pack"},
            {"item": "fridge", "price": "$800"},
            {"item": "staff", "price": "$2,000"},
            {"item": "rent", "price": "$1,200/month"},
        ]
    }

@app.route("/")
def index():
    return render_template("index.html", session_id=os.urandom(8).hex())

@app.route("/get_districts")
def get_districts():
    city = request.args.get("city", "")
    if city in districts_data:
        return jsonify(districts_data[city])
    return jsonify({"error": f"No districts found for city: {city}"}), 404

@app.route("/get_businesses", methods=["POST"])
def get_businesses():
    data = request.get_json()
    city = data.get("city")
    district = data.get("district")
    category = data.get("business_type")
    session_id = request.headers.get('X-Session-ID')

    if not all([city, district, category]):
        return jsonify({"error": "Missing required fields"}), 400

    business_type = get_osm_type(category)
    if not business_type:
        return jsonify({'error': f'Unknown category: {category}'}), 400

    geolocator = Nominatim(user_agent="entrelocate-app")
    location = geolocator.geocode(f"{district}, {city}")
    if not location:
        return jsonify({'error': 'Location not found'}), 404

    lat, lon, radius = location.latitude, location.longitude, 2500
    business_type = business_type.lower().strip()
    query_parts = []

    if business_type in TAG_MAPPING:
        for key, value in TAG_MAPPING[business_type].items():
            query_parts += [
                f'node["{key}"="{value}"](around:{radius},{lat},{lon});',
                f'way["{key}"="{value}"](around:{radius},{lat},{lon});',
                f'relation["{key}"="{value}"](around:{radius},{lat},{lon});',
            ]
    else:
        for key in ["amenity", "shop", "tourism", "leisure", "healthcare"]:
            query_parts += [
                f'node["{key}"="{business_type}"](around:{radius},{lat},{lon});',
                f'way["{key}"="{business_type}"](around:{radius},{lat},{lon});',
                f'relation["{key}"="{business_type}"](around:{radius},{lat},{lon});',
            ]

    query = f"""[out:json][timeout:25];({"".join(query_parts)});out center;"""
    try:
        response = requests.post("https://overpass-api.de/api/interpreter", data=query)
        response.raise_for_status()
        results = response.json()
        elements = results.get("elements", [])
    except Exception as e:
        return jsonify({"error": f"Overpass error: {e}"}), 500

    for el in elements:
        if 'lat' not in el or 'lon' not in el:
            if 'center' in el:
                el['lat'] = el['center']['lat']
                el['lon'] = el['center']['lon']
    elements = [el for el in elements if el.get("lat") and el.get("lon")]

    top_needs_prices = get_top_business_needs_and_prices(category, city, district)
    if session_id:
        business_context[session_id] = {"business_type": category, "top_needs_prices": top_needs_prices}

    # Save results for download
    with open("osm_filtered_businesses.json", "w", encoding="utf-8") as f:
        json.dump(elements, f, ensure_ascii=False, indent=2)

    return jsonify({
        "data": elements,
        "lat": lat,
        "lon": lon,
        "top_needs_prices": top_needs_prices,
        "business_count": len(elements)
    })

@app.route("/set_business_context", methods=["POST"])
def set_business_context():
    data = request.get_json()
    session_id = request.headers.get('X-Session-ID')
    business_type = data.get("business_type")
    city = data.get("city")
    district = data.get("district")

    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400

    result = business_context.get(session_id)
    if result:
        return jsonify({"success": True, **result})
    else:
        top_needs_prices = get_top_business_needs_and_prices(business_type, city, district)
        return jsonify({"success": True, "top_needs_prices": top_needs_prices})

@app.route("/download")
def download():
    filepath = "osm_filtered_businesses.json"
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, as_attachment=True)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    session_id = request.headers.get("X-Session-ID", os.urandom(8).hex())

    if session_id not in session_managers:
        session_managers[session_id] = Manager(user_input)

    manager = session_managers[session_id]
    manager.user_input = user_input

    try:
        response_json = asyncio.run(manager.run())
        try:
            parsed = json.loads(response_json)
            response_html = format(parsed)
        except Exception as parse_error:
            response_html = f"<pre>{response_json}</pre>" 

        return jsonify({
            "response": response_html,
            "session_id": session_id
        })
    except Exception as e:
        logger.exception(f"Error during chat processing: {e}")
        return jsonify({
            "response": f"<p><strong>Internal error:</strong> {str(e)}</p>",
            "session_id": session_id
        })

if __name__ == "__main__":
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        logger.error("Google API key not found in environment variables.")
        exit(1)
        
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

if __name__ == "__main__":
    app.run(debug=True)
