from flask import Flask, render_template, request
import json
import requests
import google.generativeai as genai

# === API KEYS ===
GEMINI_API_KEY = "AIzaSyChEFT8K6r2OEw-4xy8Sn-i8wyIcA4q7Qo"
SERPAPI_KEY = "4dadc6427791873a3bc55ca0945a43c308ab9d3b4a821811845a5d09929ba8c6"

# === SETUP GEMINI ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# === FLASK SETUP ===
app = Flask(__name__)

def fetch_price(item, district, city):
    query = f"{item} price in {district}, {city}, Türkiye"
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "tr",
        "gl": "tr",
    }
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()

    try:
        shopping_results = data.get("shopping_results", [])
        if shopping_results:
            top_result = shopping_results[0]
            return {
                "item": item,
                "title": top_result.get("title"),
                "price": top_result.get("price"),
                "source": top_result.get("source"),
                "link": top_result.get("link")
            }
        else:
            return {"item": item, "error": "No results"}
    except Exception as e:
        return {"item": item, "error": str(e)}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        city = request.form["city"].strip()
        district = request.form["district"].strip()
        business_type = request.form["business_type"].strip()

        business_info = {
            "city": city,
            "district": district,
            "business_type": business_type
        }

        # === Gemini Prompt ===
        system_prompt = (
            "You're a business advisor AI. Based on the following JSON input, return a list of "
            "5 essential physical items or equipment needed to open the given business type in the given Turkish location. "
            "Only output a plain list of items, one per line, no explanations.\n\n"
            f"JSON input:\n{json.dumps(business_info, indent=2)}"
        )

        response = model.generate_content(system_prompt)
        items = [line.strip("-• ").strip() for line in response.text.strip().split("\n") if line.strip()]
        results = [fetch_price(item, district, city) for item in items]

        return render_template("results.html", items=results, city=city, district=district, business_type=business_type)

    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
