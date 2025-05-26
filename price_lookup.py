import os
import json
import re
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def get_business_startup_prices():
    city = input("Enter the city (e.g., Istanbul): ").strip()
    district = input("Enter the district (e.g., Kadıköy): ").strip()
    business_type = input("Enter the business type (e.g., coffee shop): ").strip()

    business_info = {
        "city": city,
        "district": district,
        "business_type": business_type
    }

    print("\nGenerated JSON Input:")
    print(json.dumps(business_info, indent=2))

    # === Generate Item List ===
    system_prompt = (
        "You're a business advisor AI. Based on the following JSON input, return a list of "
        "5 essential physical items or equipment needed to open the given business type in the given Turkish location. "
        "Only output a plain list of items, one per line, no explanations.\n\n"
        f"JSON input:\n{json.dumps(business_info, indent=2)}"
    )

    response = model.generate_content(system_prompt)
    items = [line.strip("-• ").strip() for line in response.text.strip().split("\n") if line.strip()]
    print("\nGenerated Items from Gemini:")
    for item in items:
        print(f"- {item}")

    # === Fetch Prices ===
    def fetch_price(item):
        cleaned_item = item.lower()
        cleaned_item = re.sub(r"\b(with|and|for|of|a|an|the|access|system|unit|units)\b", "", cleaned_item)
        cleaned_item = re.sub(r"[^a-zA-Z0-9\sçğıöşüÇĞİÖŞÜ]", "", cleaned_item).strip()

        query = f"{cleaned_item} fiyat Türkiye"
        print(f"\n[DEBUG] Searching for: {query}")

        params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": SERP_API_KEY,
            "hl": "tr",
            "gl": "tr",
        }

        try:
            response = requests.get("https://serpapi.com/search", params=params)
            data = response.json()

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

    results = [fetch_price(item) for item in items]

    # === Display Results ===
    print("\nTop 5 Items and Prices:")
    for result in results:
        if "error" in result:
            print(f"{result['item']}: {result['error']}")
        else:
            print(f"{result['item'].title()}: {result['price']} - {result['title']} ({result['source']})")
            print(f"Link: {result['link']}\n")

    return {
        "city": city,
        "district": district,
        "business_type": business_type,
        "items": items,
        "prices": results
    }

if __name__ == "__main__":
    get_business_startup_prices()
