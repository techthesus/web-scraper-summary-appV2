import requests
from bs4 import BeautifulSoup
import streamlit as st
import json

def fetch_and_summarize(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2'])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')][:5]
        links = [(a.get_text(strip=True), a.get('href')) for a in soup.find_all('a', href=True)][:5]

        data = {
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "links": links
        }

        summary = f"""
🔎 Title: {title}

📟 Headings:
- {"\n- ".join(headings) if headings else "None"}

📄 Paragraphs:
- {"\n- ".join(paragraphs) if paragraphs else "None"}

🔗 Links:
- {"\n- ".join([f"{text} ({href})" for text, href in links]) if links else "None"}
"""

        return summary, data
    except Exception as e:
        return f"Error fetching or parsing URL: {e}", {}

st.title("🕷️ Web Scraper Summary Tool")

url = st.text_input("Enter a webpage URL:", "https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage")

if st.button("Summarize"): 
    summary, data = fetch_and_summarize(url)
    st.text_area("Summary", summary, height=300)
    st.download_button("Download JSON", data=json.dumps(data, indent=4, ensure_ascii=False), file_name="summary.json")
