import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import streamlit as st
import json
import openai
import re
from urllib.parse import urlparse
import datetime

if ("OPENAI_API_KEY" not in st.secrets and 
    "TOGETHER_API_KEY" not in st.secrets and
    "GEMINI_API_KEY" not in st.secrets):
    st.error("No API key set. Please configure at least one of: OPENAI_API_KEY, TOGETHER_API_KEY, GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

VALID_SCHEMES = ["http", "https"]

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme in VALID_SCHEMES, result.netloc])
    except Exception:
        return False

def clean_text_with_sup_sub(soup):
    for tag in soup.find_all(['sup', 'sub']):
        tag.unwrap()
    return soup.get_text(strip=True, separator=' ')

def fetch_and_summarize(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; WebScraperBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2'])]
        paragraphs = [clean_text_with_sup_sub(p) for p in soup.find_all('p')][:5]
        links = [(a.get_text(strip=True), a.get('href')) for a in soup.find_all('a', href=True)][:5]

        data = {
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "links": links
        }

        return data
    except Exception as e:
        return {"error": str(e)}

def summarize_with_gpt(data, selected_model):
    combined = "\n\n".join(data.get("paragraphs", []))
    prompt = f"""
You are an AI assistant. Read the content below and generate a categorized summary.
Include categories like Introduction, Key Ideas, Insights, and Recommendations (if any).

Content:
{combined}
"""

    try:
        if selected_model.startswith("gpt") and "OPENAI_API_KEY" in st.secrets:
            response = openai.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You summarize and categorize webpage content."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip(), "OpenAI: " + selected_model

        elif selected_model.startswith("mixtral") and "TOGETHER_API_KEY" in st.secrets:
            headers = {
                "Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": [
                    {"role": "system", "content": "You summarize and categorize webpage content."},
                    {"role": "user", "content": prompt}
                ]
            }
            res = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip(), "Together.ai: Mixtral"

        elif selected_model.startswith("gemini") and "GEMINI_API_KEY" in st.secrets:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={st.secrets['GEMINI_API_KEY']}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            res = requests.post(url, headers=headers, json=payload)
            res.raise_for_status()
            return res.json()['candidates'][0]['content']['parts'][0]['text'].strip(), "Google: Gemini"

    except Exception as e:
        return f"Model error: {e}", selected_model

    return "No available summarization model succeeded.", "None"

st.set_page_config(page_title="Web Scraper Summary Tool", page_icon="🕷️")
st.title("🕷️ Web Scraper Summary Tool")

url = st.text_input("Enter a webpage URL:", "https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage")
model_choice = st.selectbox("Choose AI Model", [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "mixtral-8x7b",
    "gemini-pro"
])

if st.button("Summarize"):
    if not is_valid_url(url):
        st.error("Please enter a valid URL starting with http:// or https://")
        st.stop()

    result = fetch_and_summarize(url)

    if "error" in result:
        st.error(f"Failed to fetch content: {result['error']}")
    else:
        st.subheader("🔍 Extracted Overview")
        st.write(f"**Title:** {result['title']}")
        st.write("**Headings:**", result['headings'])
        st.write("**Links:**", result['links'])

        with st.spinner("Generating summary using AI model..."):
            ai_summary, used_model = summarize_with_gpt(result, model_choice)

        st.subheader("🧠 Categorized Summary")
        st.text_area("Summary", ai_summary, height=400)

        st.write(f"🧾 Model used: {used_model}")

        result["summary"] = ai_summary
        result["model_used"] = used_model
        result["timestamp"] = datetime.datetime.now().isoformat()

        st.download_button("📥 Download JSON", data=json.dumps(result, indent=4, ensure_ascii=False), file_name="summary.json")
