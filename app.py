import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import streamlit as st
import json
import openai
import re
from urllib.parse import urlparse, urljoin
import datetime
import textwrap
from collections import deque

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

def score_link(link_text, url_path):
    score = 0
    score += len(link_text) * 0.5
    score -= url_path.strip('/').count('/') * 2
    keywords = ["insight", "ai", "future", "strategy", "tech", "digital"]
    if any(k in link_text.lower() for k in keywords):
        score += 10
    return score

show_link_scores = st.checkbox("Show link scores", value=False)

def fetch_and_summarize(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; WebScraperBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        paragraphs = [clean_text_with_sup_sub(p) for p in soup.find_all('p')][:10]

        raw_links = [(a.get_text(strip=True), urljoin(url, a.get('href'))) for a in soup.find_all('a', href=True)]
        domain = urlparse(url).netloc
        internal_links = [(text, link) for text, link in raw_links if urlparse(link).netloc == domain or link.startswith('/')]
        ranked_links = [(text, link, score_link(text, urlparse(link).path)) for text, link in internal_links]
        ranked_links = sorted(ranked_links, key=lambda x: x[2], reverse=True)
        top_links = [(text, link) if not show_link_scores else (f"{text} (Score: {score})", link) for text, link, score in ranked_links[:10]]

        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2'])]
        html_content = soup.prettify()

        data = {
            "url": url,
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "links": top_links,
            "html": html_content
        }

        return data
    except Exception as e:
        return {"error": str(e)}

# The rest of your existing code remains unchanged...
