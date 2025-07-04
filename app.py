import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import streamlit as st
import json
import openai
import re
from urllib.parse import urlparse, urljoin
import urllib.robotparser
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

show_link_scores = st.checkbox("Show link scores", value=False)

def fetch_and_summarize(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; WebScraperBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2'])]
        paragraphs = [clean_text_with_sup_sub(p) for p in soup.find_all('p')][:10]
        raw_links = [(a.get_text(strip=True), urljoin(url, a.get('href'))) for a in soup.find_all('a', href=True)]
        domain = urlparse(url).netloc
        internal_links = [(text, link) for text, link in raw_links if urlparse(link).netloc == domain or link.startswith('/')]
        ranked_links = []
        fallback_keywords = ['insight','ai','tech','future','digital','strategy']
        for text, link in internal_links:
            filtered_by = "not_filtered"
            text_lower = text.lower()
            score = len(text) * 0.5 - urlparse(link).path.strip('/').count('/') * 2
            score += 10 if any(k in text_lower for k in ['insight','ai','tech','future','digital','strategy']) else 0

            use_semantic_filtering = 'link_filter_prompt' in globals() and link_filter_prompt
            if use_semantic_filtering:
                try:
                    llm_prompt = f"Filter this link: '{text}' with URL '{link}' based on this intent: '{link_filter_prompt}'. Reply 'yes' if relevant, 'no' if not."
                    if model_choice.startswith("gpt") and "OPENAI_API_KEY" in st.secrets:
                        response = openai.chat.completions.create(
                            model=model_choice,
                        messages=[
                            {"role": "system", "content": "You evaluate relevance of webpage links."},
                            {"role": "user", "content": llm_prompt}
                        ]
                    )
                    decision = response.choices[0].message.content.strip().lower()
                    if "no" in decision:
                        filtered_by = model_choice
                        st.info(f"✅ Allowed by {model_choice}: {text}")
                        continue
                    else:
                        filtered_by = model_choice
                        continue
                except:
                    filtered_by = "fallback"
                    st.warning(f"⚠️ Fallback used for: {text}")
                    score += 5 if any(k in text_lower for k in fallback_keywords) else 0  # fallback score if GPT fails
                    pass

            ranked_links.append((text, link, score, filtered_by))
        ranked_links = sorted(ranked_links, key=lambda x: x[2], reverse=True)
        if len(ranked_links) == 0:
            ranked_links = [(text, link, len(text) * 0.5) for text, link in internal_links[:10]]  # fallback basic scoring
        links = [(f"{text} (Score: {score}, Filtered by: {filtered_by})" if show_link_scores else text, link) for text, link, score, filtered_by in ranked_links[:10]]

        html_content = soup.prettify()

        data = {
            "url": url,
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "links": links,
            "html": html_content
        }

        return data
    except Exception as e:
        return {"error": str(e)}

def is_allowed_by_robots(url, user_agent):
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except:
        return False

def crawl_internal_links(start_url, max_pages=3, user_agent="WebScraperBot"):
    seen = set()
    results = []
    queue = deque([start_url])

    while queue and len(seen) < max_pages:
        current_url = queue.popleft()
        if current_url in seen:
            continue
        seen.add(current_url)
        result = fetch_and_summarize(current_url)
        if "error" not in result:
            results.append(result)
            for _, link in result.get("links", []):
                if is_valid_url(link) and link not in seen and is_allowed_by_robots(link, user_agent):
                    queue.append(link)
                elif show_blocked_links:
                    st.warning(f"Blocked by robots.txt: {link}")
                continue
    return results

def summarize_with_gpt(data, selected_model, depth, tone):
    combined = "\n\n".join(data.get("paragraphs", []))
    prompt = f"""
You are an AI assistant. Summarize the following content into an executive overview, categorized bullet points, and detailed insights.
Tone: {tone}
Depth: {depth}
Include:
- Executive Summary
- Key Ideas
- Insights
- Recommendations
- Bullet format if appropriate
- Separate sections by headings

Content:
{combined}
"""

    try:
        if selected_model.startswith("gpt") and "OPENAI_API_KEY" in st.secrets:
            response = openai.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You summarize and structure webpage content."},
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
                    {"role": "system", "content": "You summarize and structure webpage content."},
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

st.sidebar.subheader("🌐 Keyword Web Crawler")
keyword_query = st.sidebar.text_input("Search query (Google-style)", "AI strategy site:mckinsey.com")
use_keyword_crawl = st.sidebar.checkbox("Use keyword-based crawling", value=False)

SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", None)
GOOGLE_CSE_API_KEY = st.secrets.get("GOOGLE_CSE_API_KEY", None)
GOOGLE_CSE_ID = st.secrets.get("GOOGLE_CSE_ID", None)

def search_web_for_keyword(query):
    urls = []
    if SERPER_API_KEY:
        try:
            url = "https://api.serper.dev/search"
            headers = {"X-API-KEY": SERPER_API_KEY}
            payload = {"q": query, "gl": "us"}
            res = requests.post(url, json=payload, headers=headers)
            res.raise_for_status()
            urls = [r["link"] for r in res.json().get("organic", [])]
            if urls:
                return urls
        except Exception as e:
            st.warning(f"Serper failed: {e}")

        if GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
        try:
            url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_CSE_API_KEY}&cx={GOOGLE_CSE_ID}&q={query}"
            res = requests.get(url)
            res.raise_for_status()
            urls = [item['link'] for item in res.json().get('items', [])]
            if urls:
                return urls
        except Exception as e:
            st.warning(f"Google CSE failed: {e}")

    return urls
        except Exception as e:
            st.warning(f"Google CSE failed: {e}")

    return urls
        except Exception as e:
            st.warning(f"Serper failed: {e}")

        if GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
        try:
            url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_CSE_API_KEY}&cx={GOOGLE_CSE_ID}&q={query}"
            res = requests.get(url)
            res.raise_for_status()
            urls = [item['link'] for item in res.json().get('items', [])]
            if urls:
                return urls
        except Exception as e:
            st.warning(f"Google CSE failed: {e}")

    return urls

st.sidebar.subheader("🛡️ Crawler Settings")
user_agent = st.sidebar.text_input("User-Agent string (for robots.txt)", "WebScraperBot")
show_blocked_links = st.sidebar.checkbox("Log links blocked by robots.txt", value=False)

url = st.text_area("Enter one or more webpage URLs (one per line):", "https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage")
model_choice = st.selectbox("Choose AI Model", [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "mixtral-8x7b",
    "gemini-pro"
])
depth = st.selectbox("Summary Depth", ["short", "medium", "long"], index=1)
tone = st.selectbox("Summary Tone", ["neutral", "professional", "friendly", "assertive"], index=0)
max_pages = st.slider("Max internal pages to crawl (per site):", 1, 10, 3)

if st.button("Summarize"):
    if use_keyword_crawl:
        urls = search_web_for_keyword(keyword_query)
    else:
        urls = [line.strip() for line in url.splitlines() if is_valid_url(line)]
    if not urls:
        st.error("Please enter at least one valid URL starting with http:// or https://")
        st.stop()

    for i, u in enumerate(urls):
        st.subheader(f"🔎 Crawling {u} (up to {max_pages} internal pages)")
        crawled_results = crawl_internal_links(u, max_pages=max_pages, user_agent=user_agent)
        st.markdown(f"✅ Crawled: {len(crawled_results)} page(s)")
        st.markdown(f"📌 Filter model: `{model_choice}`")

        for j, result in enumerate(crawled_results):
            st.markdown(f"### 📄 Summary {j+1} - {result['url']}")
            st.write(f"**Title:** {result['title']}")
            
            

            with st.spinner("Generating summary using AI model..."):
                ai_summary, used_model = summarize_with_gpt(result, model_choice, depth, tone)

            st.markdown(f"**Model used:** {used_model}")
            st.text_area("🧠 Summary", ai_summary, height=400)

            result.update({
                "summary": ai_summary,
                "model_used": used_model,
                "timestamp": datetime.datetime.now().isoformat()
            })

            st.download_button(
                label="📥 Download Summary JSON",
                data=json.dumps(result, indent=4, ensure_ascii=False),
                file_name=f"summary_{i+1}_{j+1}.json"
            )

            with st.expander("🔗 Show Links and Headings"):
                st.write("**Headings:**", result['headings'])
                st.write("**Links:**", result['links'])
