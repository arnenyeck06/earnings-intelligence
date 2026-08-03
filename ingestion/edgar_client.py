import time
import re
import requests

HEADERS = {
    "User-Agent": "earnings-intelligence contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_BASE = "https://data.sec.gov"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def get_cik(ticker):
    resp = requests.get(COMPANY_TICKERS_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found.")


def _search_filings_for_10k(filings, year):
    """Search a filings dict for a 10-K around the given year."""
    for form, accession, date in zip(
        filings["form"], filings["accessionNumber"], filings["filingDate"]
    ):
        if form not in ("10-K", "10-K405"):
            continue
        filing_year = int(date[:4])
        if filing_year in (year - 1, year, year + 1):
            return accession, date
    return None, None


def get_10k_accession(cik, year):
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Check recent filings first
    accession, date = _search_filings_for_10k(data["filings"]["recent"], year)
    if accession:
        return accession, date

    # Check older filing pages
    for file_info in data["filings"].get("files", []):
        filing_from = int(file_info["filingFrom"][:4])
        filing_to = int(file_info["filingTo"][:4])
        # Only fetch if year range overlaps
        if not (filing_from <= year + 1 and filing_to >= year - 1):
            continue
        older_url = f"{EDGAR_BASE}/submissions/{file_info['name']}"
        older_resp = requests.get(older_url, headers=HEADERS, timeout=10)
        older_resp.raise_for_status()
        older_data = older_resp.json()
        accession, date = _search_filings_for_10k(older_data, year)
        if accession:
            return accession, date

    raise ValueError(f"No 10-K found for CIK {cik} around year {year}.")


def get_filing_text(cik, accession):
    accession_nodash = accession.replace("-", "")
    filing_index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
        f"/{accession_nodash}/{accession}-index.json"
    )
    resp = requests.get(filing_index_url, headers=HEADERS, timeout=15)

    if resp.status_code == 200:
        index_data = resp.json()
        documents = index_data.get("directory", {}).get("item", [])
        primary_doc = None
        for doc in documents:
            doc_type = doc.get("type", "")
            name = doc.get("name", "")
            if doc_type in ("10-K", "10-K405") and name.endswith(".htm"):
                primary_doc = name
                break
        if not primary_doc:
            htm_docs = [
                d for d in documents
                if d.get("name", "").endswith(".htm")
                and not d.get("name", "").lower().startswith("ex")
                and "exhibit" not in d.get("name", "").lower()
            ]
            if htm_docs:
                htm_docs.sort(key=lambda x: int(x.get("size", 0)), reverse=True)
                primary_doc = htm_docs[0]["name"]
    else:
        index_htm_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
            f"/{accession_nodash}/{accession}-index.htm"
        )
        resp = requests.get(index_htm_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        pattern = r'<td[^>]*>10-K</td>.*?<a href="[^"]*?/([^"/]+\.htm)"'
        matches = re.findall(pattern, resp.text, re.IGNORECASE | re.DOTALL)
        primary_doc = matches[0] if matches else None

    if not primary_doc:
        raise ValueError(f"Could not find primary 10-K document for {accession}")

    time.sleep(0.2)
    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
        f"/{accession_nodash}/{primary_doc}"
    )
    print(f"[edgar] Primary doc: {primary_doc}")
    resp = requests.get(doc_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_10k(ticker, year):
    print(f"[edgar] Fetching CIK for {ticker}...")
    cik = get_cik(ticker)
    print(f"[edgar] CIK={cik}. Looking for 10-K around {year}...")
    accession, filing_date = get_10k_accession(cik, year)
    print(f"[edgar] Found {accession} filed={filing_date}. Fetching...")
    html_text = get_filing_text(cik, accession)
    print(f"[edgar] Fetched {len(html_text):,} chars.")
    return {
        "ticker": ticker.upper(), "cik": cik, "year": year,
        "accession": accession, "filing_date": filing_date, "html": html_text
    }
