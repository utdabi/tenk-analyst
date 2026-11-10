"""Numbers from SEC's XBRL financial data ("company facts"), and the one function that formats amounts."""
import functools
import os

import httpx

# The CIK of each company (SEC's ID number). It is written here because the data/ folder is not on Render.
CIKS = {
    "TSLA": "0001318605", "GM": "0001467858", "F": "0000037996", "RIVN": "0001874178", "LCID": "0001811210",
    "APTV": "0001521332", "BWA": "0000908255", "ALV": "0001034670", "HOG": "0000793952", "CVNA": "0001690820",
}

# For each metric: the XBRL tags to try (in this order) and the words the 10-K text uses for it.
METRICS = {
    "revenue": {"tags": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
                "phrase": "total revenues"},
    "net_income": {"tags": ["NetIncomeLoss", "ProfitLoss"], "phrase": "net income"},
    "rd_expense": {"tags": ["ResearchAndDevelopmentExpense",
                            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
                   "phrase": "research and development expenses"},
    "operating_income": {"tags": ["OperatingIncomeLoss"], "phrase": "operating income"},
}


@functools.lru_cache(maxsize=16)  # the file is large and does not change, so keep each company in memory
def company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = httpx.get(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=60)
    response.raise_for_status()
    return response.json()


def get_metric(ticker: str, metric: str, year: int):
    """Return (value in dollars, tag) for a full-year amount, or (None, None) if SEC data does not have it."""
    facts = company_facts(CIKS[ticker])["facts"].get("us-gaap", {})
    for tag in METRICS[metric]["tags"]:
        for entry in facts.get(tag, {}).get("units", {}).get("USD", []):
            if entry.get("frame") == f"CY{year}":  # CY2025 = the full calendar year 2025
                return entry["val"], tag
    return None, None


def format_amount(value: float, year: int) -> str:
    """$9.4 billion (FY2025); a loss gets a minus sign before the dollar sign; below $0.05 billion use millions."""
    sign = "-" if value < 0 else ""
    size = abs(value)
    if size >= 50_000_000:
        return f"{sign}${size / 1e9:.1f} billion (FY{year})"
    return f"{sign}${size / 1e6:.1f} million (FY{year})"
