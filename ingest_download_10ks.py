"""
===============================================================================
INGEST_DOWNLOAD_10KS.PY
===============================================================================

PURPOSE
-------
This script downloads SEC Form 10-K filings for a list of medical device
companies using the sec-edgar-downloader library.

This is the DOCUMENT INGESTION STAGE of the RAG pipeline.

It fulfills the assignment requirements for:
    • Minimum 50 documents
    • Real-world document ingestion
    • Structured corporate filings (PDF/HTML class documents)

-------------------------------------------------------------------------------
WHY USE SEC EDGAR?
-------------------------------------------------------------------------------
The SEC EDGAR database provides official public filings submitted by
public companies.

10-K filings are:
    • Annual reports (highly structured)
    • Long-form disclosures (~100+ pages)
    • Organized by sections (Item 1, 1A, 7, etc.)
    • Publicly accessible
    • Stable (not constantly changing like news)

This makes them ideal for:
    • Structured chunking
    • Citation-based retrieval
    • Financial and regulatory Q&A
    • Academic RAG evaluation

-------------------------------------------------------------------------------
WHY USE sec-edgar-downloader?
-------------------------------------------------------------------------------
Instead of manually clicking 50 filings, this library:
    • Handles SEC-compliant requests
    • Automatically sets up folder structure
    • Downloads complete filing packages
    • Organizes by ticker and form type

-------------------------------------------------------------------------------
HOW THIS MEETS THE 50 DOCUMENT REQUIREMENT
-------------------------------------------------------------------------------
We use:

    10 tickers × 5 most recent 10-K filings = 50 documents

This ensures we satisfy the assignment's minimum document count.

===============================================================================
"""

import os
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader


# =============================================================================
# MAIN INGESTION FUNCTION
# =============================================================================

def main():
    """
    Main execution function.

    Steps:
        1) Load environment variables (.env file)
        2) Configure SEC-compliant downloader
        3) Define medical device tickers
        4) Download N 10-K filings per ticker
        5) Save to local structured directory
    """

    # -------------------------------------------------------------------------
    # STEP 1 — Load Environment Variables
    # -------------------------------------------------------------------------
    load_dotenv()

    """
    The SEC requires that all automated requests include a descriptive
    User-Agent containing:
        - Your real name
        - Your real email

    This is required by SEC policy to prevent anonymous scraping.

    You must define in .env:

        SEC_DOWNLOADER_NAME=Your Real Name
        SEC_DOWNLOADER_EMAIL=your_email@domain.com
    """

    name = os.getenv("SEC_DOWNLOADER_NAME")
    email = os.getenv("SEC_DOWNLOADER_EMAIL")

    if not name or not email:
        raise ValueError(
            "SEC_DOWNLOADER_NAME and SEC_DOWNLOADER_EMAIL must be set in .env"
        )

    # -------------------------------------------------------------------------
    # STEP 2 — Define Output Directory
    # -------------------------------------------------------------------------

    """
    All downloaded filings will be saved under:

        data/filings_raw/

    The downloader automatically creates this structure:

        data/filings_raw/
            sec-edgar-filings/
                TICKER/
                    10-K/
                        ACCESSION_NUMBER/
                            filing-details.html
                            full-submission.txt
                            etc.

    We later process this structure in build_index.py.
    """

    raw_dir = "data/filings_raw"
    os.makedirs(raw_dir, exist_ok=True)

    # Create downloader instance
    dl = Downloader(name, email, raw_dir)

    # -------------------------------------------------------------------------
    # STEP 3 — Define Medical Device Company Tickers
    # -------------------------------------------------------------------------

    """
    Selected medical device / medtech companies.

    These companies are:
        • Large-cap
        • Publicly traded
        • File consistent 10-K reports
        • Representative of med device industry

    You can modify this list if needed.
    """

    tickers = [
        "EW",   # Edwards Lifesciences
        "MDT",  # Medtronic
        "SYK",  # Stryker
        "BSX",  # Boston Scientific
        "ISRG", # Intuitive Surgical
        "ABT",  # Abbott (devices + diagnostics)
        "ZBH",  # Zimmer Biomet
        "ALGN", # Align Technology
        "PODD", # Insulet
        "DXCM", # Dexcom
    ]

    # Validate tickers
    if len(tickers) == 0:
        raise ValueError("Ticker list is empty.")

    # -------------------------------------------------------------------------
    # STEP 4 — Define Number of Filings Per Company
    # -------------------------------------------------------------------------

    """
    We set:

        limit_per_ticker = 5

    Therefore:
        10 companies × 5 filings = 50 documents

    This satisfies the assignment's minimum document requirement.
    """

    limit_per_ticker = 5

    print("===============================================================================")
    print("Beginning SEC 10-K Download Process")
    print("===============================================================================")
    print(f"Companies selected: {len(tickers)}")
    print(f"Filings per company: {limit_per_ticker}")
    print(f"Expected total documents: {len(tickers) * limit_per_ticker}")
    print("-------------------------------------------------------------------------------")

    # -------------------------------------------------------------------------
    # STEP 5 — Download Filings
    # -------------------------------------------------------------------------

    for ticker in tickers:
        try:
            print(f"\nDownloading {limit_per_ticker} 10-K filings for {ticker} ...")

            """
            dl.get arguments:
                form_type: "10-K"
                ticker: company ticker symbol
                limit: number of most recent filings

            The downloader handles:
                • SEC request formatting
                • Proper User-Agent header
                • Directory creation
                • Rate limiting
            """

            dl.get("10-K", ticker, limit=limit_per_ticker)

        except Exception as e:
            print(f"[WARNING] Failed to download filings for {ticker}: {e}")

    # -------------------------------------------------------------------------
    # STEP 6 — Completion Summary
    # -------------------------------------------------------------------------

    print("\n===============================================================================")
    print("Download Complete")
    print("Raw filings stored at: data/filings_raw/")
    print("Next Step: Run build_index.py to create the vector database.")
    print("===============================================================================")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()