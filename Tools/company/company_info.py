import os, json
from apify_client import ApifyClient
from dotenv import load_dotenv
load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
client = ApifyClient(APIFY_TOKEN)


def scrape_linkedin_company(company_name: str, location: str = None) -> dict:

    # Step 1: Google search to find exact LinkedIn company URL
    search_query = f'"{company_name}" {location} site:linkedin.com/company' if location else f'"{company_name}" site:linkedin.com/company'

    google_run = client.actor("apify/google-search-scraper").call(
        run_input={
            "queries": search_query,
            "maxPagesPerQuery": 1,
            "resultsPerPage": 5,
        }
    )

    google_results = list(client.dataset(google_run["defaultDatasetId"]).iterate_items())

    # Grab first linkedin.com/company/ URL
    company_url = None
    for result in google_results:
        for item in result.get("organicResults", []):
            url = item.get("url", "")
            if "linkedin.com/company/" in url:
                company_url = url.split("?")[0].rstrip("/")
                break
        if company_url:
            break

    if not company_url:
        return {"error": f"Could not find LinkedIn URL for '{company_name}'"}

    # Step 2: Scrape the company page
    company_run = client.actor("harvestapi/linkedin-company").call(
        run_input={
            "companies": [company_url],
        }
    )

    results = list(client.dataset(company_run["defaultDatasetId"]).iterate_items())

    if not results:
        return {"error": f"No data returned for '{company_name}'", "linkedin_url": company_url}

    raw = results[0]

    # Extract headquarters from locations list
    headquarters = None
    for loc in raw.get("locations", []):
        if loc.get("headquarter"):
            parsed = loc.get("parsed", {})
            headquarters = parsed.get("text")
            break

    # Extract primary industry
    industries = raw.get("industries", [])
    industry = industries[0]["name"] if industries else None

    # Extract founded year
    founded_on = raw.get("foundedOn")
    founded = founded_on.get("year") if isinstance(founded_on, dict) else founded_on

    return {
        "name": raw.get("name"),
        "linkedin_url": raw.get("linkedinUrl"),
        "tagline": raw.get("tagline"),
        "description": raw.get("description"),
        "industry": industry,
        "company_type": raw.get("companyType"),
        "employee_count": raw.get("employeeCount"),
        "employee_count_range": raw.get("employeeCountRange"),
        "followers": raw.get("followerCount"),
        "founded": founded,
        "headquarters": headquarters,
        "logo": raw.get("logo"),
        "website": raw.get("website"),
        "specialities": raw.get("specialities"),
    }


# Test
print(json.dumps(scrape_linkedin_company("Darwinbox", "Hyderabad, India"), indent=2, default=str))
# print(json.dumps(scrape_linkedin_company("Microsoft", "Redmond"), indent=2, default=str))
