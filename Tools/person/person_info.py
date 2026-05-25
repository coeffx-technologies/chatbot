import os, json
from apify_client import ApifyClient
from dotenv import load_dotenv
load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
client = ApifyClient(APIFY_TOKEN)

def scrape_linkedin_profile(profile_url: str) -> dict:
    run_input = {
        "urls": [
            {"url": profile_url}  
        ],
    }
    run = client.actor("supreme_coder/linkedin-profile-scraper").call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    if not items:
        return {}

    raw = items[0]

    # Name
    name = f"{raw.get('firstName', '')} {raw.get('lastName', '')}".strip()

    # Location
    location = raw.get('geoLocationName') or ""

    # Current job (first position without an end date)
    current_job = None
    for pos in raw.get('positions', []):
        # Flatten possible nested "positions"
        pos_list = pos.get('positions', [pos]) if isinstance(pos, dict) and 'positions' in pos else [pos]
        for p in pos_list:
            if p.get('timePeriod', {}).get('endDate') is None:
                current_job = p.get('title')
                break
        if current_job:
            break

    # Experience (all roles)
    experience = []
    for pos in raw.get('positions', []):
        pos_list = pos.get('positions', [pos]) if isinstance(pos, dict) and 'positions' in pos else [pos]
        for p in pos_list:
            experience.append({
                "title": p.get('title'),
                "company": p.get('company', {}).get('name'),
                "start_date": p.get('timePeriod', {}).get('startDate'),
                "end_date": p.get('timePeriod', {}).get('endDate'),
            })

    # Education
    education = []
    for edu in raw.get('educations', []):
        education.append({
            "school": edu.get('schoolName'),
            "degree": edu.get('degreeName'),
            "field": edu.get('fieldOfStudy'),
            "start_year": edu.get('timePeriod', {}).get('startDate', {}).get('year'),
            "end_year": edu.get('timePeriod', {}).get('endDate', {}).get('year'),
        })

    # Certifications & Licenses
    certifications = []
    for cert in raw.get('certifications', []):
        certifications.append({
            "name": cert.get('name'),
            "issuing_org": cert.get('authority'),
        })

    return {
        "name": name,
        "location": location,
        "current_job": current_job,
        "experience": experience,
        "education": education,
        "certifications": certifications,
    }


print(json.dumps(scrape_linkedin_profile('https://www.linkedin.com/in/jeeveshpranav/'), indent=2, default=str))