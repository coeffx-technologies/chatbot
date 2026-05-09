from scrapling.fetchers import StealthyFetcher
import re 


def linkedin_fetch(company_name):

    StealthyFetcher.adaptive = True

    # --- dismiss login modal if it pops up ---
    def after_load(page):
        page.wait_for_timeout(5000)

        modal = page.locator("div#base-contextual-sign-in-modal div.modal__overlay-visible")
        try:
            modal.first.wait_for(state="visible", timeout=10000)
        except Exception:
            pass

        dismiss = page.locator("button[data-tracking-control-name='organization_guest_contextual-sign-in-modal_modal_dismiss']")
        try:
            dismiss.first.wait_for(state="attached", timeout=5000)
            dismiss.first.click(force=True)
            page.wait_for_timeout(3000)
        except Exception:
            pass

    page = StealthyFetcher.fetch(
        f"https://www.linkedin.com/company/{company_name}/",
        headless=True,
        network_idle=True,
        page_action=after_load,
    )

    # --- helpers ---
    def text(selector):
        el = page.css(selector)
        return el[0].text.strip() if el else None

    def attr(selector, attribute):
        el = page.css(selector)
        return el[0].attrib.get(attribute) if el else None

    # --- dt/dd key-value pairs (industry, hq, founded, type, specialties, company size) ---
    details = {}
    for dt, dd in zip(page.css("dt"), page.css("dd")):
        key = dt.text.strip().lower()
        val = dd.text.strip()
        if key:
            details[key] = val

    # --- followers: parse from og:description meta tag ---
    tagline_raw = attr("meta[name='description']", "content")
    followers_match = re.search(r"([\d,]+)\s+followers on LinkedIn", tagline_raw or "")
    linkedin_followers = followers_match.group(1) if followers_match else None

    # --- website: LinkedIn renders it as <a> inside dd, not plain text ---
    website_el = page.css("dd a[data-test-id='about-us__website']") or page.css("dd a[href*='http']")
    website = website_el[0].attrib.get("href") if website_el else None

    return {
        "company_slug":       company_name,
        "company_name":       (attr("meta[property='og:title']", "content") or "").replace(" | LinkedIn", "").strip(),
        "tagline":            tagline_raw,
        "logo_url":           attr("meta[property='og:image']", "content"),
        "description":        text("p[data-test-id='about-us__description']"),
        "linkedin_followers": linkedin_followers,
        "website":            website,
        "industry":           details.get("industry"),
        "hq":                 details.get("headquarters"),
        "founded":            details.get("founded"),
        "company_type":       details.get("type"),
        "employee_count":     details.get("company size"),
        "specialties":        details.get("specialties"),
    }
