from scrapling.fetchers import StealthyFetcher
import re

def followers(company_name):
    StealthyFetcher.adaptive = True

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
    def attr(selector, attribute):
        el = page.css(selector)
        return el[0].attrib.get(attribute) if el else None
    
    tagline =  attr("meta[name='description']", "content")
    followers_match = re.search(r"([\d,]+)\s+followers on LinkedIn", tagline or "")
    linkedin_followers = followers_match.group(1) if followers_match else None

    return {
        'linkedin_followers' : linkedin_followers
    }