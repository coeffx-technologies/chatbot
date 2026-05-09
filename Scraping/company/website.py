from scrapling.fetchers import StealthyFetcher
import re 


def website(company_name):

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


    website_el = page.css("dd a[data-test-id='about-us__website']") or page.css("dd a[href*='http']")
    website = website_el[0].attrib.get("href") if website_el else None

    return {
        "website" : website
    }