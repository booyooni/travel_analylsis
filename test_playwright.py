from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto(
        "https://www.verygoodtour.com/Product/PackageDetail?ProCode=APP1141-260401TR1&PriceSeq=2&MenuCode=leaveLayer#jq_id_travelReviews",
        wait_until="domcontentloaded",
        timeout=60000,
    )
    print(page.title())
    input("브라우저 확인 후 엔터를 누르세요: ")
    browser.close()
