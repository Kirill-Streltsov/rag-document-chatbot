"""Generate the sample business-report PDF used as a demo document and test
fixture.

Uses a completely fictional company ("Acme Robotics GmbH") so the repo contains
no real or confidential data. Run:

    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "sample_docs" / "acme_annual_report_2024.pdf"

# (text, style) tuples. Use style "PageBreak" as a sentinel to start a new page.
SECTIONS = [
    ("Acme Robotics GmbH - Annual Report 2024", "Title"),
    (
        "Acme Robotics GmbH is a fictional company created solely to demonstrate "
        "this document-QA application. Any resemblance to a real business is "
        "coincidental. All figures below are invented.",
        "BodyText",
    ),
    ("1. Company Overview", "Heading2"),
    (
        "Acme Robotics GmbH was founded in 2011 and is headquartered in Dortmund, "
        "Germany. The company designs autonomous warehouse robots and the software "
        "that coordinates them. As of December 2024 the company employed 512 people "
        "across three offices: Dortmund, Munich, and Porto. The legal form is a "
        "Gesellschaft mit beschränkter Haftung and the company is registered in the "
        "Handelsregister of Amtsgericht Dortmund under HRB 998877.",
        "BodyText",
    ),
    ("2. Financial Highlights", "Heading2"),
    (
        "Total revenue for the 2024 financial year was 84.6 million euros, up 23 "
        "percent from 68.8 million euros in 2023. Gross margin improved to 41 "
        "percent from 37 percent the previous year. Net profit was 9.2 million "
        "euros. Research and development spending reached 14.1 million euros, "
        "roughly 17 percent of revenue. Cash and cash equivalents at year end stood "
        "at 31.5 million euros, and the company carried no long-term debt.",
        "BodyText",
    ),
    ("3. Revenue by Region", "Heading2"),
    (
        "Germany remained the largest market at 46 percent of revenue. The rest of "
        "the European Union contributed 33 percent, the United Kingdom 12 percent, "
        "and the remaining 9 percent came from a pilot programme in Canada. "
        "Management considers the Canadian pilot a strategic beachhead for North "
        "American expansion rather than a material revenue source in 2024.",
        "BodyText",
    ),
    ("", "PageBreak"),
    ("4. Products", "Heading2"),
    (
        "The flagship product is the AR-7 mobile picking robot, which accounted for "
        "61 percent of unit sales. The AR-7 has a battery life of 11 hours and a "
        "maximum payload of 45 kilograms. It navigates using a combination of LiDAR "
        "and visual SLAM, and can operate safely alongside human workers without "
        "safety cages. The older AR-5 model remains available for price-sensitive "
        "customers and made up 18 percent of unit sales.",
        "BodyText",
    ),
    (
        "A cloud fleet-management platform, AcmeOS, is sold as an annual "
        "subscription and generated 22 million euros in recurring revenue, a figure "
        "that grew 40 percent year over year. AcmeOS coordinates robot traffic, "
        "schedules charging, and exposes a REST API for warehouse-management "
        "systems. The net revenue retention rate for AcmeOS subscriptions was 118 "
        "percent.",
        "BodyText",
    ),
    ("5. Leadership", "Heading2"),
    (
        "The Chief Executive Officer is Jane Doe, who joined in 2019. The Chief "
        "Technology Officer is Miguel Santos, who co-founded the company. The Chief "
        "Financial Officer is Priya Nair, who joined from a logistics group in 2022. "
        "The company is governed by a five-member supervisory board chaired by Dr. "
        "Anke Vogel.",
        "BodyText",
    ),
    ("6. People and Culture", "Heading2"),
    (
        "Headcount grew from 388 to 512 during the year. Engineering represented 54 "
        "percent of staff. Voluntary attrition was 6.9 percent, well below the "
        "industry average. The company offers a four-day work week in its Porto "
        "office as a pilot and reported an employee engagement score of 82 out of "
        "100.",
        "BodyText",
    ),
    ("", "PageBreak"),
    ("7. Sustainability", "Heading2"),
    (
        "Acme Robotics reduced Scope 1 and Scope 2 emissions by 18 percent per unit "
        "shipped. All three offices run on certified renewable electricity. The AR-7 "
        "is designed for disassembly, and 92 percent of a returned unit by mass can "
        "be recycled. The company aims to be carbon neutral across Scopes 1 and 2 by "
        "2028.",
        "BodyText",
    ),
    ("8. Risks", "Heading2"),
    (
        "The most significant risks identified by management are supply-chain "
        "concentration in high-precision motors, of which a single supplier provides "
        "70 percent, and foreign-exchange exposure to the British pound. A "
        "prolonged component shortage could delay AR-9 shipments. The company holds "
        "twelve weeks of critical component inventory as a buffer.",
        "BodyText",
    ),
    ("9. Outlook", "Heading2"),
    (
        "For 2025 management expects revenue between 100 and 108 million euros, "
        "driven by the launch of the AR-9 robot and expansion into the French "
        "market. The company plans to hire approximately 120 additional engineers "
        "and to open a fourth office in Lyon. The AR-9 is expected to offer a 30 "
        "percent higher payload than the AR-7 and to begin shipping in the third "
        "quarter of 2025.",
        "BodyText",
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, title="Acme Robotics GmbH - Annual Report 2024")
    flow = []
    for text, style in SECTIONS:
        if style == "PageBreak":
            flow.append(PageBreak())
            continue
        flow.append(Paragraph(text, styles[style]))
        flow.append(Spacer(1, 0.4 * cm))
    doc.build(flow)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
