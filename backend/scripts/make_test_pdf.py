"""
Generates a small fictional PDF used to test the pipeline end to end.

Writes to backend/test_fixtures/ by default - deliberately OUTSIDE the real data/ folder,
which is the live ingestion source for real documents and must never get a synthetic test
PDF mixed into it.
"""
import os

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

PAGES = [
    """Project Nightingale - Internal Overview.

Project Nightingale is a fictional research initiative studying migratory patterns
of the Siberian crane using low-power satellite tags. The project began in March 2021
and is led by Dr. Amara Chen at the Boreal Ecology Institute.

The team deployed 42 tags across three breeding grounds in northern Russia. Each tag
transmits location data every six hours and has a battery life of approximately 18
months. The primary goal of the project is to identify previously unmapped stopover
sites along the East Asian flyway.""",
    """Funding and Budget.

Project Nightingale is funded by a grant of 1.2 million dollars from the Global Wildlife
Fund, covering the period from 2021 to 2024.

Of this budget, 45 percent is allocated to tag hardware and satellite data fees, 30
percent to field operations including staff travel and local guide fees, and the
remaining 25 percent to data analysis software and a postdoctoral researcher position.
A mid-project review in 2023 recommended increasing the field operations budget by 10
percent for the final year due to rising fuel costs.""",
    """Key Findings So Far.

As of the most recent report, the team has identified two previously undocumented
stopover sites: one in the Yellow River Delta and one near Poyang Lake. Tagged cranes
spent an average of 11 days resting at the Yellow River Delta site before continuing
their migration.

Dr. Chen's team also observed that cranes departing later in the season tended to take a
more westerly route, which the team hypothesizes is linked to wind patterns rather than
food availability. A follow-up paper on this hypothesis is planned for early 2025.""",
]

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "test_fixtures", "sample.pdf")


def make_pdf(path: str = DEFAULT_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    c = canvas.Canvas(path, pagesize=letter)
    _, height = letter
    for page_text in PAGES:
        c.setFont("Helvetica", 11)
        y = height - 72
        for line in page_text.strip().split("\n"):
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    make_pdf()
