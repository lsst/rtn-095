import fitz  # PyMuPDF

# Open the PDF
doc = fitz.open("RTN-095.pdf")

# Choose the page
page = doc.load_page(20)  # Page 21 (index starts from 0)

# Define the coordinates you want to locate
x, y = 78.0935, 540.878

# Convert coordinates to a point on the page
point = fitz.Point(x, y)

# Highlight the location on the page
highlight = page.add_highlight_annot(fitz.Rect(x-5, y-5, x+5, y+5))

# Save the document with the highlight
doc.save("highlighted_output.pdf")