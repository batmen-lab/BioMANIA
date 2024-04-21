import os
import base64
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

# draw text with markdown style
def draw_markdown_text(c, text, line_num, font_name="Helvetica", font_size=10, leading=15, max_width=440):
    """
    Draw text with markdown style and handle line breaks based on max_width.
    """
    splitted_texts = []
    # Split text based on the maximum width
    for line in text.split('\n'):
        splitted_texts.extend(simpleSplit(line, font_name, font_size, max_width))
    
    for line in splitted_texts:
        # Check if there's enough space, if not, start a new page
        if line_num < 40:  # Adjust this value as needed
            c.showPage()
            line_num = 750
        
        # Differentiate text formatting based on markdown symbols
        if line.startswith("# "):
            c.setFont("Helvetica-Bold", font_size + 2)
            line_num -= leading
            c.drawString(100, line_num, line[2:])
        elif line.startswith("## "):
            c.setFont("Helvetica-Bold", font_size + 1)
            line_num -= leading
            c.drawString(100, line_num, line[3:])
        elif line.startswith("### "):
            c.setFont("Helvetica-Bold", font_size)
            line_num -= leading
            c.drawString(100, line_num, line[4:])
        elif line.startswith("* "):
            c.setFont(font_name, font_size)
            line_num -= leading
            c.drawString(110, line_num, line[2:])
        else:
            c.setFont(font_name, font_size)
            line_num -= leading
            c.drawString(100, line_num, line)
    return line_num

# Function to visualize text from a conversation and generate a PDF
def visualize_text_from_conversation(conversation):
    pdf_path = "./report/output.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    os.makedirs("./report/tmpimages", exist_ok=True)
    line_num = 750 # Initial position for text drawing
    image_count = 0
    previous_role = None
    # Iterating through the conversation's history
    for chat in conversation['history']:
        for messages in chat['messages']:
            role = messages['role']
            if role != previous_role:
                line_num = line_num - 20  # Create a separation between roles
                c.setFont("Helvetica-Bold", 12)
                line_num = draw_markdown_text(c, f"{role}:", line_num, font_name="Helvetica-Bold", font_size=12)
                previous_role = role
            c.setFont("Helvetica", 10)
            text = messages['content']
            line_num = draw_markdown_text(c, text, line_num)
            if role == 'assistant':
                tools = messages.get('tools', [])
                # Processing assistant's tools (tasks, images, tables)
                for tool in tools:
                    if 'log' in tool['block_id']:
                        title = tool['task_title']
                        main_text = tool['task']
                        line_num = draw_markdown_text(c, f"{title}:", line_num)
                        line_num = draw_markdown_text(c, main_text, line_num)
                        if tool['imageData']:
                            imageData = tool['imageData']
                            image_data = base64.b64decode(imageData)
                            image = Image.open(BytesIO(image_data))
                            image_path = f"./report/tmpimages/image_{image_count}.png"
                            image.save(image_path, "PNG")
                            image_count += 1
                            img_width, img_height = image.size
                            aspect_ratio = img_height / img_width
                            max_width = 400
                            max_height = int(max_width * aspect_ratio)
                            # Check available space for the image, otherwise, start a new page
                            if line_num - max_height < 50:
                                c.showPage()
                                c.setFont("Helvetica", 10)
                                line_num = 750
                            c.drawInlineImage(image_path, 100, line_num - max_height, width=max_width, height=max_height)
                            line_num -= max_height + 15  # Adjust space for image and separation
                        if tool['tableData']:
                            tableData = tool['tableData']
                            line_num = draw_markdown_text(c, "Table:", line_num - 15)
                            line_num = draw_markdown_text(c, tableData, line_num - 15)
                            # Check if we're too close to the bottom of the page after drawing the table
                            if line_num < 40:
                                c.showPage()
                                line_num = 750
    c.save()
    return pdf_path

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    import argparse
    from gpt.utils import load_json
    # Parsing arguments for the JSON file path
    parser = argparse.ArgumentParser(description="Extract tasks from JSON file")
    parser.add_argument("file_path", help="Path to the JSON file")
    args = parser.parse_args()
    # Loading JSON content from the file
    json_content = load_json(args.file_path)
    # Generating a PDF visualizing the conversation text and tools
    output_pdf_path = visualize_text_from_conversation(json_content)