import base64

def dataframe_to_markdown(df):
    headers = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for index, row in df.iterrows():
        rows.append("| " + " | ".join(row.values.astype(str)) + " |")
    table_markdown = "\n".join([headers, separator] + rows)
    return table_markdown

def convert_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        print("Converting image to Base64 successfully!")
        return base64_image
    except Exception as e:
        print("Error converting image to Base64:", str(e))
        return None
