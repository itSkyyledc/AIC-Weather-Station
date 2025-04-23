from PIL import Image, ImageDraw, ImageFont
import os

try:
    # Create a 400x300 image with a red background
    img = Image.new('RGB', (400, 300), color='red')

    # Get a drawing context
    d = ImageDraw.Draw(img)

    # Add text
    text = "Camera Feed\nUnavailable"
    
    # Try to load a system font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    # Calculate text size and position
    text_bbox = d.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Center the text
    x = (400 - text_width) / 2
    y = (300 - text_height) / 2

    # Draw the text
    d.text((x, y), text, font=font, fill='white')

    # Create the directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)

    # Save the image
    img.save('static/images/error.png')
    print("Error image generated successfully")
except Exception as e:
    print(f"Error generating error image: {e}") 