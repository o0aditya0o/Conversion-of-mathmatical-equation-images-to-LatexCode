import Image, ImageFont, ImageDraw

im = Image.new("L", (128,128),255)

draw = ImageDraw.Draw(im)

# use a truetype font
font = ImageFont.truetype("/home/aditya/Desktop/fonts/cmmi10.ttf", 96)

(font_width, font_height) = font.getsize("A")

# Calculate x position
x = (128 - font_width) / 2

# Calculate y position
y = (128 - font_height) / 2

# Draw text : Position, String,
# Options = Fill color, Font
draw.text((x, y), "A", 0,font=font)

# remove unneccessory whitespaces if needed
im=im.crop(im.getbbox())

# write into file
im.save("img.png")

# symbols = ['α','β','γ','θ','η','μ','λ','π','ρ','σ','τ','δ',
# 			'ϕ','≤','≥','≠','÷','×','±'
# 			,'∑','∫','∏']