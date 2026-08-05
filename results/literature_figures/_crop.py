from PIL import Image
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Mardani Fig 1 — page 3, 2125x2750
# Figure occupies top portion above the caption. Visual inspection:
# - Top: ~y=200 (small margin)
# - Bottom (just above caption "Figure 1:"): ~y=1280
# - Left: ~x=140
# - Right: ~x=2000
img = Image.open('mardani_p3_full-03.png')
left, top, right, bottom = 100, 180, 2050, 1100
img.crop((left, top, right, bottom)).save('mardani_corrdiff_fig1.png', optimize=True)
print('Fig1 cropped:', (right-left, bottom-top))

# Mardani Fig S1 — page 22, 2550x3300
# Architecture sketch occupies top ~1/3 of page
# Visual inspection:
# - Top: ~y=270 (above figure box)
# - Bottom (just above caption "Figure S1:"): ~y=1300
# - Left: ~x=200
# - Right: ~x=2300
img = Image.open('mardani_p22_full-22.png')
left, top, right, bottom = 200, 270, 2400, 1300
img.crop((left, top, right, bottom)).save('mardani_corrdiff_fig2.png', optimize=True)
print('Fig2 (S1) cropped:', (right-left, bottom-top))
