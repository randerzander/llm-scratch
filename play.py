from pyboy import PyBoy
from ascii_magic import AsciiArt
import llm_scratch
import utils

pyboy = PyBoy("pokemon_yellow.gbc")
pyboy.set_emulation_speed(0)
utils.pyboy = pyboy

button_frames=5

utils.start_new_game()
utils.get_to_battle()

img = pyboy.screen.image
img.save("battle.png")
art = AsciiArt.from_pillow_image(img)
print(art.to_terminal(monochrome=True, columns=100))

#text = utils.get_text()
#print(text)
