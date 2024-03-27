import llm_scratch

button_frames=5

def start_new_game():
    pyboy.tick(1000)
    pyboy.button("start")
    pyboy.tick(1000)
    pyboy.button("a")
    pyboy.tick(1000)
    pyboy.button("a")
    pyboy.tick(1000)

def get_to_battle():
    # prof oak's intro & naming
    advance_dialog(20)
    pyboy.button("start")

    advance_dialog(22)

    # exit house
    move("left")
    move("up", 4)
    move("right", 4)
    move("up")
    move("right")
    move("down", 6)
    move("left", 5)
    pyboy.screen.image.save("building.png")
    move("down")

    # go into grass
    move("right", 5)
    move("up", 6)
    pyboy.screen.image.save("overworld.png")

    # get pikachu
    advance_dialog(35)
    move("down")
    move("right", 2)
    move("up")
    pyboy.button("a", button_frames)
    pyboy.tick(500)
    advance_dialog(17)
    pyboy.button("start", button_frames)
    pyboy.tick(500)

    #initiate battle
    move("down", 4)
    advance_dialog(6)



def advance_dialog(n):
    for i in range(n):
        pyboy.button("a")
        pyboy.tick(500)

def move(direction, n=1):
    for step in range(n):
        pyboy.button(direction, button_frames)
        pyboy.tick(500)

def get_text():
    img = pyboy.screen.image
    bottom = create_image_from_bottom_rows(img, 50)
    bottom.save("bottom.png")
    return llm_scratch.extract_text("bottom.png")


def create_image_from_bottom_rows(img, n):
    """
    Creates a new image from the bottom n rows of the original image.

    Parameters:
    - image_path: The path to the original image.
    - n: The number of rows from the bottom to include in the new image.

    Returns:
    - A PIL Image object containing the bottom n rows of the original image.
    """
    original_image = img
    width, height = original_image.size

    # Calculate the cropping box
    left = 0
    top = height - n
    right = width
    bottom = height

    # Crop the image to get the bottom n rows
    cropped_image = original_image.crop((left, top, right, bottom))

    return cropped_image
