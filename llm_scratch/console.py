# Function to display the table and handle user interaction
def interactive_table(data, headers):
    """
    Displays a table that the user can scroll through and select a row from.

    :param data: A list of lists, where each inner list represents a row in the table.
    :param headers: A list of strings representing the column headers.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from readchar import readkey, key

    SELECTED_STYLE = "on blue"
    selected_index = 0

    def generate_table(selected_index):
        table = Table()
        for header in headers:
            table.add_column(header)
        for i, row_data in enumerate(data):
            style = SELECTED_STYLE if i == selected_index else ""
            table.add_row(*row_data, style=style)
        return table

    console = Console()

    with Live(generate_table(selected_index), refresh_per_second=10, console=console) as live:
        while True:
            key_pressed = readkey()
            if key_pressed == key.UP:  # Scroll up
                selected_index = max(0, selected_index - 1)
            elif key_pressed == key.DOWN:  # Scroll down
                selected_index = min(len(data) - 1, selected_index + 1)
            elif key_pressed == key.ENTER:  # Select the row
                console.print(f"You selected: {data[selected_index]}")
                return data[selected_index]
                break
            live.update(generate_table(selected_index))
