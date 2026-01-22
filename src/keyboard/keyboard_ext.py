import keyboard
import csv
import asyncio
from datetime import datetime
from typing import Callable


def _keyboard_listener(keyboard_csv: str):
    """Keyboard listener that records events to CSV."""
    event_count = 0
    
    f = open(keyboard_csv, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['event_type', 'scan_code', 'name', 'time'])
    
    def on_event(event):
        nonlocal event_count
        time_custom = datetime.fromtimestamp(event.time).strftime("%H:%M:%S.%f")[:-3]
        writer.writerow([event.event_type, event.scan_code, event.name, time_custom])
        event_count += 1
        
        # Save every 10 keys
        if event_count % 10 == 0:
            f.flush()  
            
        # File handling when we close the writing
        if event.name == 'esc' and event.event_type == 'down':
            keyboard.unhook_all()
            f.flush()
            f.close()
            print(f"Recording Complete. {event_count} events saved to {keyboard_csv}")
    
    print(f"Recording to {keyboard_csv}... Press ESC to stop.")
    keyboard.hook(on_event)
    keyboard.wait('esc')


async def start_keyboard(keyboard_csv: str):
    """Start keyboard recording as an async task.
    
    Runs the blocking keyboard listener in a thread pool so it doesn't
    block the event loop. Other coroutines can continue running.

    Args:
        keyboard_csv (str): Path to the output CSV file.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _keyboard_listener, keyboard_csv)


if __name__ == "__main__":
    asyncio.run(start_keyboard("events.csv"))