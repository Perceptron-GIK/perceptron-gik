import keyboard, csv, asyncio, os, threading
from datetime import datetime
from typing import Callable, Optional

# This is used to signal the other threads (asyncio) that 
stop_event = threading.Event()

def _keyboard_listener(keyboard_csv: str):
    """Keyboard listener that records events to CSV."""
    event_count = 0
    
    # If file doesn't exist create it
    if not os.path.exists(keyboard_csv):
        with open(keyboard_csv, "w") as f:
            writer = csv.writer(f)
            writer.writerow(['event_type', 'scan_code', 'name', 'time'])
    
    f = open(keyboard_csv, 'a', newline='')
    writer = csv.writer(f)
    
    def on_event(event):
        nonlocal event_count
        writer.writerow([event.event_type, event.scan_code, event.name, event.time])
        event_count += 1
        
        # Save every 10 keys
        if event_count % 10 == 0:
            f.flush()  
            
        # File handling when we close the writing
        if event.name == 'esc' and event.event_type == 'down':
            keyboard.unhook_all()
            f.flush()
            f.close()
            stop_event.set()  # Signal all tasks to stop
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
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _keyboard_listener, keyboard_csv)


if __name__ == "__main__":
    asyncio.run(start_keyboard("events.csv"))