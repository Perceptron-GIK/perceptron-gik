import keyboard
import csv

# Record events until 'esc' is pressed
events = keyboard.record(until='esc')

# Save to CSV file
with open('events1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['event_type', 'scan_code', 'name', 'time'])
    for event in events:
        writer.writerow([event.event_type, event.scan_code, event.name, event.time])