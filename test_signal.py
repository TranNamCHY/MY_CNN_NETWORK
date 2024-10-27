import signal
import time

class MyClass:
    def __init__(self):
        print("MyClass initialized.")
    
    # Method to handle the signal
    def signal_handler(self, signum, frame):
        print("Received signain class method")
        exit(0)

# Create an instance of the class
obj = MyClass()

# Assign the instance method as the signal handler
signal.signal(signal.SIGINT, obj.signal_handler)

print("Press Ctrl+C to trigger the signal handler.")
while True:
    time.sleep(1)
