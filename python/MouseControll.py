import numpy as np
import pyautogui


def init_pyautogui():
    pyautogui.FAILSAFE = False
    return




if __name__ == "__main__":    
    print(f"Check mouse controll class")
    

    # init_pyautogui()

    # screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor
    # print(f"{screenWidth} - {screenHeight}")

    print('Press Ctrl-C to quit.')

    try:
        while True:
            x, y = pyautogui.position()
            positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
            print(positionStr, end='')
            print('\b' * len(positionStr), end='', flush=True)
            # print(positionStr)

            screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor
            # print(f"Monitor: {screenWidth} - {screenHeight}")  

    except KeyboardInterrupt:
        print('\nDone.')

    # Main monitor: 0->1920, 