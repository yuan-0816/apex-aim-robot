import win32api
import win32con
import pyautogui
import time

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()  # 1920x1080

Center_X = int(SCREEN_WIDTH / 2)
Center_Y = int(SCREEN_HEIGHT / 2)

Cx = Center_X * 65535 / SCREEN_WIDTH
Cy = Center_Y * 65535 / SCREEN_HEIGHT

Grab_Screen_Width = SCREEN_WIDTH * 1 / 4
Grab_Screen_Height = SCREEN_HEIGHT * 1 / 3

SCREEN_AREA_UPPER_LEFT_X = int(SCREEN_WIDTH * 3 / 8)
SCREEN_AREA_UPPER_LEFT_Y = int(SCREEN_HEIGHT / 3)
SCREEN_AREA_LOWER_RIGHT_X = int(SCREEN_WIDTH * 5 / 8)
SCREEN_AREA_LOWER_RIGHT_Y = int(SCREEN_HEIGHT * 2 / 3)

Compensate_X = SCREEN_WIDTH * 3 / 8
Compensate_Y = SCREEN_HEIGHT / 3

SCREEN_AREA_UPPER_LEFT = (SCREEN_AREA_UPPER_LEFT_X, SCREEN_AREA_UPPER_LEFT_Y)
SCREEN_AREA_LOWER_RIGHT = (SCREEN_AREA_LOWER_RIGHT_X, SCREEN_AREA_LOWER_RIGHT_Y)

def main():

    while True:
        print(pyautogui.position())
        nx = 2 * 65535 / SCREEN_WIDTH
        ny = 1 * 65535 / SCREEN_HEIGHT
        if win32api.GetAsyncKeyState(0x30):
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(nx), int(ny))
            time.sleep(0.3)





if __name__ == "__main__":
    main()