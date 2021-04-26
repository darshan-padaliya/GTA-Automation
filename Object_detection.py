import numpy as np
from PIL import ImageGrab
import cv2   
import time
from directkeys import PressKey, ReleaseKey
import keys
from img_processing import RGB_color_selection, convert_hsl, HSL_color_selection, canny


def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # processed_img = convert_hsl(processed_img)
    # processed_img = HSL_color_selection(processed_img)
    # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
    # processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)
    # processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    # processed_img = cv2.GaussianBlur(processed_img, (13, 13), 0)
    # vertices = np.array([[0,460],[0,350],[160,250],[500,250],[660,350],[660,460]], np.int32)
    # processed_img = roi(processed_img, [vertices])
    processed_img = canny(processed_img)
    return processed_img





def main(): 
    last_time = time.time()
  
    while(True):
        
        screen = np.array(ImageGrab.grab(bbox=(0, 0, 660, 460)))
        processed_img = process_img(screen)
        cv2.imshow('Trial', processed_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        print(f'Loop took {time.time() - last_time} seconds.') 
        last_time = time.time()
    

if __name__ == '__main__':
    main()

