from detector import plate_detection_obj
import cv2
import easyocr
import traceback
from PIL import Image


class LicensePlateRecognizer():
    def __init__(self):
        print("Loading Models.....")
        self.model = plate_detection_obj
        self.reader = easyocr.Reader(lang_list = ['en'], gpu=False, model_storage_directory='./easyocr',
                        user_network_directory='./easyocr', detect_network="craft", 
                        recog_network='generation2', download_enabled=True, 
                        detector=True, recognizer=True, verbose=True, 
                        quantize=True, cudnn_benchmark=False)
        print("Finished Loading!")

    def easyocr_image(self, img):
        cv2.imwrite('./easyocr/temp.jpeg', img)
        img1 = cv2.imread('./easyocr/temp.jpeg')
        if img1.shape[0] < 80 or img1.shape[1] < 200:
            img1 = cv2.resize(img1, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)

        result = self.reader.readtext(img1, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                    workers = 0, blocklist = None, detail = 1,\
                    rotation_info = None, paragraph = False, min_size = 20,\
                    contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                    text_threshold = 0.2, low_text = 0.4, link_threshold = 0.4,\
                    canvas_size = 2560, mag_ratio = 1.,\
                    slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                    width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
                    threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
                    output_format='standard')

        text = ""
        for res in result:
            if res[2] >= 0.2:
                text += res[1]
        
        return str(text)


    def run(self, img):
        try:
            results = self.model.predict(img)
            plates = []
            for cropped in results:
                text = self.easyocr_image(cropped)
                if text != "":
                    plates.append(text)

            return plates
        except:
            print(traceback.print_exc())
            return []

service_obj = LicensePlateRecognizer()