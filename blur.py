import numpy as np
import cv2
import matplotlib.pyplot as plt


class Blur():

    def __init__(self):
        super().__init__()


    def motion_blur(self, image, size):  
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        blurred_image = cv2.filter2D(image, -1, kernel_motion_blur)

        return blurred_image
    
    def create_cadr(self, image):
        height, width = np.shape(image)[:2]
        height_s = int(0.01 * np.random.randint(5, 20)*height)
        height_e = int( 0.01 * np.random.randint(70, 100) * (height - height_s) )
        width_s = int( 0.01 * np.random.randint(5, 20)* width )
        width_e =  int( 0.01 * np.random.randint(70, 100) * (width - width_s) )

        return [width_s, height_s, width_e, height_e ]


    def face_landmarks(self, image ):

        frontal_cascade_path =  "project\\stunmaster\\improve_quality\\my_model\\haarcascade_frontalface_default.xml"
        frontal_face_classifier =  cv2.CascadeClassifier(frontal_cascade_path)
        frontal_landmark = frontal_face_classifier.detectMultiScale(image)

        if len(frontal_landmark)!=0:
            frontal_landmark = frontal_landmark[0]
        if len(frontal_landmark) and frontal_landmark[2] >= 0.40 * image.shape[1] :
            return frontal_landmark


        profile_cascade_path = "project\\stunmaster\\improve_quality\\my_model\\haarcascade_profileface.xml"
        profile_classifier = cv2.CascadeClassifier(profile_cascade_path)
        profile_landmark = profile_classifier.detectMultiScale(image)

        if len(profile_landmark):
            return profile_landmark[0]
        
        else:
            pass



    


    def extract_head(self, image, rect):
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)

        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = image*mask2[:,:,np.newaxis]
        return img

    def crop_image(self, image, rect):
        img = image[ rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], : ]

        return img

    



    def add_cadr(self, cropped_image, real_image, face_rect):
        real_image[face_rect[1]:face_rect[1]+face_rect[3], face_rect[0]:face_rect[0]+face_rect[2]] = cropped_image

        return real_image

    def blurring(self, real_image):
        face_rect = self.face_landmarks(real_image)
        face_image = self.crop_image(real_image, face_rect)
        face_feature_rect = self.create_cadr(face_image)
        face_feature_image = self.crop_image(face_image, face_feature_rect)
        blur_size = np.random.randint(30, 100, 1)
        blured_face_feature = self.motion_blur(face_feature_image, blur_size[0])
        blured_face = self.add_cadr(blured_face_feature, face_image, face_feature_rect)
        blure_image = self.add_cadr(blured_face, real_image, face_rect)

        return blure_image



        # return output

    def detect_head(self, image):
        face = self.face_landmarks(image)

        for (x, y, w, h) in face:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(img_rgb)
        return img_rgb
        


blur = Blur()