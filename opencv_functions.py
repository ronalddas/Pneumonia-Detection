"""
import os
filename="9ee07fd5-612d-4a33-89a6-12fa8ce9bb48.dcm.png"
ss="python3 -m keras_segmentation predict  --checkpoints_path='path_to_checkpoints'  --input_path='static/uploads/100983183.jpeg'  --output_path='static/predictions/'"
os.system(ss)
"""
import cv2 as cv
import traceback
#original_image_loc="/home/ronald/WORK/Personal/PS/static/uploads/0100515c-5204-4f31-98e0-f35e4b00004a.dcm.png"
#segmented_image_loc="/home/ronald/WORK/Personal/PS/static/predictions/0100515c-5204-4f31-98e0-f35e4b00004a.dcm.png"
def merge_images(original,segmented):
    try:
        alpha=0.8
        original_image=cv.imread(original)
        segmented_image=cv.imread(segmented)
        original_image=cv.resize(original_image,(600,600))
        segmented_image=cv.resize(segmented_image,(600,600))
        print("done resizing")
        beta=1-alpha
        blended_image=cv.addWeighted(original_image,alpha,segmented_image,beta,0.0)
        filename = original.split("/")[-1]
        cv.imwrite("static/merged/"+filename,blended_image)
    except Exception:
        traceback.print_exc()


#merge_images(original_image_loc,segmented_image_loc)