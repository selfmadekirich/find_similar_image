import streamlit as st
from imports import *

from ImageWorkerKNN import ImageWorkerKNN

def run():
    image_worker = ImageWorkerKNN('db.csv')
    st.title('Find similar images')
    file = st.file_uploader('Upload image')
    if file is not None:
        buf = np.frombuffer(file.getbuffer(), dtype=np.uint8)
        img = cv.imdecode(buf, cv.IMREAD_COLOR)
        img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.image(img)
        st.title('5 most similar images')
        for ind in image_worker.find_n_similar(img):
            img = image_worker.get_image(ind)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            st.image(img)
    pass

if __name__ == '__main__':
    run()