import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from io import BytesIO
from PIL import Image
import io

st.set_page_config(page_title="Matrix Transformations & Image Filters", layout="wide")

# =================== Utility Image Loader =====================

def load_image(uploaded):
    if uploaded is None:
        return None
    try:
        image = Image.open(BytesIO(uploaded.read()))
        img = np.array(image)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        img = np.clip(img, 0, 1)
        return img
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}. Please upload a valid JPG, JPEG, or PNG image.")
        return None

def show(img, caption=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img, vmin=0, vmax=1)
    ax.axis("off")
    if caption:
        ax.set_title(caption)
    st.pyplot(fig)
    plt.close(fig)

def get_image_bytes(img, format='PNG'):
    """Convert numpy array to bytes for download."""
    img_uint8 = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img_uint8 = np.stack([img_uint8] * 3, axis=-1)  # Convert grayscale to RGB
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

# =================== Affine Transform Functions =====================

def transform(img, M, interp="nearest"):
    h, w = img.shape[:2]
    ys, xs = np.indices((h, w))
    coords = np.stack([xs.ravel(), ys.ravel(), np.ones(h*w)], axis=0)
    M_inv = np.linalg.inv(M)
    mapped = M_inv @ coords
    src_x, src_y = mapped[0].reshape(h, w), mapped[1].reshape(h, w)

    def sample(ch):
        if interp == "nearest":
            xi = np.round(src_x).astype(int)
            yi = np.round(src_y).astype(int)
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            output = np.zeros_like(src_x)
            output[valid] = ch[yi[valid], xi[valid]]
            return output
        else:
            x0=np.floor(src_x).astype(int); x1=x0+1
            y0=np.floor(src_y).astype(int); y1=y0+1
            x0=np.clip(x0,0,w-1); x1=np.clip(x1,0,w-1)
            y0=np.clip(y0,0,h-1); y1=np.clip(y1,0,h-1)
            Ia=ch[y0,x0]; Ib=ch[y1,x0]; Ic=ch[y0,x1]; Id=ch[y1,x1]
            wa=(x1-src_x)*(y1-src_y)
            wb=(x1-src_x)*(src_y-y0)
            wc=(src_x-x0)*(y1-src_y)
            wd=(src_x-x0)*(src_y-y0)
            return wa*Ia + wb*Ib + wc*Ic + wd*Id

    if img.ndim==2:
        return sample(img)
    else:
        return np.stack([sample(img[...,c]) for c in range(img.shape[2])], axis=2)

# =================== Convolution Filters =====================

def apply_kernel(im, k):
    h, w = im.shape[:2]
    kh, kw = k.shape
    pad = kh//2
    out=np.zeros_like(im)

    if im.ndim==3:
        for c in range(im.shape[2]):
            padded = np.pad(im[...,c], pad, mode="edge")
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kh, j:j+kw]
                    out[i,j,c] = np.sum(region*k)
    else:
        padded=np.pad(im,pad,mode="edge")
        for i in range(h):
            for j in range(w):
                region=padded[i:i+kh,j:j+kw]
                out[i,j]=np.sum(region*k)

    return np.clip(out,0,1)

kernels = {
    "Blur": np.ones((3,3))/9,
    "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    "Edge": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
}

# =================== Background Removal =====================

def remove_bg_hsv(image, threshold=0.4):
    hsv=cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_RGB2HSV)/255.0
    mask=hsv[...,2]>threshold
    bg_removed=image.copy()
    bg_removed[~mask]=1
    return bg_removed

def remove_bg_grabcut(image):
    img=(image*255).astype(np.uint8)
    mask=np.zeros(img.shape[:2],np.uint8)
    bgm=np.ones((1,65),np.float64)
    fgm=np.ones((1,65),np.float64)
    rect=(10,10,img.shape[1]-10,img.shape[0]-10)
    cv2.grabCut(img,mask,rect,bgm,fgm,5,cv2.GC_INIT_WITH_RECT)
    mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return (img*mask2[...,None])/255.0

# =================== UI =====================

st.title("Matrix Transformations & Image Processing")

uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg","jpeg","png"])
img = load_image(uploaded)

tab1, tab2, tab3 = st.tabs(["Geometric Transformations","Image Filtering","Background Removal"])

# ==== TAB 1 – Transformations ====
with tab1:
    st.subheader("Geometric Transformations (Matrix-based)")
    if img is None:
        st.info("Upload image first.")
    else:
        show(img, "Original")

        t = st.selectbox("Select Transform",["Translation","Scaling","Rotation","Shearing","Reflection"])
        h,w=img.shape[:2]
        center=(w/2,h/2)

        if t=="Translation":
            tx=st.slider("Shift X", -200,200,0)
            ty=st.slider("Shift Y", -200,200,0)
            M=np.array([[1,0,tx],[0,1,ty],[0,0,1]])

        elif t=="Scaling":
            sx=st.slider("Scale X",0.2,3.0,1.0)
            sy=st.slider("Scale Y",0.2,3.0,1.0)
            cx,cy=center
            M=np.array([[sx,0,cx*(1-sx)],[0,sy,cy*(1-sy)],[0,0,1]])

        elif t=="Rotation":
            angle=st.slider("Angle",-180,180,0)
            theta=np.radians(angle)
            c,s=np.cos(theta),np.sin(theta)
            cx,cy=center
            M=np.array([
                [c,-s,cx-c*cx+s*cy],
                [s, c,cy-s*cx-c*cy],
                [0,0,1]
            ])

        elif t=="Shearing":
            sh=st.slider("Shear X",-1.0,1.0,0.0)
            M=np.array([[1,sh,0],[0,1,0],[0,0,1]])

        elif t=="Reflection":
            axis=st.radio("Axis",["Vertical","Horizontal"])
            if axis=="Vertical":
                M=np.array([[-1,0,w],[0,1,0],[0,0,1]])
            else:
                M=np.array([[1,0,0],[0,-1,h],[0,0,1]])

        interp=st.radio("Interpolation",["nearest","bilinear"])
        out=transform(img,M,interp)
        show(out,"Transformed Output")

        # Download button
        img_bytes = get_image_bytes(out)
        st.download_button(
            label="Download Transformed Image",
            data=img_bytes,
            file_name=f"transformed_{t.lower()}.png",
            mime="image/png"
        )

# ==== TAB 2 – Filtering ====
with tab2:
    st.subheader("Image Filtering (Convolution)")

    if img is None:
        st.info("Upload image first.")
    else:
        show(img,"Original")
        choice=st.selectbox("Filter",["Blur","Sharpen","Edge"])
        kernel=kernels[choice]
        out=apply_kernel(img,kernel)
        show(out,choice+" Output")

        # Download button
        img_bytes = get_image_bytes(out)
        st.download_button(
            label=f"Download {choice} Filtered Image",
            data=img_bytes,
            file_name=f"filtered_{choice.lower()}.png",
            mime="image/png"
        )

# ==== TAB 3 – Background Removal ====
with tab3:
    st.subheader("Optional Feature: Background Removal")

    if img is None:
        st.info("Upload image first.")
    else:
        method=st.selectbox("Method",["Color Thresholding (HSV)","GrabCut"])

        if method=="Color Thresholding (HSV)":
            thr=st.slider("Brightness Threshold",0.1,1.0,0.4)
            out=remove_bg_hsv(img,thr)
        elif method=="GrabCut":
            out=remove_bg_grabcut(img)

        show(img,"Original")
        show(out,"Background Removed")
