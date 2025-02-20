import streamlit as st
import numpy as np
import zipfile
import cv2
import io
import os
from PIL import Image, ImageOps
from enhanced_lwfa_algo import (
    local_water_filling,
    separate_umbra_and_penumbra,
    umbra_enhancement,
    penumbra_enhancement,
)
from threading import Lock

st.set_page_config(layout="wide")
MAX_UPLOAD = 5

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1   

if "uploader_disabled" not in st.session_state:
    st.session_state["uploader_disabled"] = False

if "processed_images" not in st.session_state:
    st.session_state["processed_images"] = {}

# === SIDE BAR ===
with st.sidebar:
    st.title("Document Image Shadow Detection and Removal App")
    st.subheader("Using Enhanced Local Water-Filling Algorithm")

    def check_upload_limit():
        # -- Get the length of the uploaded images
        current_value = st.session_state["uploader_key"]
        num_uploaded_images = len(st.session_state[current_value])
        # -- Check if the number of uploaded images exceeds the limit
        if num_uploaded_images >= MAX_UPLOAD:
            st.session_state["uploader_disabled"] = True
        else:
            st.session_state["uploader_disabled"] = False

    uploaded_images = st.file_uploader(
        "Please upload document image/s", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key=st.session_state["uploader_key"],
        on_change=check_upload_limit,                         
        disabled=st.session_state["uploader_disabled"],
    )
    
    # -- If length of uploaded images is greater than 1
    if len(uploaded_images) > 1 and len(uploaded_images) <= MAX_UPLOAD:
        st.divider()
            
        def clear_uploader_key():
            # -- Delete last uploader key from session state
            last_value = st.session_state["uploader_key"]
            del st.session_state[last_value]
            # -- Create new uploader key and reset session states
            st.session_state["uploader_key"] += 1
            st.session_state["uploader_disabled"] = False
            st.session_state["processed_images"] = {}

        clear_button = st.button(
            "Clear Uploaded Images",
            on_click=clear_uploader_key,
            type="primary",
            icon=":material/delete_forever:",
            use_container_width=True,
        )

        # If there are processed images, batch download is enabled.
        if "processed_images" in st.session_state and st.session_state["processed_images"] is not None:
            def create_zip():
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zipf:
                    for filename, image in st.session_state["processed_images"].items():
                        img_bytes = cv2.imencode(".png", image)[1].tobytes()
                        zipf.writestr(filename, img_bytes)
                zip_buffer.seek(0)
                return zip_buffer

            zip_file = create_zip()
            batch_download_button = st.download_button(
                label="Batch Download Images",
                data=zip_file,
                file_name="processed_images.zip",
                mime="application/zip",
                type="primary",
                icon=":material/folder_zip:",
                use_container_width=True
            )

            if batch_download_button:
                st.toast("Downloading all output images...", icon="📥")



# === TAB FRAGMENT ===
@st.fragment
def tab_fragment(index_ref, input_image_ref):
    # -- Initialize initial values for parameter inputs
    toggle_figures_val = st.session_state.setdefault(f"toggle_figures_{index_ref}", False)
    toggle_histogram_val = st.session_state.setdefault(f"toggle_histogram_{index_ref}", False)
    max_iterations_val = st.session_state.setdefault(f"max_iterations_input_{index_ref}", 3)
    penumbra_size_val = st.session_state.setdefault(f"penumbra_size_input_{index_ref}", 2)
    upper_bounds_val = st.session_state.setdefault(f"upper_bounds_input_{index_ref}", 16)
    
    try:
        # -- Initialize input image
        image = Image.open(input_image_ref)
        image = ImageOps.exif_transpose(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        process_toast = st.toast("Processing image...", icon="⏳")

        # -- Process input image through the Enhanced Local Water-Filling Algorithm
        shading_map = local_water_filling(image, max_iterations=max_iterations_val)
        shading_map_rgb = cv2.cvtColor(shading_map, cv2.COLOR_BGR2RGB)

        median_map, umbra_mask, penumbra_mask, colored_mask, histogram_figure, channel_masks = separate_umbra_and_penumbra(image, shading_map, upper_bounds=upper_bounds_val, penumbra_size=penumbra_size_val)
        colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)

        enhanced_umbra, global_color, local_color = umbra_enhancement(image, median_map, umbra_mask)
        enhanced_umbra_rgb = cv2.cvtColor(enhanced_umbra, cv2.COLOR_BGR2RGB)

        output_image = penumbra_enhancement(enhanced_umbra, penumbra_mask, global_color, local_color)
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        process_toast.toast("Image processed!", icon="✅")

        # -- If toggled, display all ELWF figures, else display input and output images
        if toggle_figures_val:
            top_row = st.columns(2)
            with top_row[0]:
                st.subheader("Input Image", anchor=False)
                st.image(input_image_ref, use_container_width=True)
            with top_row[1]:
                st.subheader("Output Image", anchor=False)
                st.image(output_image_rgb, use_container_width=True)
            bottom_row = st.columns(3)
            with bottom_row[0]:
                st.subheader("Shading Map", anchor=False)
                st.image(shading_map_rgb, use_container_width=True)
            with bottom_row[1]:
                st.subheader("Colored Mask", anchor=False)
                st.image(colored_mask_rgb, use_container_width=True)
            with bottom_row[2]:
                st.subheader("Enhanced Umbra", anchor=False)
                st.image(enhanced_umbra_rgb, use_container_width=True)
        else:
            two_cols = st.columns(2)
            with two_cols[0]:
                st.subheader("Input Image", anchor=False)
                st.image(input_image_ref, use_container_width=True)
            with two_cols[1]:
                st.subheader("Output Image", anchor=False)
                st.image(output_image_rgb, use_container_width=True)

        # -- If toggled, display the channel-wise histogram analysis
        if toggle_histogram_val:
            st.divider()
            st.subheader("Channel-wise Histogram Analysis", anchor=False)

            _lock = Lock()
            with _lock:
                st.pyplot(fig=histogram_figure, use_container_width=True)
            
            mask_cols = st.columns(3)
            with mask_cols[0]:
                st.image(channel_masks[0], caption="Blue Channel Mask", use_container_width=True)
            with mask_cols[1]:
                st.image(channel_masks[1], caption="Green Channel Mask", use_container_width=True)
            with mask_cols[2]:
                st.image(channel_masks[2], caption="Red Channel Mask", use_container_width=True)

        # -- Display algorithm parameters
        st.divider()
        st.subheader("Algorithm Parameters", anchor=False)
        st.toggle("Show Enhanced Local Water-Filling Figures", key=f"toggle_figures_{index_ref}")
        st.toggle("Show Channel-wise Histogram Analysis", key=f"toggle_histogram_{index_ref}")

        param_cols = st.columns(2)
        with param_cols[0]:
            st.number_input(
                "Max Iterations",
                min_value=3,
                step=1,
                key=f"max_iterations_input_{index_ref}",
            )
        with param_cols[1]:
            st.number_input(
                "Penumbra Size",
                min_value=2,
                step=1,
                key=f"penumbra_size_input_{index_ref}",
            )

        st.slider(
            "Upper Bounds",
            min_value=1,
            max_value=256,
            step=1,
            key=f"upper_bounds_input_{index_ref}",
        )

        st.divider()

        # -- Get input image file name and remove extension
        filename_wo_ext, _ = os.path.splitext(input_image_ref.name)
        # -- Store processed image for batch download
        st.session_state["processed_images"][f"processed_{filename_wo_ext}.png"] = output_image
        # -- Convert processed image into bytes
        output_buffer = io.BytesIO()
        output_buffer.write(cv2.imencode(".png", output_image)[1].tobytes())
        output_buffer.seek(0)

        download_button = st.download_button(
            label="Download Output Image",
            data=output_buffer,
            file_name=f"processed_{filename_wo_ext}.png",
            mime="image/png",
            type="primary",
            icon=":material/download:",
        )

        if download_button:
            process_toast.toast("Downloading output image...", icon="📥")

        # FOR DEBUGGING
        st.write(st.session_state)

    except Exception as e:
        st.error("An error occurred while processing the image: " + str(e))
        if st.button("Undo Action", icon=":material/undo:"):
            st.rerun()



# === MAIN PAGE ===
main_container = st.container()
with main_container:
    current_value = st.session_state["uploader_key"]
    input_images = st.session_state[current_value]

    # -- Check if there are uploaded images or if the number of uploaded images exceeds the limit
    if len(input_images) == 0:
        st.info("Please upload at least one image to begin the process.")
    elif len(input_images) > MAX_UPLOAD:
        st.warning(f"You can only upload {MAX_UPLOAD} images at a time. Please remove an image to upload a new one.")
    else:
        # -- Get tab names for each uploaded image
        tab_names = [img_name.name for img_name in input_images]
        # -- Create tabs for each uploaded image
        tabs = st.tabs(tab_names)
        for i, (tab, img) in enumerate(zip(tabs, input_images)):
            with tab:
                tab_fragment(i, img)
