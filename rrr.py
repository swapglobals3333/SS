try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install streamlit: pip install streamlit")

import numpy as np
import os

# --- Simulated Homomorphic Encryption (replace with real HE lib as needed) ---
class HE:
    def encrypt(self, value):
        # Placeholder for encryption - just returns value
        return value

    def compare(self, encrypted_val1, encrypted_val2):
        # Placeholder comparison on plaintext values
        return encrypted_val1 > encrypted_val2

he = HE()

# --- Helper function for matching ---
def get_good_matches(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

# --- Streamlit UI ---
st.title("Surface Sleuth - Fingerprint Matcher")

DB_PATH = r"D:\Majorproject\SOCOFing\Real"  # Change to your database path

uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image as grayscale
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    sample = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if sample is None:
        st.error("Failed to decode the uploaded image. Please upload a valid image.")
    else:
        # Enhance contrast for better feature detection
        sample = cv2.equalizeHist(sample)
        st.image(sample, caption="Query Fingerprint (Enhanced)", channels="GRAY")

        # Initialize ORB with more features
        orb = cv2.ORB_create(nfeatures=2000)

        kp1, des1 = orb.detectAndCompute(sample, None)

        if des1 is None or len(des1) == 0:
            st.error("No descriptors found in the uploaded fingerprint. Try a clearer image.")
        else:
            best_score = -1
            best_filename = None
            best_image = None
            best_kp2 = None
            best_matches = []

            # Loop over database images
            for file in os.listdir(DB_PATH):
                full_path = os.path.join(DB_PATH, file)
                if not os.path.isfile(full_path):
                    continue

                img = cv2.imread(full_path, 0)
                if img is None:
                    continue

                img = cv2.equalizeHist(img)  # Enhance contrast for DB image

                kp2, des2 = orb.detectAndCompute(img, None)
                if des2 is None or len(des2) == 0:
                    continue

                good_matches = get_good_matches(des1, des2)
                min_keypoints = min(len(kp1), len(kp2))
                if min_keypoints == 0:
                    continue

                match_percent = (len(good_matches) / min_keypoints) * 100
                encrypted_score = he.encrypt(match_percent)

                if he.compare(encrypted_score, best_score):
                    best_score = encrypted_score
                    best_filename = file
                    best_image = img
                    best_kp2 = kp2
                    best_matches = good_matches

            if best_image is not None:
                st.success(f"Best Match: {best_filename}\nMatching Percentage: {best_score:.2f}%")

                # Draw matches
                result = cv2.drawMatches(sample, kp1, best_image, best_kp2, best_matches, None, flags=2)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption="Matching Result", use_container_width=True)
            else:
                st.warning("No suitable match found in the database.")
