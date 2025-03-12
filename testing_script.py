import cv2
import json
import numpy as np
import onnxruntime
import pandas as pd
from matplotlib import pyplot as plt

# Define fixed X values for plotting
X_IDXS = np.array([0., 0.1875, 0.75, 1.6875, 3., 4.6875, 6.75, 9.1875, 12., 
                   15.1875, 18.75, 22.6875, 27., 31.6875, 36.75, 42.1875, 48., 
                   54.1875, 60.75, 67.6875, 75., 82.6875, 90.75, 99.1875, 
                   108., 117.1875, 126.75, 136.6875, 147., 157.6875, 168.75, 
                   180.1875, 192.])

def parse_image(frame):
    """Process the input image into YUV format and reshape."""
    H = (frame.shape[0] * 2) // 3
    W = frame.shape[1]
    print(f"Image shape after processing: {H}x{W}")

    parsed = np.zeros((6, H // 2, W // 2), dtype=np.float32)
    parsed[0] = frame[0:H:2, 0::2]
    parsed[1] = frame[1:H:2, 0::2]
    parsed[2] = frame[0:H:2, 1::2]
    parsed[3] = frame[1:H:2, 1::2]
    parsed[4] = frame[H:H + H // 4].reshape((-1, H // 2, W // 2))
    parsed[5] = frame[H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))

    return parsed

def seperate_points_and_std_values(df):
    """Separate the model output into lane points and standard deviation values."""
    points = df.iloc[lambda x: x.index % 2 == 0]
    std = df.iloc[lambda x: x.index % 2 != 0]
    return pd.concat([points], ignore_index=True), pd.concat([std], ignore_index=True)

def main():
    model_path = "selfdrive/modeld/models/supercombo.onnx"
    image_path = "frame.jpg"

    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Image not found or could not be loaded!")
        return

    width, height = 512, 256
    frame_resized = cv2.resize(frame, (width, height))
    frame_yuv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV_I420)
    parsed = parse_image(frame_yuv)

    parsed_arr = np.array([parsed])
    parsed_arr.resize((1, 12, 128, 256))

    session = onnxruntime.InferenceSession(model_path, None)

    # Get model inputs
    input_imgs = session.get_inputs()[0].name
    big_input_imgs = session.get_inputs()[1].name
    desire = session.get_inputs()[2].name
    traffic_convention = session.get_inputs()[3].name
    nav_features = session.get_inputs()[4].name
    nav_instructions = session.get_inputs()[5].name
    features_buffer = session.get_inputs()[6].name
    output_name = session.get_outputs()[0].name

    # Prepare inputs - maybe discarding some information -> not that good detection result?
    big_input_imgs_data = np.zeros((1, 12, 128, 256), dtype=np.float16)
    desire_data = np.zeros((1, 100, 8), dtype=np.float16)
    traffic_convention_data = np.zeros((1, 2), dtype=np.float16)
    nav_features_data = np.zeros((1, 2), dtype=np.float16)
    nav_instructions_data = np.zeros((1, 100, 1), dtype=np.float16)
    features_buffer_data = np.zeros((1, 99, 512), dtype=np.float16)

    # Run inference
    result = session.run([output_name], {
        input_imgs: parsed_arr.astype('float16'),
        big_input_imgs: big_input_imgs_data,
        desire: desire_data,
        traffic_convention: traffic_convention_data,
        nav_features: nav_features_data,
        nav_instructions: nav_instructions_data,
        features_buffer: features_buffer_data
    })

    res = np.array(result)

    lanes = res[:, :, 4955:4955+528]
    lane_road = res[:, :, 5483:5483+264]

    lanes_flat = lanes.flatten()
    df_lanes = pd.DataFrame(lanes_flat)

    ll_t = df_lanes[0:66]
    ll_t2 = df_lanes[66:132]
    points_ll_t, _ = seperate_points_and_std_values(ll_t)
    points_ll_t2, _ = seperate_points_and_std_values(ll_t2)

    l_t = df_lanes[132:198]
    l_t2 = df_lanes[198:264]
    points_l_t, _ = seperate_points_and_std_values(l_t)
    points_l_t2, _ = seperate_points_and_std_values(l_t2)

    # Compute middle path
    middle = points_ll_t2.add(points_l_t, fill_value=0) / 2

    # Plot results
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(middle, X_IDXS, color="g", label="Predicted Path")
    axes[0].scatter(points_ll_t2, X_IDXS, color="y", label="Lane Line Left")
    axes[0].scatter(points_l_t, X_IDXS, color="y", label="Lane Line Right")

    axes[0].set_title("Road and Lane Lines Prediction")
    axes[0].set_xlabel("Lane & Road Data")
    axes[0].set_ylabel("Range")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].imshow(frame_rgb)
    axes[1].axis("off")  # Hide axis
    axes[1].set_title("Original Image")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
