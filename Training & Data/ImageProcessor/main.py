import os
import cv2

# Define the directory here.
TARGET_DIRECTORY = os.path.join("", "", "")


def main() -> None:
    for file_name in os.listdir(TARGET_DIRECTORY):
        # If it starts with "processed_" then we don't need to process this image.
        # If this image does not end with .png or .jpg, then it's not an image file.
        if file_name.startswith("processed_") or (not file_name.endswith(".png") and not file_name.endswith(".jpg")):
            continue

        img = cv2.imread(os.path.join(TARGET_DIRECTORY, file_name), cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(img, (100, 100))
        cv2.imwrite(os.path.join(TARGET_DIRECTORY, "processed_" + file_name), resized_image)


if __name__ == '__main__':
    main()
