import os
import cv2

TARGET_DIRECTORY = os.path.join("", "", "")


def main() -> None:
    for file in os.listdir(TARGET_DIRECTORY):
        # If it starts with "processed_" then we don't need to process this image.
        # If this image does not end with .png or .jpg, then it's not an image file.
        if file.startswith("processed_") or (not file.endswith(".png") and not file.endswith(".jpg")):
            continue

        img_file_path = os.path.join(TARGET_DIRECTORY, "processed_" + file)

        img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(img, (100, 100))
        cv2.imwrite(img_file_path, resized_image)


if __name__ == '__main__':
    main()
