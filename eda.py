import os;
import cv2

directory_path = 'Mushrooms'
image_paths = []

for category_name in os.listdir(directory_path):
    category_path = os.path.join(directory_path, category_name)

    if os.path.isdir(category_path):
        for image_name in os.listdir(category_path):
            if image_name.lower().endswith(("jpg", "jpeg","png")):
                image_path = os.path.join(category_path, image_name)
                image_paths.append(image_path)

index = 5000  # Change this to the index you want

if 0 <= index < len(image_paths):
    selected_image_path = image_paths[index]
    print(f"Loading image: {selected_image_path}")

    img = cv2.imread(selected_image_path)

    # Display the image
    cv2.imshow("Selected Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Invalid index. Please choose a value between 0 and", len(image_paths) - 1)
