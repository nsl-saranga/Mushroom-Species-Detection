import os;

directory_path = 'Mushrooms'

for category_name in os.listdir(directory_path):
    category_path = os.path.join(directory_path, category_name)

    if os.path.isdir(category_path):
        count = 1
        for image_name in os.listdir(category_path):
            if image_name.lower().endswith(("jpg", "jpeg","png")):
                file_extension = os.path.splitext(image_name)[1]
                # print(file_extension)
                new_name = f"{category_name}_{count}{file_extension}"
                old_path = os.path.join(category_path, image_name)
                new_path = os.path.join(category_path, new_name)
                os.rename(old_path, new_path)
                print(new_path)
                count += 1