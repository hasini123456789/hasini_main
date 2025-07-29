def clear_file(file):
    with open(file,"w") as f:
        f.write("")
clear_file("image_ids.txt")
clear_file("used_ids.txt")

