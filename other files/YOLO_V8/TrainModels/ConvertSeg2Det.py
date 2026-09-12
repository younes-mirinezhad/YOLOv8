import os, shutil

segDatasetDir = "/path/to/Dataset/"
outputDir = "/our/dir/"
img_type = '.jpg'
annot_type = '.txt'

def convert_segmentation_to_detection(input_dir, output_dir):
    if os.path.exists(output_dir):
        os.rmdir(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            newLinesData = []

            with open(input_path, 'r') as file:
                for line in file:
                    lineData = line.split(' ')
                    classID = lineData[0]
                    del lineData[0] # drop class id
                    coordinates = list(zip(lineData[::2], lineData[1::2]))  # Pairs of (x, y)
                    # Extract min and max values
                    min_x = float(min(coord[0] for coord in coordinates))
                    max_x = float(max(coord[0] for coord in coordinates))
                    min_y = float(min(coord[1] for coord in coordinates))
                    max_y = float(max(coord[1] for coord in coordinates))
                    cx = (min_x + max_x) / 2
                    cy = (min_y + max_y) / 2
                    w = abs(max_x - min_x)
                    h = abs(max_y - min_y)
                    newLine = classID + " " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h)
                    newLinesData.append(newLine)
            
            if len(newLinesData) > 0:
                with open(output_path, "w") as file:
                    for line in newLinesData:
                        file.write(line + '\n')
                shutil.copy(input_path.replace(annot_type, img_type), output_path.replace(annot_type, img_type))

convert_segmentation_to_detection(segDatasetDir, outputDir)