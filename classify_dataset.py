import os

# Paths
ucf101_path = 'data/UCF101'
classind_path = 'data/UCFTrainTestList/classInd.txt'

# Get all folder names from UCF101 directory
actual_folders = set([name for name in os.listdir(ucf101_path) if os.path.isdir(os.path.join(ucf101_path, name))])

# Read classInd.txt and extract folder/class names
with open(classind_path, 'r') as f:
    listed_folders = set()
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            _, class_name = parts
        else:
            class_name = parts[0]  # In case no number is present
        listed_folders.add(class_name)

# Find folders that are in actual_folders but not in listed_folders
unlisted_folders = actual_folders - listed_folders

# Output
print("Folders present in UCF101 but NOT listed in classInd.txt:")
for folder in sorted(unlisted_folders):
    print("-", folder)
