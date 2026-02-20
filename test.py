import kagglehub

# Download latest version
path = kagglehub.dataset_download("aslanahmedov/number-plate-detection")

print("Path to dataset files:", path)