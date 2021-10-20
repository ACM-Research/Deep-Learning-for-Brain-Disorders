files_list <- list.files(path = ".", pattern = ".nii", all.files = TRUE, recursive = TRUE)

temp_vector = c()

# there exists nii data that is complex???
for (file in files_list) {
  temp <- read.fs.volume.nii(file)
  print(temp)
}

print(temp_vector)

data_frame <- as.data.frame(tensors)
write.csv(data_frame, file="autism_tensors.csv")