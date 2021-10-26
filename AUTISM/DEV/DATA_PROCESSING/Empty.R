library(freesurferformats)

file_path <- "./DATA/50054/MP-RAGE/2000-01-01_00_00_00.0/I328862/ABIDE_50054_MRI_MP-RAGE_br_raw_20120830185721042_S164852_I328862.nii"

brain <- read.fs.volume.nii(file_path)

print(dim(brain)) # 176, 256, 256, 1

brain <- brain[1:176, 1:256, 1:256, ] # drop last dim

for (i in 1:176) print(sum(x[i, ,])) # get empty slices

# brain <- brain[5:173, 1:256, 1:256] # TODO drop empty slices
# print(dim(brain)) # 171 256 256
