library(freesurferformats)
library(animation)

some_brain <- read.fs.volume.nii("DATA/50054/MP-RAGE/2000-01-01_00_00_00.0/I328862/ABIDE_50054_MRI_MP-RAGE_br_raw_20120830185721042_S164852_I328862.nii")

print(dim(some_brain))

saveGIF({
  for (i in 1:176) image(some_brain[i , , ,])
})