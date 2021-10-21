# https://rdrr.io/rforge/oro.nifti/f/inst/doc/nifti.pdf
temp_brain <- readNIfTI("DATA/50054/MP-RAGE/2000-01-01_00_00_00.0/I328862/ABIDE_50054_MRI_MP-RAGE_br_raw_20120830185721042_S164852_I328862.nii")
oro.nifti::image(temp_brain)