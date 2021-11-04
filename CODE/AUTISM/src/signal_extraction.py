import sys

from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets

def main():
    atlas = datasets.fetch_atlas_aal()

    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)
    time_series = masker.fit_transform(sys.argv[0])

    print(time_series)

if __name__ == "__main__":
    main()
