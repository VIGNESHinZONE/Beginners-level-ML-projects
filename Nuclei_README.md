A collection of datasets converted into COCO segmentation format.

## Preprocessing:
    Resized few images
    Tiled some images with lot of annotations to fit in memory
    Extracted masks when only outlines were available
        This is done by finding contours

## Folder hierarchy

```python

DATASETS = {
    'nuclei_stage1_train': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_1_train',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage1_train.json'
    },
    'nuclei_stage_1_local_train_split': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_1_train',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage_1_local_train_split.json'
    },
    'nuclei_stage_1_local_val_split': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_1_train',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage_1_local_val_split.json'
    },
    'nuclei_stage_1_test': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_1_test',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage_1_test.json'
    },
    'nuclei_stage_2_test': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_2_test',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage_2_test.json'
    },
    'cluster_nuclei': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/cluster_nuclei',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/cluster_nuclei.json'
    },
    'BBBC007': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/BBBC007',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/BBBC007.json'
    },
    'BBBC006': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/BBBC006',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/BBBC006.json'
    },
    'BBBC018': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/BBBC018',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/BBBC018.json'
    },
    'BBBC020': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/BBBC020',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/BBBC020.json'
    },
    'nucleisegmentationbenchmark': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/nucleisegmentationbenchmark',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/nucleisegmentationbenchmark.json'
    },
    '2009_ISBI_2DNuclei': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/2009_ISBI_2DNuclei',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/2009_ISBI_2DNuclei.json'
    },
    'nuclei_partial_annotations': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/nuclei_partial_annotations',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/nuclei_partial_annotations.json'
    },
    'TNBC_NucleiSegmentation': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/TNBC_NucleiSegmentation',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/TNBC_NucleiSegmentation.json'
    },
}
```

## Example usage:

```python

import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

ROOT_DIR = Path('/media/gangadhar/DataSSD1TB/ROOT_DATA_DIR/')
DATASET_WORKING_DIR = ROOT_DIR / 'Nuclei'

annotations_file = DATASET_WORKING_DIR / 'annotations/stage1_train.json'

COCO = json.load(open(annotations_file.as_posix()))

image_metadata = COCO['images'][0]
print image_metadata

# {u'file_name': u'4ca5081854df7bbcaa4934fcf34318f82733a0f8c05b942c2265eea75419d62f.jpg',
#  u'height': 256,
#  u'id': 0,
#  u'nuclei_class': u'purple_purple_320_256_sparce',
#  u'width': 320}


def get_masks(im_metadata):
    image_annotations = []
    for annotation in COCO['annotations']:
        if annotation['image_id'] == im_metadata['id']:
            image_annotations.append(annotation)

    segments = [annotation['segmentation'] for annotation in image_annotations]
    masks = mask_util.decode(segments)
    return masks


masks = get_masks(image_metadata)

print masks.shape
# (256, 320, 37)


def show(i):
    i = np.asarray(i, np.float)
    m,M = i.min(), i.max()
    I = np.asarray((i - m) / (M - m + 0.000001) * 255, np.uint8)
    Image.fromarray(I).show()


show(np.sum(masks, -1))
# this should show an image with all masks

```

## References

- [2018 Data Science Bowl: Find the nuclei in divergent images to advance medical discovery](https://www.kaggle.com/c/data-science-bowl-2018).
  Competition, Kaggle, Apr. 2018.
  Download: https://www.kaggle.com/c/data-science-bowl-2018/data

- [2018 Data Science Bowl: Kaggle Data Science Bowl 2018 dataset fixes](https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes).
  Konstantin Lopuhin, Apr. 2018.
  Download: https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes

- [TNBC_NucleiSegmentation: A dataset for nuclei segmentation based on Breast Cancer patients](https://zenodo.org/record/1175282).
  Naylor Peter Jack; Walter Thomas; Laé Marick; Reyal Fabien. 2018.
  Download: https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip

- [A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology](https://www.ncbi.nlm.nih.gov/pubmed/28287963).
  Kumar N, Verma R, Sharma S, Bhargava S, Vahadane A, Sethi A. 2017.
  Download: https://nucleisegmentationbenchmark.weebly.com/dataset.html

- [Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases](http://www.jpathinformatics.org/article.asp?issn=2153-3539;year=2016;volume=7;issue=1;spage=29;epage=29;aulast=Janowczyk).
  Andrew Janowczyk, Anant Madabhushi. 2016.
  Download: http://andrewjanowczyk.com/wp-static/nuclei.tgz

- [Nuclei Dataset: Include 52 images of 200x200 pixels](https://imagej.nih.gov/ij/plugins/ihc-toolbox/index.html).
  Jie Shu, Guoping Qiu, Mohammad Ilyas.
  Immunohistochemistry (IHC) Image Analysis Toolbox, Jan. 2015.
  Download: https://www.dropbox.com/s/9knzkp9g9xt6ipb/cluster%20nuclei.zip?dl=0

- [BBBC006v1: image set available from the Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC007/).
  Vebjorn Ljosa, Katherine L Sokolnicki & Anne E Carpenter. 2012.
  Download: https://data.broadinstitute.org/bbbc/BBBC007/

- [BBBC007v1: image set available from the Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC006/).
  Vebjorn Ljosa, Katherine L Sokolnicki & Anne E Carpenter. 2012.
  Download: https://data.broadinstitute.org/bbbc/BBBC006/

- [BBBC018v1: image set available from the Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC018/).
  Vebjorn Ljosa, Katherine L Sokolnicki & Anne E Carpenter. 2012.
  Download: https://data.broadinstitute.org/bbbc/BBBC018/

- [BBBC020v1: image set available from the Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC020/).
  Vebjorn Ljosa, Katherine L Sokolnicki & Anne E Carpenter. 2012.
  Download: https://data.broadinstitute.org/bbbc/BBBC020/

- [Nuclei Segmentation In Microscope Cell Images: A Hand-Segmented Dataset And Comparison Of Algorithms](http://murphylab.web.cmu.edu/data/2009_ISBI_Nuclei.html).
  L. P. Coelho, A. Shariff, and R. F. Murphy
  Proceedings of the 2009 IEEE International Symposium on Biomedical Imaging (ISBI 2009), pp. 518-521, 2009.
  Download: http://murphylab.web.cmu.edu/data/2009_ISBI_2DNuclei_code_data.tgz

