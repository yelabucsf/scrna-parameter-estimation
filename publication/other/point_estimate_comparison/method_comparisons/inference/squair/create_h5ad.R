library(Seurat)
# library(SeuratData)
library(SeuratDisk)

data_path <- '/data_volume/memento/method_comparison/squair/sc_rnaseq/h5Seurat/'

files = c(
    'Hagai2018_mouse-lps',
    'Hagai2018_mouse-pic',
    'Hagai2018_pig-lps',
    'Hagai2018_rabbit-lps',
    'Hagai2018_rat-lps',
    'Hagai2018_rat-pic'
)

for (fname in files) {
    
    full_path <- paste(
        data_path,
        fname,
        '.h5Seurat',
        sep='')
    
    Convert(full_path, dest = "h5ad")
    
    }