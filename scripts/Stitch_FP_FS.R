# Compiling images to a single large image
# author @ Rahul Venugopal 06.05.2022

# Loading libraries
library(magick)

# load all images .png
images_list <- list.files(pattern = "*.png")

# Load location of images
imgs <- image_read(images_list)

# Find grid size automagically
no_of_cols <- ceiling(length(imgs)/6)

no_of_rows <- floor(length(imgs)/6)

# Set the layout
imgs <- c(imgs[1:6],
          imgs[7:12],
          imgs[13:18],
          imgs[19:24],
          imgs[25:30],
          imgs[31:36])

# tile it
stitched <- image_montage(imgs,
                          tile = '6x6', # layout of grid cols*rows
                          # size of individual thumbnail and spacing
                          # Horizontal gap
                          # Vertical gap
                          geometry = "x600+5+5") # size of individual thumbnail and spacing

# saving the images
image_write(stitched,path = "stiched_FS_Alpha.png",
            quality = 100, # no compression
            density = 900, #dpi
            format = "png")