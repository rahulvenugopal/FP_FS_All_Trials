# Compiling images to a single large image
# author @ Rahul Venugopal 19.04.2026

# Loading libraries
library(magick)

# load all images .png
images_list <- list.files(pattern = "*.png")

# Load location of images
imgs <- image_read(images_list)

# Find grid size automagically
no_of_cols <- 3

no_of_rows <- 1

# Set the layout
imgs <- c(imgs[1:3])

# tile it
stitched <- image_montage(imgs,
                          tile = '3x1', # layout of grid cols*rows
                          # size of individual thumbnail and spacing
                          # Horizontal gap
                          # Vertical gap
                          geometry = "x1500+5+5") # size of individual thumbnail and spacing

# saving the images
image_write(stitched,path = "stiched_alpha_theta_exp.png",
            quality = 100, # no compression
            density = 900, #dpi
            format = "png")