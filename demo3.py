import numpy as np
import functions3
from functions3 import PhongMaterial
from functions3 import PointLight
import cv2
from datetime import datetime
import time

data = np.load('h3.npy', allow_pickle = True)


verts = data[()]['verts']
vertex_colors = data[()]['vertex_colors']
face_indices = data[()]['face_indices']
cam_eye = data[()]['cam_eye'].reshape(3,1)
cam_up = data[()]['cam_up'].reshape(3,1)
cam_lookat = data[()]['cam_lookat'].reshape(3,1)
ka = data[()]['ka']
kd = data[()]['kd']
ks = data[()]['ks']
n = data[()]['n']
light_positions = np.array(data[()]['light_positions'])
light_intensities = np.array(data[()]['light_intensities'])
Ia = data[()]['Ia']
M = data[()]['M']
N = data[()]['N']
W = data[()]['W']
H = data[()]['H']
bg_color = data[()]['bg_color']
focal = 70

#Create 1 PhongMaterial and  a list of three PointLight sources
material = PhongMaterial(0, 0, 0, n)

lightSources = [PointLight(light_positions[0], light_intensities[0]),
                PointLight(light_positions[1], light_intensities[1]),
                PointLight(light_positions[2], light_intensities[2])]


def printImg(shade_t: str, light_t: str, ka: float, kd: float, ks: float, material: PhongMaterial, lightSources: PointLight):

    """
            Render image using functions3.RenderObject() and show result

            Arguments:
                shade_t: The shading technnique to be used
                light_t: The light type
                ka: Ambient coefficient
                kd: Diffuse coeff
                ks: Specular coeff
                material: The PhongMaterial type object describing the objects texture
                lightSources:  list containing PointLight type objects/ light sources 
                
            Output:
                None , shows the resulting image
        """

    material.ambientCoeff = ka
    material.diffuseCoeff = kd
    material.specularCoeff = ks

    startTime = time.time()

    print('Rendering ' + shade_t + ' ' + light_t + '...')

    renderedImg = functions3.RenderObject(shade_t, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices,
                                          material,
                                          lightSources,
                                          Ia
                                         )
    cv2.imwrite(
    shade_t + ' ' + light_t + '.jpg',
    cv2.cvtColor( (renderedImg*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )

    print('Render  runtime:     %s seconds    ' % (time.time() - startTime))

    cv2.imshow(shade_t + ' ' + light_t, renderedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Step 1 -------------------- AMBIENT GOURAUD-----------------------

printImg('gouraud', 'ambient', ka, 0, 0, material, lightSources)

#Step 2 -------------------DIFFUSE GOURAUD-------------------------

printImg('gouraud', 'diffuse', 0, kd, 0, material, lightSources)

#Step 3 ------------------SPECULAR GOURAUD-------------------------

printImg('gouraud', 'specular', 0, 0, ks, material, lightSources)

#Step 4 -----------------GOURAUD ALL-------------------------------

printImg('gouraud', 'all', ka, kd, ks, material, lightSources)


#Step 1 -------------------AMBIENT PHONG--------------------------

printImg('phong', 'ambient', ka, 0, 0, material, lightSources)

#Step 2 ------------------DIFFUSE PHONG---------------------------

printImg('phong', 'diffuse', 0, kd, 0, material, lightSources)

#Step 3-------------------SPECULAR PHONG--------------------------

printImg('phong', 'specular', 0, 0, ks, material, lightSources)

#Step 4 ------------------PHONG ALL--------------------------------

printImg('phong', 'all', ka, kd, ks, material, lightSources)

###EXTRA------------------Flat ambient-----------------------------

printImg('flat', 'ambient', ka, 0, 0, material, lightSources)

#------------------------Flat diffuse------------------------------

printImg('flat', 'diffuse', ka, kd, 0, material, lightSources)

#------------------------Flat Specular----------------------------

printImg('flat', 'specular', ka, kd, ks, material, lightSources)

#Step 4 -----------------Flat ALL-------------------------------

printImg('flat', 'all', ka, kd, ks, material, lightSources)