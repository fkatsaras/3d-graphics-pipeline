import numpy as np
import functions2


class PhongMaterial:

    def __init__(self,ka: float = 0, kd: float = 0, ks: float = 0, n: int = 0):

        self.ambientCoeff = ka
        self.diffuseCoeff = kd
        self.specularCoeff = ks
        self.phongCoeff = n

class PointLight:

    def __init__(self, pos: np.array, intensity: np.array):

        self.position = pos
        self.intensity = intensity

def light(point: np.array, normal: np.array, vcolor: np.array,
           cam_pos: np.array, mat: PhongMaterial, lights: PointLight, Ia: np.array):
    
    """
            Calculate reflection on point

            Arguments:
                point: Point to calculate reflection on (3x1 vector)
                normal: Surface perpendicular vector on point (3x1)
                vcolor: Color of point (1x3)
                cam_pos: Camera center coordinates (3x1)
                mat: PhongMaterial type object 
                lights: List containing N PointLight type objects
                Ia: Ambient light intensity
                
            Output:
                color of point including light reflection 
        """

    #Calculate L, V

    light_vectors = np.zeros(( len(lights), 3)) #Initialize array of size  len(lights) x 3

    # Calculate unit vectors of each light source with point as the start
    for index, light in enumerate(lights):

        light_vectors[index] = light.position.T - np.squeeze(point.T) #Transpose point for correct matrix shape    

    light_vectors = light_vectors / np.linalg.norm(light_vectors)


    # Calculate unit vector of camera center with point as the start
    camera_vector = cam_pos - point
    camera_vector = camera_vector / np.linalg.norm(camera_vector)

    # Calculate (cosine of) angle of each light source with N (dot product of each light vector with N)
    # Diffuse - Specular 

    #diffuseAngles is calculated as a 1xN vector
    diffuseLightAngles = np.einsum('ij,jk->ik', light_vectors, normal) #Find dot product of each row of light vectors w/ normal vector 

    #specularAngles is 1x3 also. # (2 * (L * N) * N - L) * V
    specularLightAngles = np.einsum('ij,jk->ik', 
                                    
                                    2 * np.einsum('ij,jk->ik', light_vectors, normal) * normal.T - light_vectors, 

                                      camera_vector)
    
    #Calculate AMBIENT Light intensity:
    I = mat.ambientCoeff * Ia

    #Find and add diffuse/ specular light created by each light source 
    for index, light in enumerate(lights):

        #Add DIFFUSE Light intensity:
        #Each source has its intensity and light angle  
        I += light.intensity * mat.diffuseCoeff * diffuseLightAngles[index]

        #Add SPECULAR Light intensity:
        I += light.intensity * mat.specularCoeff * (np.absolute(specularLightAngles[index])**mat.phongCoeff) * np.sign(specularLightAngles[index])


    return vcolor + np.sum(I, axis=0)

def CalculateNormals(verts: np.array, faces: np.array)-> np.array:

    """
            Calculate surface normals of a 3d 'object'

            Arguments:
                verts: The (x,y,z) coords of all the 3D points of the object (3xN array) 
                faces: Indexes of the verts3d array. Each row represents a triangle (3xL array)
                
            Output:
                vertexNormals: The (x,y,z) coords of the objects' surface normal at each vertex (3xN array)
    """
    
    v1 = verts[:, faces[0, :]] #Group every triangle's vertices' coords
    v2 = verts[:, faces[1, :]]
    v3 = verts[:, faces[2, :]]

    triangleNormals = np.cross( v2 - v1, v3 - v1, axis=0) #Find every triangle's normal (At vertex v0) and find its unit

    triangleNormals /= np.linalg.norm(triangleNormals, axis= 0)
    triangleNormals = triangleNormals.T

    vertexNormals = np.zeros_like(verts).T  #Transpose for easier calculation

    for index, face in enumerate(faces.T):  #For every vertex, add together all the triangle normals calculated
        vertexNormals[face] += triangleNormals[index]

    vertexNormals = vertexNormals.T

    return vertexNormals / np.linalg.norm(vertexNormals, axis=0) #Return 3xN array of unit normals

def RenderObject(
        shader: str,
        focal: float, 
        eye: np.array, 
        lookat: np.array, 
        up: np.array, 
        background_color: np.array, 
        imageM: int, 
        imageN: int, 
        camera_height: float,
        camera_width: float,
        verts3d: np.array,
        vcolors: np.array,
        face_indices: np.array,
        mat: PhongMaterial,
        lights: PointLight,
        Ia: np.array
    ) -> np.array:

    """
            Render a 3D object; Illumination + 2D Projection (using functions2.py) + Shading (using functions.py)

            Arguments:
               shader: The shading method that will be implemented
               focal: The focal length of the camera
               eye: The camera center position
               lookat: The camera lookat point 
               up: The camera up vector
               background_color: The background color of the image 
               imageM: Image height
               imageN: Image width
               camera_height: Camera height
               camera_width: Camera width
               verts3d: The (x,y,z) coords of all the 3D points of the object (3xN array) 
               vcolors: The (R,G,B) components of every 3D point (3xN array)
               face_indices: Indexes of the verts3d array. Each row represents a triangle (3xL array)
               mat: A PhongMaterial type object
               lights:  list containing N PointLight objects
               Ia: The ambient light intensity vector (3x1)
                
            Output:
                renderedImg: The fully rendered image with all the effects added (MxN array)
        """

    #Calculate object normals 3xN
    vnormals = CalculateNormals(verts3d, face_indices)

    #Find projection of object using Functions2 render ###### Transposing vcolors - vnormals / functions2 nd functions work with Nx3 arrays
    renderedImg = functions2.RenderObject(verts3d, face_indices.T, vcolors.T, imageM, imageN, camera_height, camera_width, focal, eye, lookat, up, shader,
                                     vnormals.T, 
                                     mat,
                                     lights,
                                     Ia,
                                     background_color
                                     )
    
    renderedImg = np.clip(renderedImg, 0., 1.)
    
    return renderedImg
    





