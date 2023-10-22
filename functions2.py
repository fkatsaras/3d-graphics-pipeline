from typing import Tuple
import numpy as np
import functions

def rotmat(theta = 0., axis: np.array = np.nan) -> np.array:

    """
            Create a rotation matrix, given the angle and axis of rotation.

            Arguments:
                theta: The angle of rotation 
                axis: The axis of rotation
                
            Output:
                The required rotation matrix (3x3 array)
    """
    
    if axis is np.nan or theta == 0: #If rotation is not required, identity will be returned (No rotation)
        return np.eye(3)
    
    # Get unit vector of axis
    u = axis / np.linalg.norm(axis)

    sin0 = np.sin(theta)
    cos0 = np.cos(theta)

    #Calculate cross product matrix of axis
    uCross = np.array([[0, -u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1], u[0], 0]])
    
    #Return the complete rotation matrix using formula
    return cos0 * np.eye(3) + sin0 * uCross + (1 - cos0) * np.outer(u,u)

def RotateTranslate(points3d: np.array, theta = 0., axis: np.array = np.nan, translation = np.array([0, 0, 0])) -> np.array:

    """
            Rotation & translation of a single point / an array of 3D points given theta & axis

            Arguments:
                points3d: The array of points to be rotated & translated (3xN array) 
                theta: The angle of rotation
                axis: The axis of rotation (3x1 vector)
                translation: The vector describing the point of translation (3x1 vector)
                
            Output:
                points3d: The rotated & translated points
    """

    # Create rotation matrix
    rotationMatrix = rotmat(theta, axis)

    # Rotate
    points3d = np.matmul(rotationMatrix, points3d)

    # Translate
    points3d = np.add(points3d, translation.reshape(3,1)) # Reshape translation as column vector for matrix multiplication

    return points3d

def ChangeCoordinateSystem(points3d: np.array, rotMat: np.array = np.nan, newOrigin = np.array([0, 0, 0])) -> np.array:

    """
            Changing coordinate system; Rotation & translation of an array of 3D points given a rotation matrix

            Arguments:
                points3d: The array of points to be rotated & translated (3xN array) 
                rotMat: The matrix describing the rotation
                newOrigin: The origin(x,y,z) of the new coord system
                
            Output:
                points3d: The rotated & translated points
    """
    
    if rotMat is not np.nan:

        #Rotate
        points3d = np.matmul(rotMat, points3d)

    #Translate
    points3d = np.add(points3d, -newOrigin) #Reshape origin as column vector

    return points3d

def PinHole(f: float, cameraOrigin: np.array, x: np.array, y: np.array, z: np.array, points3d: np.array) -> Tuple[np.array, np.array]:

    """
            Find the 2D projection / depth of 3D points, given WCS ,camera origin / unit vectors

            Arguments:
                f: focal length
                cameraOrigin: the camera position (xyz) point 
                x: Camera x vector
                y: Camera y vector
                z: Camera z vector
                points3d: 3D point to find the 2D projection of (3xN array)
                
            Output:
                points2d: The 2D projection of the 3D points (2xN array)
                depth: The depth of every 2D projected point (1xN array)
    """

    #Points are given in WCS -> transform to CCS by changing coordinate system
    cameraPoints = ChangeCoordinateSystem(points3d, np.array([x, y, z]), -cameraOrigin)

    # Find Depth of points/ Extracting the third row (z) of cameraPoints(3xN)
    depth = (cameraPoints[2, :, None]).T #Adding new columnn for depth to be compatible for matrix multiplication

    # Find projection of points: (x', y') = f * (x, y) / z
    points2d = f * ( cameraPoints[[0, 1], :] / depth )


    return points2d, depth

def CameraLookingAt(f: float, cameraOrigin: np.array, lookat: np.array, up: np.array, points3d: np.array) -> Tuple[np.array, np.array]:

    """
            Find 2D projection/depth of 3D points, given WCS, camera's origin, lookat, up vectors

            Arguments:
                f: focal length
                cameraOrigin: the camera position (xyz) point 
                lookat: Camera lookat vector
                up: Camera up vector 
                points3d: 3D point to find the 2D projection of (3xN array)
                
            Output:
                Using PinHole function:
               
                points2d: The 2D projection of the 3D points (2xN array)
                depth: The depth of every 2D projected point (1xN array)
    """
    #Reshape cameraOrigin, lookat, up as 1D arrays for easier operations

    # Find unit vectors for the camera's system
    z = (np.squeeze(lookat) - np.squeeze(cameraOrigin)) / np.linalg.norm(np.squeeze(lookat) - np.squeeze(cameraOrigin))

    t = np.squeeze(up) - np.dot(np.squeeze(up), z) * z
    y = t / np.linalg.norm(t)

    x = np.cross(y, z)

    #Pass unit vectors to project_cam to find 2D projection/ depth
    return PinHole(f, cameraOrigin, x, y, z, points3d)

def rasterize(points2d: np.array, imgColumns: int, imgRows: int, camHeight: float, camWidth: float) -> np.array:

    """
            Build image from projected 2D points

            Arguments:
                points2d: The coordinates of the 2D points to be rasterized into an image (2xN array)
                imgColumns: Image height
                imgRows:  Image width
                camHeight: Camera height
                camWidth: Camera width
                
            Output:
               
                points2d: The rasterized points 
    """
    
    # Scale points
    points2d[0, :] = points2d[0, :] * imgRows / camWidth
    points2d[1, :] = points2d[1, :] * imgColumns / camHeight
    
    cameraOrigin = np.array([[-imgRows/2], [-imgColumns/2]])

    # Transform to image coordinates
    points2d = ChangeCoordinateSystem(points2d, np.nan, cameraOrigin).round()
    # Rearrange image y points
    points2d[1, :] = imgColumns - points2d[1, :]
    
    return points2d

def RenderObject(points3d: np.array, faces: np.array, vcolors: np.array,
                  imgHeight: int, imgWidth: int, camHeight: float, camWidth: float, f: float,
                  camOrigin: np.array, lookat: np.array, up: np.array, shade: str,
                  vnormals: np.array,
                  mat,#: PhongMaterial, #ADDED Illumination inputs
                  lights,#: PointLight,
                  Ia: np.array,
                  background: np.array
                  ) -> np.array:
        """
            Shade & illuminate the projection of a 3D object; 2D Projection + Illumination(using functions3.py) + Shading(using functions.py)

            Arguments:
               points3d: The (x,y,z) coords of all the 3D points of the object (3xN array)
               faces: Indexes of the points3d array. Each row represents a triangle (3xL array)
               vcolors: The (R,G,B) components of every 3D point (3xN array)
               imageHeight: Image height
               imageWidth: Image width
               camHeight: Camera height
               camWidth: Camera width
               f: The focal length of the camera
               camOrigin: The camera center position
               lookat: The camera lookat point 
               up: The camera up vector
               shade: The shading method that will be implemented
               vnormals: The xyz coords of the objects 3D surface normals (3xN array)
               mat: A PhongMaterial type object describing the texture of the object
               lights:  list containing N PointLight objects describing the different light sources acting on the object
               Ia: The ambient light intensity vector (3x1)
               background_color: The background color of the image 
                
            Output:
                Using functions.render(), the shaded and illuminated image (MxN array)
        """

        [points2d, depth] = CameraLookingAt(f, camOrigin, lookat, up, points3d)

        points2d = rasterize(points2d, imgHeight, imgWidth, camHeight, camWidth)

        #Transpose points2d and depth because render() works with Nx3 arrays
        #Return colored image (512x512)
        return functions.render(points2d.T, faces, vcolors, depth.T, shade,
                                points3d.T,
                                vnormals,
                                camOrigin,
                                mat,
                                lights,
                                Ia,
                                background,
                                imgHeight,
                                imgWidth
                                )