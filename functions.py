import numpy as np
import functions3

def interpolate_Vectors(p1: float, p2:float, V1: np.array, V2: np.array, xy: float, dim: int) -> np.array:
    """
            Interpolate vector value at x or y between two points (values) given y or x

            Arguments:
                p1: first point
                p2: second point
                V1: (vector) value of p1 
                V2: (vector) value of p2
                xy: Given coordinate for interpolation
                dim: Required coordinate
                
            Output:
                Interpolated vector value at requested point 
    """

    if dim==1:
        # x is the known variable xy, y is the unknown
        l = abs(p2[0] - xy)/abs(p2[0] - p1[0])
    elif dim==2:
        # y is the known variable xy, x is the unknown
        l = abs(p2[1] - xy)/abs(p2[1] - p1[1])

    return np.add( np.multiply(l, V1), np.multiply(1-l, V2) )

def shade_flat(mat,#: PhongMaterial,
               lights,#: PointLight,
               canvas: np.array,
               vertices: np.array,
               vcolors: np.array,
               bcoords: np.array = np.nan,
               vnormals:np.array = np.nan,
               cam_pos: np.array = np.nan,
               Ia: np.array = np.nan,
             ) -> np.array:
   
   """
            Shade (& illuminate) a triangle using the flat shading technique

            Arguments:
               mat: A PhongMaterial type object that describes the texture of the trianlge
               lights: A list containing N PointLight type objects, describing the light sources near the triangle
               canvas: The image containing the triangle before shadin & illumination (M x N array)
               vertices: The coords of the three 2D points of the triangle (3x2 array) 
               vcolors: The (RGB) components of each of the triangles' three vertices
               bcoords: The (xyz) coords of the 3D barycentric point of the triangle (before projection) (3x1) 
               vnormals: The (xyz) coords of the 3D normal vectors at each vertex of the triangle (3x1)
               cam_pos: The (xyz) coords of the 3D camera position point 
               Ia: The ambient light intensity vector (3x1)
                
            Output:
                canvas: The image containing the triangle after shading & illumination (M x N array)
    """

   #Sorting the vertices, v0 goes to bottom of triangle, v1 at mid, v2 at top
   sorted_idx = vertices[:, 1].argsort()
   vertices = vertices[sorted_idx]
   vcolors = vcolors[sorted_idx]

   #ADDED Calculating the  new vertice colors with added light effects
   #For flats Diffuse/Specular light is added at first on the vertex colors -> Then interpolated 

   for vertex in range(3): 
      vcolors[vertex] = functions3.light(bcoords, vnormals[vertex].reshape(3,1), vcolors[vertex], cam_pos, mat, lights, Ia)

   
   triangleColor = np.average(vcolors, axis=0) #flat coloring: Avrg of vertices' color

   x1, y1 = vertices[0]
   x2, y2 = vertices[1]
   x3, y3 = vertices[2]

   xArr = np.array((x1,x2,x3))
   yArr = np.array((y1,y2,y3))

   #Find slope of each side, account for vertical sides to avoid division by zero
   slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf   
   slope2 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else np.inf
   slope3 = (y3 - y1) / (x3 - x1) if (x3 - x1) != 0 else np.inf

   slopes = np.array((slope1,slope2,slope3)) # Slopes of e0-e1-e2

   ymin = int(yArr[0])  #Finding scanline y-range 
   ymax = int(yArr[2])

   #CASE 0 : If v1-v2-v3 are collinear (cross product = 0): Then returns empty canvas

   if ((x3 - x2)*(y1 - y2) - (y3 - y2)*(x1 - x2)) < 0: # CASE 1: v0 is to the right of line v1-v2
         
      X = np.empty(2) # First scanline at y = ymin, Scanning the whole y-range of the triangle

      cross_count = 0

      ##REMOVED unnecessary for loop

      for y in range(ymin, int(y2)):   #Scan the side with the smallest y-range at the bottom

         X[0] = int((y - y1) * (1 / slopes[0]) + x1)    #Active marginal points on each side
         X[1] = int((y - y1) * (1 / slopes[2]) + x1)

         for x in range(int(X[0]), int(X[1])):

            if x == X[0] or x == X[1]:

               cross_count =+ 1
               
            if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: 

               if cross_count % 2 != 0:
                     
                  canvas[y,x] = triangleColor  #Fill space inbetween

      for y in range(int(y2), ymax):  #Scan the remaining y-range, do the same
               
         X[0] = int((y - y2) * (1 / slopes[1]) + x2)
         X[1] = int((y - y1) * (1 / slopes[2]) + x1)

         for x in range(int(X[0]), int(X[1])):

            if x == X[0] or x == X[1]:

               cross_count =+ 1
                  
            if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]:  

               if cross_count % 2 != 0:
                     
                  canvas[y,x] = triangleColor  #Fill space inbetween
                        
   elif ((x3 - x2)*(y1 - y2) - (y3 - y2)*(x1 - x2)) > 0:  #CASE 2: v0 is to the left of line v1v2
         
      X = np.empty(2)

      cross_count = 0

      for y in range(ymin, int(y2)):

         X[0] = int((y - y1) * (1 / slopes[2]) + x1)
         X[1] = int((y - y1) * (1 / slopes[0]) + x1)

         for x in range(int(X[0]), int(X[1])):

            if x == X[0] or x == X[1]:

               cross_count =+ 1
               
            if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: 

               if cross_count % 2 != 0:
                     
                  canvas[y,x] = triangleColor  #Fill space inbetween

      for y in range(int(y2), ymax):
               
         X[0] = int((y - y1) * (1 / slopes[2]) + x1)
         X[1] = int((y - y2) * (1 / slopes[1]) + x2)

         for x in range(int(X[0]), int(X[1])):

            if x == X[0] or x == X[1]:

               cross_count =+ 1
                  
            if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]:

               if cross_count % 2 != 0:
                     
                  canvas[y,x] = triangleColor  #Fill space inbetween                   

   return canvas

def shade_gouraud(mat,#: PhongMaterial,
                  lights,#: PointLight,
                  canvas: np.array,
                  vertices: np.array,
                  vcolors: np.array,
                  bcoords: np.array = np.nan,
                  vnormals:np.array = np.nan,
                  cam_pos: np.array = np.nan,
                  Ia: np.array = np.nan,
             ) -> np.array:
   
   """
            Shade (& illuminate) a triangle using the gouraud shading technique

            Arguments:
               mat: A PhongMaterial type object that describes the texture of the trianlge
               lights: A list containing N PointLight type objects, describing the light sources near the triangle
               canvas: The image containing the triangle before shadin & illumination (M x N array)
               vertices: The coords of the three 2D points of the triangle (2x3 array) 
               vcolors: The (RGB) components of each of the triangles' three vertices
               bcoords: The (xyz) coords of the 3D barycentric point of the triangle (before projection) (3x1) 
               vnormals: The (xyz) coords of the 3D normal vectors at each vertex of the triangle (3x1)
               cam_pos: The (xyz) coords of the 3D camera position point 
               Ia: The ambient light intensity vector (3x1)
                
            Output:
                canvas: The image containing the triangle after shading & illumination (M x N array)
    """

   sorted_idx = vertices[:, 1].argsort()
   vertices = vertices[sorted_idx]
   vcolors = vcolors[sorted_idx]
   vnormals = vnormals[sorted_idx] #ADDED :Sorting Normals 3 normals for each vertex of the triagnle

   #ADDED Calculating the  new vertice colors with added light effects
   #For gourauds Diffuse light is added at first on the vertex colors -> Then interpolated 

   for vertex in range(3):
      vcolors[vertex] = functions3.light(bcoords, vnormals[vertex].reshape(3,1), vcolors[vertex], cam_pos, mat, lights, Ia)

   triangleColor = 0 #Initialize color

   x1, y1 = vertices[0]
   x2, y2 = vertices[1]
   x3, y3 = vertices[2]

   xArr = np.array((x1,x2,x3))
   yArr = np.array((y1,y2,y3))

   slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
   slope2 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else np.inf
   slope3 = (y3 - y1) / (x3 - x1) if (x3 - x1) != 0 else np.inf

   slopes = np.array((slope1,slope2,slope3)) # Slopes of e0-e1-e2

   ymin = int(yArr[0])
   ymax = int(yArr[2])

   #CASE 0 : If v1-v2-v3 are collinear (cross product = 0): Then returns empty canvas

   if ((x3 - x2)*(y1 - y2) - (y3 - y2)*(x1 - x2)) < 0: # CASE 1: v0 is to the right of line v1-v2

      X = np.empty(2) #Initializing active marginal points 
      colorA = np.empty(3)
      colorB = np.empty(3)

      cross_count = 0
        
      # First scanline at y = ymin, Scanning the whole y-range of the triangle
      for y in range(ymin, int(y2)):   #Scan the side with the smallest y-range at the bottom


         X[0] = int((y - y1) * (1 / slopes[0]) + x1)    #Active marginal points
         X[1] = int((y - y1) * (1 / slopes[2]) + x1)

         colorA = interpolate_Vectors([x1, y1], [x2, y2], vcolors[0], vcolors[1], y, 2) # Find color of marginal points (Interpolating v0 -v1 -> A)
         colorB = interpolate_Vectors([x1, y1], [x3, y3], vcolors[0], vcolors[2], y, 2) #Interpolating v0 - v2 -> B

         for x in range(int(X[0]), int(X[1])):

            if x == X[0] or x == X[1]:

               cross_count =+ 1

            if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: # ADDED CLIPPING CONDITION

               if cross_count % 2 != 0:

                  triangleColor = interpolate_Vectors([X[0], y], [X[1], y], colorA, colorB, x, 1) #Find color of points inside X[0] - X[1] Interpolating A - B

                  canvas[y,x] = triangleColor  #Fill space inbetween

               
      for y in range(int(y2), ymax):  #Scan the remaining y-range, do the same
               
      
         X[0] = int((y - y2) * (1 / slopes[1]) + x2)
         X[1] = int((y - y1) * (1 / slopes[2]) + x1)

         colorA = interpolate_Vectors([x2, y2], [x3, y3], vcolors[1], vcolors[2], y, 2) # Find color of marginal points (Interpolating v1 -v2 -> A)
         colorB = interpolate_Vectors([x1, y1], [x3, y3], vcolors[0], vcolors[2], y, 2) #Interpolating v0 - v2 -> B

         for x in range(int(X[0]), int(X[1])):

            if x == X[0] or x == X[1]:

               cross_count =+ 1
                  
            if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: 

               if cross_count % 2 != 0:

                  triangleColor = interpolate_Vectors([X[0], y], [X[1], y], colorA, colorB, x, 1) #Find color of points inside X[0] - X[1] Interpolating A - B

                  canvas[y,x] = triangleColor  #Fill space inbetween

                        
   elif ((x3 - x2)*(y1 - y2) - (y3 - y2)*(x1 - x2)) > 0:  #CASE 2: v0 is to the left of line v1v2

      X = np.empty(2) #Initializing active marginal points 
      colorA = np.empty(3)
      colorB = np.empty(3)

      cross_count = 0
      
      for y in range(ymin, int(y2)):


         X[0] = int((y - y1) * (1 / slopes[2]) + x1)
         X[1] = int((y - y1) * (1 / slopes[0]) + x1)

         colorA = interpolate_Vectors([x1, y1], [x3, y3], vcolors[0], vcolors[2], y, 2) # Find color of marginal points (Interpolating v0 -v2 -> A)
         colorB = interpolate_Vectors([x1, y1], [x2, y2], vcolors[0], vcolors[1], y, 2) #Interpolating v0 - v1 -> B

         for x in range(int(X[0]), int(X[1])):
           
           if x == X[0] or x == X[1]:

               cross_count =+ 1
               
           if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: 

               if cross_count % 2 != 0:

                  triangleColor = interpolate_Vectors([X[0], y], [X[1], y], colorA, colorB, x, 1) #Find color of points inside X[0] - X[1] Interpolating A - B

                  canvas[y,x] = triangleColor  #Fill space inbetween


      for y in range(int(y2), ymax):

               
         X[0] = int((y - y1) * (1 / slopes[2]) + x1)
         X[1] = int((y - y2) * (1 / slopes[1]) + x2)

         colorA = interpolate_Vectors([x1, y1], [x3, y3], vcolors[0], vcolors[2], y, 2) # Find color of marginal points (Interpolating v0 -v2 -> A)
         colorB = interpolate_Vectors([x2, y2], [x3, y3], vcolors[1], vcolors[2], y, 2) #Interpolating v1 - v2 -> B

         for x in range(int(X[0]), int(X[1])):
           
           if x == X[0] or x == X[1]:

               cross_count =+ 1
                  
           if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: 

               if cross_count % 2 != 0:

                  triangleColor = interpolate_Vectors([X[0], y], [X[1], y], colorA, colorB, x, 1) #Find color of points inside X[0] - X[1] Interpolating A - B

                  canvas[y,x] = triangleColor  #Fill space inbetween

   return canvas

def shade_phong(mat,#: PhongMaterial,
               lights,#: PointLight,
               canvas: np.array,
               vertices: np.array,
               vcolors: np.array,
               bcoords: np.array = np.nan,
               vnormals:np.array = np.nan,
               cam_pos: np.array = np.nan,
               Ia: np.array = np.nan,
             ) -> np.array:
   
   """
            Shade (& illuminate) a triangle using the phong shading technique

            Arguments:
               mat: A PhongMaterial type object that describes the texture of the trianlge
               lights: A list containing N PointLight type objects, describing the light sources near the triangle
               canvas: The image containing the triangle before shadin & illumination (M x N array)
               vertices: The coords of the three 2D points of the triangle (2x3 array) 
               vcolors: The (RGB) components of each of the triangles' three vertices
               bcoords: The (xyz) coords of the 3D barycentric point of the triangle (before projection) (3x1) 
               vnormals: The (xyz) coords of the 3D normal vectors at each vertex of the triangle (3x1)
               cam_pos: The (xyz) coords of the 3D camera position point 
               Ia: The ambient light intensity vector (3x1)
                
            Output:
                canvas: The image containing the triangle after shading & illumination (M x N array)
    """

   sorted_idx = vertices[:, 1].argsort()
   vertices = vertices[sorted_idx]
   vcolors = vcolors[sorted_idx]
   vnormals = vnormals[sorted_idx] #ADDED :Sorting Normals 3 normals for each vertex of the triagnle

   #ADDED Calculating the  new vertice colors with added light effects
   #For gourauds Diffuse light is added at first on the vertex colors -> Then interpolated 

   for vertex in range(3):
      vcolors[vertex] = functions3.light(bcoords, vnormals[vertex].reshape(3,1), vcolors[vertex], cam_pos, mat, lights, Ia)


   triangleColor = 0 #Initialize color

   x1, y1 = vertices[0]
   x2, y2 = vertices[1]
   x3, y3 = vertices[2]

   xArr = np.array((x1,x2,x3))
   yArr = np.array((y1,y2,y3))

   slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
   slope2 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else np.inf
   slope3 = (y3 - y1) / (x3 - x1) if (x3 - x1) != 0 else np.inf

   slopes = np.array((slope1,slope2,slope3)) # Slopes of e0-e1-e2

   ymin = int(yArr[0])
   ymax = int(yArr[2])

   #CASE 0 : If v1-v2-v3 are collinear (cross product = 0): Then returns empty canvas

   if ((x3 - x2)*(y1 - y2) - (y3 - y2)*(x1 - x2)) < 0: # CASE 1: v0 is to the right of line v1-v2

      X = np.empty(2) #Initializing active marginal points 
      colorA = np.empty(3)
      colorB = np.empty(3)

      cross_count = 0
        
      # First scanline at y = ymin, Scanning the whole y-range of the triangle
      for y in range(ymin, int(y2)):   #Scan the side with the smallest y-range at the bottom


         X[0] = int((y - y1) * (1 / slopes[0]) + x1)    #Active marginal points
         X[1] = int((y - y1) * (1 / slopes[2]) + x1)

         colorA = interpolate_Vectors([x1, y1], [x2, y2], vcolors[0], vcolors[1], y, 2) # Find color of marginal points (Interpolating v0 -v1 -> A)
         colorB = interpolate_Vectors([x1, y1], [x3, y3], vcolors[0], vcolors[2], y, 2) #Interpolating v0 - v2 -> B

         #ADDED Interpolating NORMALS for Phong
         normalA = interpolate_Vectors([x1, y1], [x2, y2], vnormals[0], vnormals[1], y, 2) #Find normal of marginal points (Interpolating normalv0 - v1 ->A)
         normalB = interpolate_Vectors([x1, y1], [x3, y3], vnormals[0], vnormals[2], y, 2) #Interpolating nv0 - nv2 -> B

         for x in range(int(X[0]), int(X[1])):

            if x == X[0] or x == X[1]:

               cross_count =+ 1

            if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: # ADDED CLIPPING CONDITION

               if cross_count % 2 != 0:

                  triangleColor = interpolate_Vectors([X[0], y], [X[1], y], colorA, colorB, x, 1) #Find color of points inside X[0] - X[1] Interpolating A - B
                  pixelNormal = interpolate_Vectors([X[0], y], [X[1], y], normalA, normalB, x, 1) #ADDED interpolating normals inside X[0] - [1]

                  triangleColor = functions3.light(bcoords, pixelNormal.reshape(3,1), triangleColor, cam_pos, mat, lights, Ia)

                  canvas[y,x] = triangleColor  #Fill space inbetween

               
      for y in range(int(y2), ymax):  #Scan the remaining y-range, do the same
               
      
         X[0] = int((y - y2) * (1 / slopes[1]) + x2)
         X[1] = int((y - y1) * (1 / slopes[2]) + x1)

         colorA = interpolate_Vectors([x2, y2], [x3, y3], vcolors[1], vcolors[2], y, 2) # Find color of marginal points (Interpolating v1 -v2 -> A)
         colorB = interpolate_Vectors([x1, y1], [x3, y3], vcolors[0], vcolors[2], y, 2) #Interpolating v0 - v2 -> B

         #ADDED Interpolating NORMALS for Phong
         normalA = interpolate_Vectors([x2, y2], [x3, y3], vnormals[1], vnormals[2], y, 2) #Find normal of marginal points (Interpolating normalv0 - v1 ->A)
         normalB = interpolate_Vectors([x1, y1], [x3, y3], vnormals[0], vnormals[2], y, 2) #Interpolating nv0 - nv2 -> B

         for x in range(int(X[0]), int(X[1])):

            if x == X[0] or x == X[1]:

               cross_count =+ 1
                  
            if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: 

               if cross_count % 2 != 0:

                  triangleColor = interpolate_Vectors([X[0], y], [X[1], y], colorA, colorB, x, 1) #Find color of points inside X[0] - X[1] Interpolating A - B
                  pixelNormal = interpolate_Vectors([X[0], y], [X[1], y], normalA, normalB, x, 1) #ADDED interpolating normals inside X[0] - [1]

                  #triangleColor = functions3.DiffuseLight(bcoords, pixelNormal.reshape(3,1), triangleColor, kd, light_positions, light_intensities)
                  triangleColor = functions3.light(bcoords, pixelNormal.reshape(3,1), triangleColor, cam_pos, mat, lights, Ia)

                  canvas[y,x] = triangleColor  #Fill space inbetween

                        
   elif ((x3 - x2)*(y1 - y2) - (y3 - y2)*(x1 - x2)) > 0:  #CASE 2: v0 is to the left of line v1v2

      X = np.empty(2) #Initializing active marginal points 
      colorA = np.empty(3)
      colorB = np.empty(3)

      cross_count = 0
      
      for y in range(ymin, int(y2)):


         X[0] = int((y - y1) * (1 / slopes[2]) + x1)
         X[1] = int((y - y1) * (1 / slopes[0]) + x1)

         colorA = interpolate_Vectors([x1, y1], [x3, y3], vcolors[0], vcolors[2], y, 2) # Find color of marginal points (Interpolating v0 -v2 -> A)
         colorB = interpolate_Vectors([x1, y1], [x2, y2], vcolors[0], vcolors[1], y, 2) #Interpolating v0 - v1 -> B

         #ADDED Interpolating NORMALS for Phong
         normalA = interpolate_Vectors([x1, y1], [x3, y3], vnormals[0], vnormals[2], y, 2) #Find normal of marginal points (Interpolating normalv0 - v1 ->A)
         normalB = interpolate_Vectors([x1, y1], [x2, y2], vnormals[0], vnormals[1], y, 2) #Interpolating nv0 - nv2 -> B

         for x in range(int(X[0]), int(X[1])):
           
           if x == X[0] or x == X[1]:

               cross_count =+ 1
               
           if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: 

               if cross_count % 2 != 0:

                  triangleColor = interpolate_Vectors([X[0], y], [X[1], y], colorA, colorB, x, 1) #Find color of points inside X[0] - X[1] Interpolating A - B
                  pixelNormal = interpolate_Vectors([X[0], y], [X[1], y], normalA, normalB, x, 1) #ADDED interpolating normals inside X[0] - [1]

                  triangleColor = functions3.light(bcoords, pixelNormal.reshape(3,1), triangleColor, cam_pos, mat, lights, Ia)

                  canvas[y,x] = triangleColor  #Fill space inbetween


      for y in range(int(y2), ymax):

               
         X[0] = int((y - y1) * (1 / slopes[2]) + x1)
         X[1] = int((y - y2) * (1 / slopes[1]) + x2)

         colorA = interpolate_Vectors([x1, y1], [x3, y3], vcolors[0], vcolors[2], y, 2) # Find color of marginal points (Interpolating v0 -v2 -> A)
         colorB = interpolate_Vectors([x2, y2], [x3, y3], vcolors[1], vcolors[2], y, 2) #Interpolating v1 - v2 -> B

         #ADDED Interpolating NORMALS for Phong
         normalA = interpolate_Vectors([x1, y1], [x3, y3], vnormals[0], vnormals[2], y, 2) #Find normal of marginal points (Interpolating normalv0 - v1 ->A)
         normalB = interpolate_Vectors([x2, y2], [x3, y3], vnormals[1], vnormals[2], y, 2) #Interpolating nv0 - nv2 -> B

         for x in range(int(X[0]), int(X[1])):
           
           if x == X[0] or x == X[1]:

               cross_count =+ 1
                  
           if 0 <= y < np.shape(canvas)[0] and 0 <= x < np.shape(canvas)[1]: 

               if cross_count % 2 != 0:

                  triangleColor = interpolate_Vectors([X[0], y], [X[1], y], colorA, colorB, x, 1) #Find color of points inside X[0] - X[1] Interpolating A - B
                  pixelNormal = interpolate_Vectors([X[0], y], [X[1], y], normalA, normalB, x, 1) #ADDED interpolating normals inside X[0] - [1]

                  triangleColor = functions3.light(bcoords, pixelNormal.reshape(3,1), triangleColor, cam_pos, mat, lights, Ia)

                  canvas[y,x] = triangleColor  #Fill space inbetween                     

   return canvas

def sortTriangles(faces: np.array, depth: np.array) -> np.array:

    # This function sorts the triangles in descending order based on their vertices' depth
    #  triangle depth = avrg depth of its vertices 
    # (Greatest depth = furthest away from pov, Shading begins form furthest triangle)
    #
    # Input - Triangles (faces array Lx3) - L vertices, each row represents 1 triangle
    # Vertice depths(depth array Lx1) - L vertices 
    # Output -  Triangles (faces array Lx3), but sorted according to depth

   """
            Sort the triangles in descending order based on their vertices' depth

            Arguments:
                faces: Indexes of the verts3d array. Each row represents a triangle (Lx3 array)
                depth: Depth of each face (Calculated as avrg of triangles' vertices depth)
                
            Output:
                sortedFaces: The same faces array but sorted based on descending depth
   """
    
   triangleDepths = np.empty(np.shape(faces)[0])

   for i, face in enumerate(faces):
      triangleDepths[i] = ( depth[face[0]] + depth[face[1]] + depth[face[2]] )/3

   sortedFaces = faces[np.argsort(-triangleDepths)]

   return sortedFaces

def render(verts2d: np.array, faces: np.array, vcolors: np.array, depth: np.array, shade_t: str,
           verts3d: np.array,
           vnormals: np.array,
           cam_pos: np.array,
           mat,#: PhongMaterial, #ADDED Illumination inputs
           lights,#: PointLight,
           Ia: np.array,
           background:np.array,
           M: int,
           N: int
           ) -> np.array:
   
   """
            Shade & illuminate an image; Illumination(using functions3.py) + Shading

            Arguments:
               verts2d: The (x,y) coords of all the 2D points of the object (Nx2 array)
               faces: Indexes of the verts2d array. Each row represents a triangle (Lx3 array)
               vcolors: The (R,G,B) components of every 2D point (Nx3 array)
               depth: The depth of every 2D point (Nx1 array)
               shade_t: The shading method that will be implemented
               vnormals: The xyz coords of the objects 3D surface normals (3xN array)
               mat: A PhongMaterial type object describing the texture of the object
               lights:  list containing N PointLight objects describing the different light sources acting on the object
               Ia: The ambient light intensity vector (3x1)
               background: The background color of the image 
                
            Output:
                shadedImg: The shaded & illuminated image  (MxN array)
        """

   img = np.full( (M, N, 3) , background) #initialize background

   if shade_t == 'flat': # FLAT SHADING

      for face in sortTriangles(faces, depth):

         ## ADDED barycenter calculation (FOR EACH TRIANGLE)
         bcoords = np.mean(verts3d[face], axis=0).reshape(3,1)
       
         shadedImg = shade_flat(
            mat,
            lights,
            img, 
            np.array([ verts2d[face[0]], verts2d[face[1]], verts2d[face[2]] ]),
            np.array([ vcolors[face[0]], vcolors[face[1]], vcolors[face[2]] ]),
            bcoords,
            np.array([vnormals[face[0]], vnormals[face[1]], vnormals[face[2]]]),
            cam_pos,
            Ia
          )
         
      return shadedImg
   
   elif shade_t == 'gouraud':  #GOURAUD SHADING
      for face in sortTriangles(faces, depth):

        bcoords = np.mean(verts3d[face], axis=0).reshape(3,1)

        shadedImg = shade_gouraud(
            mat,
            lights,
            img, 
            np.array([ verts2d[face[0]], verts2d[face[1]], verts2d[face[2]] ]),
            np.array([ vcolors[face[0]], vcolors[face[1]], vcolors[face[2]] ]),
            bcoords,
            np.array([vnormals[face[0]], vnormals[face[1]], vnormals[face[2]]]),
            cam_pos,
            Ia
          )
      return shadedImg
   
   elif shade_t == 'phong':  #ADDED PHONG SHADING
      for face in sortTriangles(faces, depth):

        bcoords = np.mean(verts3d[face], axis=0).reshape(3,1) #ADDED Calculating the barycentric coords of each triangle

        shadedImg = shade_phong(
            mat,
            lights,
            img, 
            np.array([ verts2d[face[0]], verts2d[face[1]], verts2d[face[2]] ]),
            np.array([ vcolors[face[0]], vcolors[face[1]], vcolors[face[2]] ]),
            bcoords,
            np.array([vnormals[face[0]], vnormals[face[1]], vnormals[face[2]]]),
            cam_pos,
            Ia
          )
      return shadedImg