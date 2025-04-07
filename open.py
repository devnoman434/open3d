import open3d as o3d
import numpy as np
import copy

# Load the mesh
mesh_path = "scaned.obj"
mesh = o3d.io.read_triangle_mesh(mesh_path)

# Check mesh validity
print("Has vertices:", mesh.has_vertices())
print("Has triangles:", mesh.has_triangles())
print("Bounding box min:", mesh.get_min_bound())
print("Bounding box max:", mesh.get_max_bound())

# Abort if empty
if not mesh.has_vertices():
    print("Mesh appears to be empty.")
    exit()

# Compute normals
mesh.compute_vertex_normals()

# Center and normalize the mesh scale
center = mesh.get_center()
mesh.translate(-center)

bbox = mesh.get_axis_aligned_bounding_box()

# Calculate extent manually
extent = bbox.max_bound - bbox.min_bound

# Scale the mesh
scale = 1.0 / max(extent)
mesh.scale(scale, center=(0, 0, 0))

# Apply color if no texture
if not mesh.has_vertex_colors() and not mesh.has_textures():
    mesh.paint_uniform_color([0.8, 0.3, 0.2])  # reddish

# Coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

# ➤ Crop mesh (remove +Z half)
crop_box = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=bbox.get_min_bound(),
    max_bound=[bbox.get_max_bound()[0], bbox.get_max_bound()[1], 0]
)
mesh_cropped = mesh.crop(crop_box)

# ➤ Rotate mesh (90° around X)
rotated_mesh = copy.deepcopy(mesh)
R = rotated_mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
rotated_mesh.rotate(R, center=(0, 0, 0))

# ➤ Translate meshes so they don't overlap
original = copy.deepcopy(mesh).translate([-1.5, 0, 0])
cropped = copy.deepcopy(mesh_cropped).translate([1.5, 0, 0])
rotated = copy.deepcopy(rotated_mesh).translate([0, -1.5, 0])

# ➤ Create high-quality point cloud from the mesh
point_cloud = mesh.sample_points_poisson_disk(
    number_of_points=5000,
    init_factor=5
)
point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
point_cloud.normalize_normals()
point_cloud.paint_uniform_color([0.2, 0.6, 0.9])  # Sky-blue

# ➤ Visualize all geometries
o3d.visualization.draw_geometries(
    [frame, original, cropped, rotated, point_cloud],
    window_name="Refined Mesh Viewer",
    width=1280,
    height=720,
    mesh_show_back_face=True,
    point_show_normal=False
)

