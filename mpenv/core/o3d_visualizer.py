import open3d as o3d
import numpy as np


class Open3DVisualizer:
    def __init__(self):
        self.viz = None
        self.pcd = o3d.geometry.PointCloud()
        self.lines = o3d.geometry.LineSet()

    def __del__(self):
        if self.viz is not None:
            self.viz.destroy_window()

    def _create_viz(self):
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        self.viz.add_geometry(self.pcd)

    def show_pcd(self, points, normals, colors=None, blocking=False):
        new_instance = self.viz is None
        if new_instance and not blocking:
            self._create_viz()

        if new_instance:
            self.pcd = o3d.geometry.PointCloud()

        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.normals = o3d.utility.Vector3dVector(normals)
        if colors is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

        if not blocking:
            if new_instance:
                self.viz.add_geometry(self.pcd)
            else:
                self.viz.update_geometry(self.pcd)
                self.viz.poll_events()
                self.viz.update_renderer()
        else:
            o3d.visualization.draw_geometries([self.pcd])

    def show_lines(self, p0, p1, blocking=False):
        new_instance = self.viz is None
        if new_instance and not blocking:
            self._create_viz()

        n = p0.shape[0]
        if new_instance:
            pcd0 = o3d.geometry.PointCloud()
            pcd1 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(p0)
            pcd1.points = o3d.utility.Vector3dVector(p1)
            correspondances = np.stack((np.arange(n), np.arange(n)), 1)
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(np.vstack((p0, p1)))
            self.lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                pcd0, pcd1, correspondances
            )
        else:
            self.pcd.points = o3d.utility.Vector3dVector(np.vstack((p0, p1)))
            self.lines.points = self.pcd.points
            correspondances = np.stack((np.arange(0, n), np.arange(n, 2 * n)), 1)
            self.lines.lines = o3d.utility.Vector2iVector(correspondances)

        if not blocking:
            if new_instance:
                self.viz.add_geometry(self.lines)
                self.viz.add_geometry(self.pcd)
            else:
                self.viz.update_geometry(self.lines)
                self.viz.update_geometry(self.pcd)
                self.viz.poll_events()
                self.viz.update_renderer()
        else:
            o3d.visualization.draw_geometries([self.lines])

    def show_voxels(self, voxels, blocking=True):
        if not blocking:
            raise NotImplementedError
        o3d.visualization.draw_geometries([voxels])
