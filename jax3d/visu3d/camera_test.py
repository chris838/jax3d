# Copyright 2022 The jax3d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for camera."""

from etils import enp
from jax3d import visu3d as v3d
import numpy as np

# Activate the fixture
set_tnp = enp.testing.set_tnp


H, W = 64, 128


def _make_cam(
    *,
    xnp: enp.NpModule,
) -> v3d.Camera:
  """Create a camera at (0, 4, 0) looking at the center."""
  ray = v3d.Ray.from_look_at(
      pos=[0, 4, 0],
      end=[0, 0, 0],
  )
  spec = v3d.PinholeCamera.from_focal(resolution=(H, W), focal_in_px=34.)
  cam = v3d.Camera.from_ray(spec=spec.as_xnp(xnp), ray=ray.as_xnp(xnp))
  return cam


@enp.testing.parametrize_xnp()
def test_camera_properties(xnp: enp.NpModule):
  cam = _make_cam(xnp=xnp)

  # Properties
  assert cam.resolution == (H, W)
  assert cam.h == H
  assert cam.w == W


@enp.testing.parametrize_xnp()
def test_camera_normalized_rays(xnp: enp.NpModule):
  cam = _make_cam(xnp=xnp)

  rays = cam.rays()
  assert isinstance(rays, v3d.Ray)
  assert rays.xnp is xnp
  assert rays.shape == (H, W)
  np.testing.assert_allclose(rays.pos, cam.cam_to_world.broadcast_to((H, W)).t)
  # Ray is normalized
  np.testing.assert_allclose(
      np.linalg.norm(rays.dir, axis=-1),
      np.ones((H, W)),
      atol=1e-6,
  )


@enp.testing.parametrize_xnp()
def test_camera_non_normalized_rays(xnp: enp.NpModule):
  cam = _make_cam(xnp=xnp)  # Camera on the `y` axis
  rays = cam.rays(normalize=False)
  assert isinstance(rays, v3d.Ray)
  assert rays.xnp is xnp
  assert rays.shape == (H, W)
  np.testing.assert_allclose(rays.pos, cam.cam_to_world.broadcast_to((H, W)).t)
  # Ray destinations are aligned with the y=3 plane
  np.testing.assert_allclose(rays.end[..., 1], np.full((H, W), 3.))


@enp.testing.parametrize_xnp()
def test_camera_render(xnp: enp.NpModule):
  cam = _make_cam(xnp=xnp)  # Camera on the `y` axis
  points = v3d.Point(
      p=[
          [0, 0, 0],
      ],
      rgb=[
          [255., 255., 255.],
      ],
  )
  points = points.as_xnp(xnp)
  img = cam.render(points)
  assert img.shape == (H, W, 3)
  assert img.dtype == np.uint8
  np.testing.assert_allclose(img[0, 0], [0, 0, 0])
  np.testing.assert_allclose(img[H // 2, W // 2], [255, 255, 255])
