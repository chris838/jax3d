import json
import os
import numpy
import cv2

import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing.pool import ThreadPool

import jax.numpy as jnp
import numpy as np


def blender(scene_dir, scene_type, white_bkgd, samples_dir):

    data = {'train': load_blender(scene_dir, 'train', white_bkgd),
            'test': load_blender(scene_dir, 'test', white_bkgd)}

    splits = ['train', 'test']
    for s in splits:
        print(s)
        for k in data[s]:
            print(f'  {k}: {data[s][k].shape}')

    images, poses, hwf = data['train']['images'], data['train']['c2w'], data['train']['hwf']
    write_floatpoint_image(samples_dir+"/training_image_sample.png", images[0])

    for i in range(3):
        plt.figure()
        plt.scatter(poses[:, i, 3], poses[:, (i+1) % 3, 3])
        plt.axis('equal')
        plt.savefig(samples_dir+"/training_camera"+str(i)+".png")

    return data

def llff(scene_dir, scene_type, white_bkgd, samples_dir):

    data = {'train': load_LLFF(scene_dir, scene_type, 'train'),
            'test': load_LLFF(scene_dir, scene_type, 'test')}

    splits = ['train', 'test']
    for s in splits:
        print(s)
        for k in data[s]:
            print(f'  {k}: {data[s][k].shape}')

    images, poses, hwf = data['train']['images'], data['train']['c2w'], data['train']['hwf']
    write_floatpoint_image(
        samples_dir + "/training_image_sample.png", images[0])

    for i in range(3):
        plt.figure()
        plt.scatter(poses[:, i, 3], poses[:, (i+1) % 3, 3])
        plt.axis('equal')
        plt.savefig(samples_dir + "/training_camera" + str(i)+".png")

    bg_color = jnp.mean(images)

    return data, bg_color

def nerfstudio(scene_dir, samples_dir, num_test_frames=3, train_on_test_frames=False):

    data = {'train': load_nerfstudio(scene_dir, 'train', num_test_frames, train_on_test_frames),
            'test': load_nerfstudio(scene_dir, 'test', num_test_frames, train_on_test_frames)}

    splits = ['train', 'test']
    for s in splits:
        print(s)
        for k in data[s]:
            print(f'  {k}: {data[s][k].shape}')

    images, poses, hwf = data['train']['images'], data['train']['c2w'], data['train']['hwf']
    write_floatpoint_image(
        samples_dir + "/training_image_sample.png", images[0])

    for i in range(3):
        plt.figure()
        plt.scatter(poses[:, i, 3], poses[:, (i+1) % 3, 3])
        plt.axis('equal')
        plt.savefig(samples_dir + "/training_camera" + str(i)+".png")

    bg_color = jnp.mean(images)

    return data, bg_color





def load_blender(data_dir, split, white_bkgd):
    with open(
            os.path.join(data_dir, "transforms_{}.json".format(split)), "r") as fp:
        meta = json.load(fp)

    cams = []
    paths = []
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        cams.append(jnp.array(frame["transform_matrix"], dtype=jnp.float32))

        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        paths.append(fname)

    with ThreadPool() as pool:
        images = pool.map(image_read_fn, paths)
        pool.close()
        pool.join()

    images = jnp.stack(images, axis=0)
    if white_bkgd:
        images = (images[..., :3] * images[..., -1:] +
                  (1. - images[..., -1:]))
    else:
        images = images[..., :3] * images[..., -1:]

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = .5 * w / jnp.tan(.5 * camera_angle_x)

    hwf = jnp.array([h, w, focal], dtype=jnp.float32)
    poses = jnp.stack(cams, axis=0)
    return {'images': images, 'c2w': poses, 'hwf': hwf}

def load_LLFF(data_dir, scene_type, split, factor=4, llffhold=8):
    # Load images.
    imgdir_suffix = ""
    if factor > 0:
        imgdir_suffix = "_{}".format(factor)
    imgdir = os.path.join(data_dir, "images" + imgdir_suffix)
    if not os.path.exists(imgdir):
        raise ValueError("Image folder {} doesn't exist.".format(imgdir))
    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]

    with ThreadPool() as pool:
        images = pool.map(image_read_fn, imgfiles)
        pool.close()
        pool.join()
    images = np.stack(images, axis=-1)

    # Load poses and bds.
    with open(os.path.join(data_dir, "poses_bounds.npy"),
              "rb") as fp:
        poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    if poses.shape[-1] != images.shape[-1]:
        raise RuntimeError("Mismatch between imgs {} and poses {}".format(
            images.shape[-1], poses.shape[-1]))

    # Update poses according to downsampling.
    poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    # Correct rotation matrix ordering and move variable dim to axis 0.
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    if scene_type == "real360":
        # Rotate/scale poses to align ground with xy plane and fit to unit cube.
        poses, _ = _transform_poses_pca(poses)
    else:
        # Rescale according to a default bd factor.
        scale = 1. / (bds.min() * .75)
        poses[:, :3, 3] *= scale
        bds *= scale
        # Recenter poses
        poses = _recenter_poses(poses)

    # Select the split.
    i_test = np.arange(images.shape[0])[::llffhold]
    i_train = np.array(
        [i for i in np.arange(int(images.shape[0])) if i not in i_test])
    if split == "train":
        indices = i_train
    else:
        indices = i_test
    images = images[indices]
    poses = poses[indices]

    camtoworlds = poses[:, :3, :4]
    focal = poses[0, -1, -1]
    h, w = images.shape[1:3]

    hwf = np.array([h, w, focal], dtype=np.float32)

    return {'images': jnp.array(images), 'c2w': jnp.array(camtoworlds), 'hwf': jnp.array(hwf), 'poses': poses}

def load_nerfstudio(data_dir, split, num_test_frames, train_on_test_frames):
    with open(
            os.path.join(data_dir, "transforms.json"), "r") as fp:
        meta = json.load(fp)

    fx = []
    fy = []
    cams = []
    paths = []

    if "fl_x" in meta:
      fx.append(float(meta["fl_x"]))
    if "fl_y" in meta:
      fy.append(float(meta["fl_y"]))

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]

        if "fl_x" in frame:
          fx.append(float(frame["fl_x"]))
        if "fl_y" in frame:
          fy.append(float(frame["fl_y"]))

        cams.append(np.array(frame["transform_matrix"], dtype=np.float32))

        fname = os.path.join(data_dir, frame["file_path"])
        paths.append(fname)

    with ThreadPool() as pool:
        images = pool.map(image_read_fn, paths)
        pool.close()
        pool.join()

    images = np.stack(images, axis=0)

    h, w = images.shape[1:3]

    # Get focal length from metadata, or first pose
    focal = fx[0]

    hwf = np.array([h, w, focal], dtype=np.float32)
    poses = np.stack(cams, axis=0)

    # Recenter poses, so the average pose sits at the origin
    poses = _recenter_poses(poses)

    # Select the train/eval split.
    i_all = np.arange(int(images.shape[0]))
    i_test = np.random.choice(i_all, num_test_frames)
    if train_on_test_frames:
      i_train = i_all
    else:
      i_train = np.array([i for i in i_all if i not in i_test])

    if split == "train":
        indices = i_train
    else:
        indices = i_test
    images = images[indices]
    poses = poses[indices]

    # Looks like the transform matrix is in homogeneous coords, but this doesn't quite match the llff data?
    # Following what the LLFF loader does and cutting the extra dim, although I don't think it matters
    c2w = poses[:, 0:3, :]

    # Convert to jax before returning
    return {'images': jnp.array(images), 'c2w': jnp.array(c2w), 'hwf': jnp.array(hwf), 'poses': poses}





def write_floatpoint_image(name, img):
    img = numpy.clip(numpy.array(img)*255, 0, 255).astype(numpy.uint8)
    cv2.imwrite(name, img[:, :, ::-1])

def image_read_fn(fname):
    with open(fname, "rb") as imgin:
        image = jnp.array(Image.open(imgin), dtype=jnp.float32) / 255.
    return image

def _viewmatrix(z, up, pos):
    """Construct lookat view matrix."""
    vec2 = _normalize(z)
    vec1_avg = up
    vec0 = _normalize(np.cross(vec1_avg, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def _normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def _poses_avg(poses):
    """Average poses according to the original NeRF code."""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = _normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([_viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def _recenter_poses(poses):
    """Recenter poses according to the original NeRF code."""
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = _poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

def _transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes."""
    poses_ = poses.copy()
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    bottom = np.broadcast_to([0, 0, 0, 1.], poses[..., :1, :4].shape)
    pad_poses = np.concatenate([poses[..., :3, :4], bottom], axis=-2)
    poses_recentered = transform @ pad_poses
    poses_recentered = poses_recentered[..., :3, :4]
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    poses_[:, :3, :4] = poses_recentered[:, :3, :4]
    poses_recentered = poses_
    return poses_recentered, transform
