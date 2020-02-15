""" Tests for utilities module """


import numpy as np
import matplotlib.pyplot as mplt

import toolbox.computation as comp 
import toolbox.plotting.shapes as shapes


def test_rotate2d(trials, width, angles):
	"""
	Generate a radom set of 2D points, rotate them by the 
	passed angles, and display plot of resulting rotations. 
	"""
	x = (np.random.random(trials) - 0.5)*width
	y = (np.random.random(trials) - 0.5)*width
	points_shape = list(x.shape)
	points_shape.append(2)
	points = np.full(points_shape, np.nan)
	points[..., 0], points[..., 1] = x, y
	rotated = comp.rotate2d(points, np.deg2rad(angles), form='cart')
	points2d = np.array([x.flatten(), y.flatten()]).T
	xrot_flat = rotated[..., 0].flatten()
	yrot_flat = rotated[..., 1].flatten()
	rotated2d = np.array([xrot_flat, yrot_flat]).T
	multiple_angles = not np.isscalar(angles)
	if multiple_angles:
		angles2d = np.asarray(angles).flatten()
	fig, ax = mplt.subplots()
	for index, (pt, pt_rot) in enumerate(zip(points2d, rotated2d)):
		ax.plot(*pt, marker='o', color='b', linestyle='', alpha=0.8)
		ax.plot([0, pt[0]], [0, pt[1]], marker='',
			    color='k', alpha=0.8, linestyle='-')
		ax.plot(*pt_rot, marker='o', color='r', linestyle='', alpha=0.8)
		if multiple_angles:
			ax.annotate(angles2d[index], xy=pt_rot,
						xytext=pt_rot + np.absolute(pt_rot)*0.1)
		ax.plot([0, pt_rot[0]], [0, pt_rot[1]], marker='',
			    alpha=0.8, color='k', linestyle='-')
		pt_radius = np.sqrt(np.sum(pt**2))
		ax.add_patch(shapes.circle([0, 0], pt_radius, alpha=0.4,
					               facecolor='none', edgecolor='k'))
	ax.plot([], marker='o', color='b', linestyle='',
		    alpha=0.8, label='original')
	ax.plot([], marker='o', color='r', linestyle='',
		    alpha=0.8, label='rotated')
	ax.legend()
	ax.axhline(0, linestyle='--', alpha=0.4) 
	ax.axvline(0, linestyle='--', alpha=0.4)
	ax.set_aspect('equal')
	title = "utilities.rotate2d test: shape {}".format(trials)
	if not multiple_angles:
		title += ", angle {}".format(angles)
	else: 
		title += ", multiple angles"
	ax.set_title(title)
	mplt.show()