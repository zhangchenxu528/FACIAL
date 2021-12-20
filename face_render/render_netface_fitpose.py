import os, sys
import numpy as np
from skimage import io
import cv2
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel
from load_data import BFM 
from scipy.io import loadmat,savemat
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Render_setting')
parser.add_argument('--real_params_path', type=str, default='../video_preprocess/train1_posenew.npz')
parser.add_argument('--outpath', type=str, default = '../video_preprocess/train_A/')

opt = parser.parse_args()

def Compute_norm(face_shape,facemodel):

	face_id = facemodel.tri # vertex index for each triangle face, with shape [F,3], F is number of faces
	point_id = facemodel.point_buf # adjacent face index for each vertex, with shape [N,8], N is number of vertex
	shape = face_shape
	face_id = (face_id - 1).astype(np.int32)
	point_id = (point_id - 1).astype(np.int32)
	v1 = shape[:,face_id[:,0],:]
	v2 = shape[:,face_id[:,1],:]
	v3 = shape[:,face_id[:,2],:]
	e1 = v1 - v2
	e2 = v2 - v3
	face_norm = np.cross(e1,e2) # compute normal for each face
	face_norm = np.concatenate([face_norm,np.zeros([1,1,3])], axis = 1) # concat face_normal with a zero vector at the end
	v_norm = np.sum(face_norm[:,point_id,:], axis = 2) # compute vertex normal using one-ring neighborhood
	v_norm = v_norm/np.expand_dims(np.linalg.norm(v_norm,axis = 2),2) # normalize normal vectors

	return v_norm

def Illumination_layer(face_texture,norm,gamma):

	num_vertex = np.shape(face_texture)[1]

	init_lit = np.array([0.8,0,0,0,0,0,0,0,0])
	gamma = np.reshape(gamma,[-1,3,9])
	gamma = gamma + np.reshape(init_lit,[1,1,9])

	# parameter of 9 SH function
	a0 = np.pi 
	a1 = 2*np.pi/np.sqrt(3.0)
	a2 = 2*np.pi/np.sqrt(8.0)
	c0 = 1/np.sqrt(4*np.pi)
	c1 = np.sqrt(3.0)/np.sqrt(4*np.pi)
	c2 = 3*np.sqrt(5.0)/np.sqrt(12*np.pi)

	Y0 = np.tile(np.reshape(a0*c0,[1,1,1]),[1,num_vertex,1]) 
	Y1 = np.reshape(-a1*c1*norm[:,:,1],[1,num_vertex,1]) 
	Y2 = np.reshape(a1*c1*norm[:,:,2],[1,num_vertex,1])
	Y3 = np.reshape(-a1*c1*norm[:,:,0],[1,num_vertex,1])
	Y4 = np.reshape(a2*c2*norm[:,:,0]*norm[:,:,1],[1,num_vertex,1])
	Y5 = np.reshape(-a2*c2*norm[:,:,1]*norm[:,:,2],[1,num_vertex,1])
	Y6 = np.reshape(a2*c2*0.5/np.sqrt(3.0)*(3*np.square(norm[:,:,2])-1),[1,num_vertex,1])
	Y7 = np.reshape(-a2*c2*norm[:,:,0]*norm[:,:,2],[1,num_vertex,1])
	Y8 = np.reshape(a2*c2*0.5*(np.square(norm[:,:,0])-np.square(norm[:,:,1])),[1,num_vertex,1])

	Y = np.concatenate([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],axis=2)

	# Y shape:[batch,N,9].

	lit_r = np.squeeze(np.matmul(Y,np.expand_dims(gamma[:,0,:],2)),2) #[batch,N,9] * [batch,9,1] = [batch,N]
	lit_g = np.squeeze(np.matmul(Y,np.expand_dims(gamma[:,1,:],2)),2)
	lit_b = np.squeeze(np.matmul(Y,np.expand_dims(gamma[:,2,:],2)),2)

	# shape:[batch,N,3]
	face_color = np.stack([lit_r*face_texture[:,:,0],lit_g*face_texture[:,:,1],lit_b*face_texture[:,:,2]],axis = 2)
	lighting = np.stack([lit_r,lit_g,lit_b],axis = 2)*128

	return face_color,lighting

def render(facemodel, chi):
    fitted_R = mesh.transform.angle2matrix(chi[0:3])

    fitted_s = chi[3] 
    fitted_t = chi[4:7].copy()
    fitted_t[2] = 1.0
    fitted_ep = np.expand_dims(chi[7:71],1)
    fitted_sp = np.expand_dims(facemodel.sp,1)
    tex_coeff = np.expand_dims(facemodel.tex,1)
    expression1 = facemodel.exBase.dot(fitted_ep)
	
    gamma = np.expand_dims(facemodel.gamma,0)
	

    vertices = facemodel.meanshape.T + facemodel.idBase.dot(fitted_sp) + expression1
    vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T

    face_norm = Compute_norm(np.expand_dims(vertices,0),facemodel)
    face_norm_r = np.matmul(face_norm,np.expand_dims(fitted_R,0))

    colors = facemodel.meantex.T + facemodel.texBase.dot(tex_coeff)
    colors = np.reshape(colors, [int(3), int(len(colors)/3)], 'F').T
    face_color,lighting = Illumination_layer(np.expand_dims(colors, 0), face_norm_r, gamma)
    colors = face_color[0,:]
    colors = np.minimum(np.maximum(colors, 0), 255)
    transformed_vertices = mesh.transform.similarity_transform(vertices, fitted_s, fitted_R, fitted_t)
    projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection

    h = 512
    w = 512
    colors[mask3,:] = 255.0
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    image = mesh.render.render_colors(image_vertices, triangles-1, colors, h, w)

    return image


def gen_gaosi_filter(r, sigma):
    GaussTemp = np.ones(r*2-1)
    for i in range(0 ,r*2-1):
        GaussTemp[i] = np.exp(-(i-r)**2/(2*sigma**2))/(sigma*np.sqrt(2*3.1415926))
    return GaussTemp


mask3 = np.load('eyemask.npy')
#gaosifilter
r = 3
sigma = 1
gaosifilter = gen_gaosi_filter(r,sigma)
sumgaosi = np.sum(gaosifilter)
gaosifilter = gaosifilter/sumgaosi

trainnum = 'test3'

real_params = opt.real_params_path
realparams = np.load(open(real_params, 'rb'))
realparams = realparams['face']

idparams = realparams[0,71:151]
texparams = realparams[0,151:231]
gammaparams = realparams[0,231:]




# --- 1. load model
facemodel = BFM()
nver = facemodel.idBase.shape[0]/3
ntri = facemodel.tri.shape[0]
n_shape_para = facemodel.idBase.shape[1]
n_exp_para = facemodel.exBase.shape[1]
n_tex_para = facemodel.texBase.shape[1]

kpt_ind = facemodel.keypoints
triangles = facemodel.tri

facemodel.sp = idparams
facemodel.tex = texparams
facemodel.gamma = gammaparams

for i in range(1,realparams.shape[0]+1):
	if i %1000 == 0:
		print(i)
	chi_next = realparams[i-1,:71].copy() 
	if i>3 and i<realparams.shape[0]-2:
		for j in range(6):
			chi_next[j] = np.sum([realparams[i-3,j], realparams[i-2,j], realparams[i-1,j], realparams[i,j], realparams[i+1,j]]*gaosifilter)
	image = render(facemodel,chi_next).astype(np.uint8)

	save_folder = opt.outpath
	if not os.path.exists(save_folder):
		os.mkdir(save_folder)

	io.imsave(os.path.join(save_folder,str("%06d"%(i))+'.jpg'), image)
