# started Aug 18 2020
# Postprocess Images. Detect cable.

################ IMPORTS ###############################################
from modules import *
from functions import *

################ MAIN ##################################################

## read settings files
test_matrix = pd.read_csv(
	os.path.join("Settings","testMatrix.csv"),sep="	")
idx_cases = np.where(test_matrix.process == "x")
test_matrix = test_matrix.iloc[idx_cases[0]]
ncases = len(test_matrix)
save = pd.read_csv(
	os.path.join("Settings","saveTo.csv"),sep="	").iloc[0].directory
datafolder = pd.read_csv(
	os.path.join("Settings","dataFolder.csv")).iloc[0].directory
if not os.path.isdir(save):
	os.mkdir(save)

## cycle marked cases
pbarc = tqdm(total=ncases)
for c in range(ncases):
	# files, files..
	direct = os.path.join(datafolder,test_matrix.iloc[c].Name)
	# create file to save edges
	savedirect = os.path.join(save,test_matrix.iloc[c].Name)
	if not os.path.isdir(savedirect):
		os.mkdir(savedirect)
	else:
		warnings.warn("A folder in the save directory of %s existed previously, files may be overwritten."%test_matrix.iloc[c].Name)
	# list with all the images in the directory
	fnames = dir_ftype(direct,".tif")
	
	## read processing parameters
	calibration = pd.read_csv(os.path.join(
		"Settings",test_matrix.iloc[c].calibration+".csv"),sep="	")
	roi = np.s_[int(calibration.tl[1]):int(calibration.br[1]),
		int(calibration.tl[0]):int(calibration.br[0])]# crop position
	# read the camera calibration polynomial constants
	Kx = np.array(calibration.Kx)[:10]
	Ky = np.array(calibration.Ky)[:10]
	# for the bead x-positions
	nbead = pd.Series.last_valid_index(calibration.beadl)+1
	beadsx = np.squeeze(
		(calibration.beadl[:nbead],calibration.beadr[:nbead]))
	# read the output file of tune_ghosal to get the parameters for edge
	# detection, edge thresholding and blurring
	pars = read_external_data(
		os.path.join("Settings",str(test_matrix.iloc[c].GPars)))[0]
	K_s, k_min, k_max, l_max, phi_min, outlier_sigma, blur = \
		int(pars[0]),pars[1],pars[2],pars[3],pars[4],pars[5],pars[6]

	## detection and saving	
	# initialization of variables
	end = len(fnames)
	pbar = tqdm(total=end)
	# cycle timesteps
	for i in range(0,end):
		# open original image
		img_o = read_image(fnames[i])
		h, w = img_o.shape
		# crop image
		img_c = crop_image(img_o,roi)
		# blur image
		img_f = blurd_image(img_c,order=1,strength=blur,speed='fast')
		# cycle beads
		for b in range(nbead):
			# isolating bead, take coordinate system into account
			img = img_f[:,int(beadsx[0,b]-calibration.tl[0]):int(beadsx[1,b]-calibration.tl[0])]
			# detect edges: Ghosal
			edg_full, org = ghosal_edge_v2(img,K_s,kmax=k_max,
				kmin=k_min,lmax=l_max,phimin=phi_min,mirror=True)
			# outlier removal
			pts,sig = line_fit(edg_full[:,1],edg_full[:,0])
			ptsy = np.mean(pts[:,1])
			edg_b = edg_full[(edg_full[:,0]<(ptsy+outlier_sigma*sig))
				&(edg_full[:,0]>(ptsy-outlier_sigma*sig)),:]
			# put edges in an array common coord syst (centered)
			edg_b = edg_b+[calibration.tl[1]-h/2,beadsx[0,b]+calibration.tl[0]-w/2]
			# convert to mm
			edg_b[:,0] = poly_2d(edg_b[:,1],edg_b[:,0],Ky)
			edg_b[:,1] = poly_2d(edg_b[:,1],edg_b[:,0],Kx)
			if b==0:
				edg = edg_b
			else:
				edg = np.concatenate((edg,edg_b))
		fn = os.path.split(fnames[i])[1]
		savename = os.path.join(savedirect,fn[:-4]+".txt")
		np.savetxt(savename,edg,delimiter="	")
		if i%100==0:
			pbar.write("%s:	%d of	%d	found %d points"%(test_matrix.iloc[c].Name,i,end,len(edg[:,0])))
		pbar.update(1)
	pbar.close()
	pbarc.update(1)
pbarc.close()

# for windows execution
input('Press ENTER to exit')
	
