# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant juste les tenseurs)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip, numpy, torch
    
if __name__ == '__main__':
	batch_size = 5 # nombre de données lues à chaque fois
	nb_epochs = 10 # nombre de fois que la base de données sera lue
	eta = 0.00001 # taux d'apprentissage
	
	# on lit les données
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

	print("taille de data_train", data_train.shape, " => avec", data_train.shape[0], "comme nombre d'images et chacune avec", data_train.shape[1], "dimentions")
	print("taille de label_train", label_train.shape," => chaque image est representer par 10 chiffres")
	print("taille de data_train", data_test.shape)
	print("taille de label_train", label_test.shape)

	# on initialise le modèle et ses poids
	w = torch.empty((data_train.shape[1],label_train.shape[1]),dtype=torch.float)
	b = torch.empty((1,label_train.shape[1]),dtype=torch.float)
	torch.nn.init.uniform_(w,-0.001,0.001)
	torch.nn.init.uniform_(b,-0.001,0.001)

	print("taille de W", w.shape, "=> on crée",  w.shape[0], "weight pour chaque connection (784 en totale)  entre l'entrer x et notre neuronne")
	print("taille de b", b.shape, "=> meme chose pour le bias")

	nb_data_train = data_train.shape[0]
	nb_data_test = data_test.shape[0]
	indices = numpy.arange(nb_data_train)

	tmp = 0
	for n in range(nb_epochs):
		# on mélange les (indices des) données
		numpy.random.shuffle(indices)
		# on lit toutes les (indices des) données d'apprentissage
		for i in range(0,nb_data_train,batch_size):
			# on récupère les entrées
			x = data_train[indices[i:i+batch_size]]
			# on calcule la sortie du modèle
			y = torch.mm(x,w)+b
			# on regarde les vrais labels
			t = label_train[indices[i:i+batch_size]]
			# on met à jour les poids
			grad = (t-y)
			w += eta * torch.mm(x.T,grad)
			b += eta * grad.sum(axis=0)

			if tmp == 0:
				tmp = 1
				print("taille de l'entrée x", x.shape, "=> chaque fois on prend 5 images")
				print("taille de la sortie y", y.shape)
				print("taille de label t qui doit etre identique a celle de y", t.shape)
				print("taille de grad", grad.shape)

		# test du modèle (on évalue la progression pendant l'apprentissage)
		acc = 0.
		# on lit toutes les donnéees de test
		for i in range(nb_data_test):
			# on récupère l'entrée
			x = data_test[i:i+1]
			# on calcule la sortie du modèle
			y = torch.mm(x,w)+b
			# on regarde le vrai label
			t = label_test[i:i+1]
			# on regarde si la sortie est correcte
			acc += torch.argmax(y,1) == torch.argmax(t,1) # on verfie si indice de y == indice de y
		# on affiche le pourcentage de bonnes réponses
		print(acc/nb_data_test)
